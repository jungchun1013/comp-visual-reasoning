"""Add-object hallucination renders (E7, v2 §A1 substrate section).

For direct-query questions ("What color is the large metal cube?"), render a pair:
  base_XXXX.png   — re-render of the ORIGINAL scene (controls render-domain shift)
  added_XXXX.png  — same scene + ONE distractor object

Distractor construction: start from the target's described (filtered) attributes,
flip exactly ONE described attribute (so it no longer satisfies the full referring
expression — the question's answer is unchanged and stays unique, verified via
program execution), and give it a DIFFERENT value on the queried attribute — the
"bait answer". A model whose binding fixates on the described object ignores the
distractor; a bag-of-features binder may retrieve the bait.

Eligibility is restricted to programs made only of scene/filter_*/unique/query_*
(CLEVR direct families) so the simplified program executor is exact.

Run via Blender (CPU ok; reuses render_visual_corruptions helpers):
    BLENDER=${BLENDER_TOOLS_ROOT:-../SteerViT-legacy/tools}/blender/blender  # adjust
    $BLENDER --background --python scripts/analysis/render_add_object.py -- \
        --query-attr color --num-pairs 100 \
        --output-dir outputs/analysis/add_object

Companion eval: scripts/analysis/add_object_eval.py (reads pairs.json).
"""
from __future__ import print_function
import sys, os, json, math, argparse, random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_visual_corruptions as rvc  # helpers + Blender/legacy path setup

# Full CLEVR executor for the relational modes (pure stdlib — safe inside
# Blender's bundled python; the 1-pair smoke verifies the import).
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))
from data.clevr_programs import (  # noqa: E402
    evaluate_answer_strict, find_anchor, find_target, _get_related_objects)
from data.clevr_sampling import RETRIEVAL_CATEGORIES  # noqa: E402

ATTRS = ['color', 'material', 'shape', 'size']
VALUES = {
    'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
    'material': ['rubber', 'metal'],
    'shape': ['cube', 'sphere', 'cylinder'],
    'size': ['large', 'small'],
}
SIMPLE_FNS = ('scene', 'unique')

MODES = ('bait', 'shared_anchor', 'translate')
CATEGORIES = ('attr_query_same', 'attr_query_spatial')
MAX_SPEC_TRIES = 40      # shared_anchor: distractor spec rejection budget
TRANSLATE_MAX = 1.5      # translate: global shift bound (±), objects stay in [-3,3]^2

parser = argparse.ArgumentParser()
parser.add_argument('--scenes', default=os.path.join(
    os.environ.get('CLEVR_ROOT', '/home/jungchun/data/clevr/CLEVR_v1.0'),
    'scenes', 'CLEVR_val_scenes.json'))
parser.add_argument('--questions', default=os.path.join(
    os.environ.get('CLEVR_ROOT', '/home/jungchun/data/clevr/CLEVR_v1.0'),
    'questions', 'CLEVR_val_questions.json'))
parser.add_argument('--output-dir', default='outputs/analysis/add_object')
parser.add_argument('--query-attr', default='color',
                    choices=ATTRS)
parser.add_argument('--num-pairs', default=100, type=int)
parser.add_argument('--width', default=480, type=int)
parser.add_argument('--height', default=320, type=int)
parser.add_argument('--render-samples', default=128, type=int)
parser.add_argument('--base-scene', default=rvc.parser.get_default('base_scene'))
parser.add_argument('--properties-json', default=rvc.parser.get_default('properties_json'))
parser.add_argument('--shape-dir', default=rvc.parser.get_default('shape_dir'))
parser.add_argument('--material-dir', default=rvc.parser.get_default('material_dir'))
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--min-dist', default=1.5, type=float)
parser.add_argument('--mode', default='bait', choices=MODES,
                    help='bait (default, direct-query distractor) or a relational '
                         'shortcut-exclusion mode: shared_anchor / translate')
parser.add_argument('--category', default='attr_query_same', choices=CATEGORIES,
                    help='relational family group (used only by the relational modes)')


def is_simple_query(program, query_fn):
    """Only scene/filter_*/unique chain ending in query_fn — executor-exact."""
    if not program or program[-1]['function'] != query_fn:
        return False
    return all(
        s['function'] in SIMPLE_FNS or s['function'].startswith('filter_')
        for s in program[:-1])


def described_attrs(program):
    """(attr, value) pairs from filter_* steps."""
    out = []
    for s in program:
        fn = s['function']
        if fn.startswith('filter_'):
            out.append((fn.replace('filter_', ''), s['value_inputs'][0]))
    return out


def sample_position(objects, min_dist, rng, tries=80):
    for _ in range(tries):
        x = rng.uniform(-3.0, 3.0)
        y = rng.uniform(-3.0, 3.0)
        if all(math.hypot(x - o['3d_coords'][0], y - o['3d_coords'][1]) >= min_dist
               for o in objects):
            return x, y
    return None


def build_distractor(target, filters, qattr, objects, rng, min_dist):
    """Distractor spec: described attrs minus one flip, bait value on qattr."""
    spec = {a: target[a] for a in ATTRS}
    # attrs not pinned by the referring expression: randomize
    filter_attrs = {a for a, _ in filters}
    for a in ATTRS:
        if a not in filter_attrs and a != qattr:
            spec[a] = rng.choice(VALUES[a])
    # flip exactly one DESCRIBED attribute so the full filter chain excludes it
    flippable = [(a, v) for a, v in filters if a != qattr]
    if not flippable:
        return None
    flip_attr, flip_from = flippable[rng.randrange(len(flippable))]
    choices = [v for v in VALUES[flip_attr] if v != flip_from]
    spec[flip_attr] = rng.choice(choices)
    # bait: different value on the queried attribute
    bait_choices = [v for v in VALUES[qattr] if v != target[qattr]]
    spec[qattr] = rng.choice(bait_choices)

    pos = sample_position(objects, min_dist, rng)
    if pos is None:
        return None
    spec['3d_coords'] = [pos[0], pos[1], 0.0]
    spec['rotation'] = rng.uniform(0, 360)
    spec['flip_attr'] = flip_attr
    return spec


# ── Relational shortcut-exclusion modes (shared_anchor / translate) ──────────

def _relate_direction(program):
    """Direction of the first ``relate`` step, or None (attr_query_same has none)."""
    for s in program:
        if s.get('function', s.get('type')) == 'relate':
            vi = s.get('value_inputs', [])
            return vi[0] if vi else None
    return None


def _target_in_direction(tx, ty, ax, ay, direction):
    """True if target (tx,ty) stands in `direction` w.r.t. an anchor at (ax,ay);
    mirrors data.clevr_programs._get_related_objects."""
    if direction == 'left':
        return tx < ax
    if direction == 'right':
        return tx > ax
    if direction == 'front':
        return ty < ay
    if direction == 'behind':
        return ty > ay
    return False


def sample_position_relational(objects, anchor, target, category, direction,
                               min_dist, rng, tries=80):
    """sample_position + a relational placement predicate.

    attr_query_spatial: target must stand in `direction` w.r.t. the distractor,
      so 'the thing <direction> of X' is ambiguous between anchor and distractor.
    attr_query_same:    distractor must be closer to the target than the anchor is.
    """
    tx, ty = target['3d_coords'][:2]
    ax, ay = anchor['3d_coords'][:2]
    anchor_td = math.hypot(tx - ax, ty - ay)
    for _ in range(tries):
        x = rng.uniform(-3.0, 3.0)
        y = rng.uniform(-3.0, 3.0)
        if not all(math.hypot(x - o['3d_coords'][0], y - o['3d_coords'][1]) >= min_dist
                   for o in objects):
            continue
        if category == 'attr_query_spatial':
            if not _target_in_direction(tx, ty, x, y, direction):
                continue
        else:  # attr_query_same
            if not (math.hypot(tx - x, ty - y) < anchor_td):
                continue
        return x, y
    return None


def build_shared_anchor_distractor(anchor, described, rng):
    """Distractor attrs = anchor's 4 attrs with exactly ONE described attr flipped
    (so the anchor stays the unique referent) and non-described attrs randomized.

    Returns (attr_dict, flip_attr) or (None, None) if no describable attr to flip.
    """
    flippable = [a for a in described if a in ATTRS]
    if not flippable:
        return None, None
    spec = {a: anchor[a] for a in ATTRS}
    for a in ATTRS:
        if a not in described:
            spec[a] = rng.choice(VALUES[a])
    flip_attr = flippable[rng.randrange(len(flippable))]
    spec[flip_attr] = rng.choice([v for v in VALUES[flip_attr] if v != anchor[flip_attr]])
    return spec, flip_attr


def relational_tail_answer(objects, program, distractor):
    """Answer the model would give if it anchored on `distractor` instead of the
    true anchor: substitute the distractor as the anchor, then run the
    relate/same_* tail + target filters + query. Returns None if undefined.

    Downstream add_object_eval.py reads this as `bait_answer` unchanged, so the
    hallucination_rate becomes the anchored-on-distractor rate.
    """
    pivot = None
    for i, step in enumerate(program):
        fn = step.get('function', step.get('type', ''))
        if fn == 'relate' or fn.startswith('same_'):
            pivot = i
            break
    if pivot is None:
        return None
    pfn = program[pivot].get('function', program[pivot].get('type', ''))
    if pfn == 'relate':
        vi = program[pivot].get('value_inputs', [])
        cur = _get_related_objects(distractor, vi[0] if vi else '', objects)
    else:  # same_*
        attr = pfn[len('same_'):]
        val = distractor.get(attr)
        cur = [o for o in objects if o.get(attr) == val]
    for i in range(pivot + 1, len(program)):
        fn = program[i].get('function', program[i].get('type', ''))
        if fn.startswith('filter_'):
            a = fn[len('filter_'):]
            vi = program[i].get('value_inputs', [])
            cur = [o for o in cur if o.get(a) == (vi[0] if vi else '')]
        elif fn == 'unique':
            cur = cur[:1]
        elif fn.startswith('query_'):
            a = fn[len('query_'):]
            return cur[0].get(a) if cur else None
        else:
            break
    return None


def _make_shared_anchor(args, q, objects, program, rng, counts):
    """Return a validated distractor spec (with 3d_coords/rotation/flip_attr) or
    None (incrementing the matching rejection counter)."""
    anchor, described = find_anchor(objects, program)
    target, _ = find_target(objects, program)
    if anchor is None or target is None or not described:
        counts['rejected_setup'] += 1
        return None
    direction = _relate_direction(program)
    last_cause = 'position'
    for _ in range(MAX_SPEC_TRIES):
        attrs, flip_attr = build_shared_anchor_distractor(anchor, described, rng)
        if attrs is None:
            counts['rejected_setup'] += 1
            return None
        pos = sample_position_relational(objects, anchor, target, args.category,
                                         direction, args.min_dist, rng)
        if pos is None:
            last_cause = 'position'
            continue
        spec = dict(attrs)
        spec['3d_coords'] = [pos[0], pos[1], 0.0]
        spec['rotation'] = rng.uniform(0, 360)
        spec['flip_attr'] = flip_attr
        ans = evaluate_answer_strict(objects + [spec], program)
        if ans is None:
            last_cause = 'uniqueness'
            continue
        if ans != q['answer']:
            last_cause = 'answer'
            continue
        return spec
    counts['rejected_' + last_cause] += 1
    return None


def _make_translate(args, q, objects, program, rng, counts, tries=80):
    """Return (translated_objects, (dx,dy)) or None (incrementing a counter)."""
    shift = None
    for _ in range(tries):
        dx = rng.uniform(-TRANSLATE_MAX, TRANSLATE_MAX)
        dy = rng.uniform(-TRANSLATE_MAX, TRANSLATE_MAX)
        if all(-3.0 <= o['3d_coords'][0] + dx <= 3.0
               and -3.0 <= o['3d_coords'][1] + dy <= 3.0 for o in objects):
            shift = (dx, dy)
            break
    if shift is None:
        counts['rejected_position'] += 1
        return None
    dx, dy = shift
    translated = []
    for o in objects:
        t = dict(o)
        c = o['3d_coords']
        t['3d_coords'] = [c[0] + dx, c[1] + dy, c[2]]
        translated.append(t)
    ans = evaluate_answer_strict(translated, program)  # holds by construction; still assert
    if ans is None:
        counts['rejected_uniqueness'] += 1
        return None
    if ans != q['answer']:
        counts['rejected_answer'] += 1
        return None
    return translated, (dx, dy)


def _target_idx(objects, program):
    tgt, _ = find_target(objects, program)
    return next((k for k, o in enumerate(objects) if o is tgt), None)


def run_relational(args, rng, scenes, questions):
    """Driver for the shared_anchor / translate modes (family-selected queries)."""
    families = set(RETRIEVAL_CATEGORIES[args.category])
    eligible = [q for q in questions
                if q.get('question_family_index') in families
                and q.get('program')
                and q['image_filename'] in scenes]
    print(f'[{args.mode}/{args.category}] eligible questions: {len(eligible)}', flush=True)
    rng.shuffle(eligible)

    out_dir = Path(args.output_dir) / f'{args.mode}_{args.category}'
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    color_to_rgba = rvc.load_properties(args)

    counts = {'eligible': len(eligible), 'rejected_setup': 0,
              'rejected_position': 0, 'rejected_answer': 0,
              'rejected_uniqueness': 0}
    pairs = []

    for q in eligible:
        if len(pairs) >= args.num_pairs:
            break
        scene = scenes[q['image_filename']]
        objects = scene['objects']
        program = q['program']
        i = len(pairs)
        base_name = f'base_{args.category}_{i:04d}.png'
        added_name = f'added_{args.category}_{i:04d}.png'
        qattr = program[-1].get('function', '').replace('query_', '')

        if args.mode == 'shared_anchor':
            spec = _make_shared_anchor(args, q, objects, program, rng, counts)
            if spec is None:
                continue
            rvc.setup_scene(args)
            rvc.clear_objects()
            for obj in objects:
                rvc.add_object_from_spec(args, color_to_rgba, obj)
            rvc.render_to_file(str(img_dir / base_name))
            rvc.add_object_from_spec(args, color_to_rgba, spec)
            rvc.render_to_file(str(img_dir / added_name))
            pairs.append({
                'pair_idx': i, 'mode': args.mode, 'category': args.category,
                'original_image': q['image_filename'],
                'base_image': base_name, 'added_image': added_name,
                'question': q['question'], 'answer': q['answer'],
                'query_attr': qattr, 'target_obj_idx': _target_idx(objects, program),
                'bait_answer': relational_tail_answer(objects, program, spec),
                'flip_attr': spec['flip_attr'],
                'distractor': {a: spec[a] for a in ATTRS},
                'n_scene_objects': len(objects),
            })
        else:  # translate
            res = _make_translate(args, q, objects, program, rng, counts)
            if res is None:
                continue
            translated, (dx, dy) = res
            rvc.setup_scene(args)
            rvc.clear_objects()
            for obj in objects:
                rvc.add_object_from_spec(args, color_to_rgba, obj)
            rvc.render_to_file(str(img_dir / base_name))
            rvc.setup_scene(args)
            rvc.clear_objects()
            for obj in translated:
                rvc.add_object_from_spec(args, color_to_rgba, obj)
            rvc.render_to_file(str(img_dir / added_name))
            pairs.append({
                'pair_idx': i, 'mode': args.mode, 'category': args.category,
                'original_image': q['image_filename'],
                'base_image': base_name, 'added_image': added_name,
                'question': q['question'], 'answer': q['answer'],
                'query_attr': qattr, 'target_obj_idx': _target_idx(objects, program),
                'bait_answer': None, 'translate': [dx, dy],
                'n_scene_objects': len(objects),
            })

        if len(pairs) % 10 == 0:
            print(f'  Rendered {len(pairs)}/{args.num_pairs}', flush=True)
            (out_dir / 'pairs.json').write_text(json.dumps(pairs, indent=2))

    (out_dir / 'pairs.json').write_text(json.dumps(pairs, indent=2))
    print(f'[{args.mode}/{args.category}] counts: {counts}', flush=True)
    print(f'Saved {len(pairs)} pairs to {out_dir}/pairs.json', flush=True)


def main():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:] if '--' in argv else []
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)

    with open(args.scenes) as f:
        scenes = {s['image_filename']: s for s in json.load(f)['scenes']}
    with open(args.questions) as f:
        questions = json.load(f)['questions']

    if args.mode != 'bait':
        run_relational(args, rng, scenes, questions)
        return

    qattr = args.query_attr
    query_fn = 'query_' + qattr

    eligible = []
    for q in questions:
        prog = q.get('program', [])
        if not is_simple_query(prog, query_fn):
            continue
        if q['image_filename'] not in scenes:
            continue
        scene = scenes[q['image_filename']]
        tidx = rvc.find_target_object(prog, scene['objects'])
        if tidx is None:
            continue
        eligible.append((q, tidx))
    print(f'Eligible simple {query_fn} questions: {len(eligible)}', flush=True)
    rng.shuffle(eligible)

    out_dir = Path(args.output_dir) / qattr
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    color_to_rgba = rvc.load_properties(args)

    pairs = []
    for q, tidx in eligible:
        if len(pairs) >= args.num_pairs:
            break
        scene = scenes[q['image_filename']]
        objects = scene['objects']
        target = objects[tidx]
        filters = described_attrs(q['program'])
        spec = build_distractor(target, filters, qattr, objects, rng, args.min_dist)
        if spec is None:
            continue
        # verify: answer unchanged and target still unique with distractor present
        aug = objects + [spec]
        ans = rvc.execute_program(q['program'], aug)
        if ans != q['answer']:
            continue

        i = len(pairs)
        rvc.setup_scene(args)
        rvc.clear_objects()
        for obj in objects:
            rvc.add_object_from_spec(args, color_to_rgba, obj)
        base_name = f'base_{qattr}_{i:04d}.png'
        rvc.render_to_file(str(img_dir / base_name))

        rvc.add_object_from_spec(args, color_to_rgba, spec)
        added_name = f'added_{qattr}_{i:04d}.png'
        rvc.render_to_file(str(img_dir / added_name))

        pairs.append({
            'pair_idx': i,
            'original_image': q['image_filename'],
            'base_image': base_name,
            'added_image': added_name,
            'question': q['question'],
            'answer': q['answer'],
            'query_attr': qattr,
            'target_obj_idx': tidx,
            'bait_answer': spec[qattr],
            'flip_attr': spec['flip_attr'],
            'distractor': {a: spec[a] for a in ATTRS},
            'n_scene_objects': len(objects),
        })
        if (i + 1) % 10 == 0:
            print(f'  Rendered {i + 1}/{args.num_pairs}', flush=True)
            (out_dir / 'pairs.json').write_text(json.dumps(pairs, indent=2))

    (out_dir / 'pairs.json').write_text(json.dumps(pairs, indent=2))
    print(f'Saved {len(pairs)} pairs to {out_dir}/pairs.json')


if __name__ == '__main__':
    main()

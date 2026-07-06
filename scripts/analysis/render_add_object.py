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

ATTRS = ['color', 'material', 'shape', 'size']
VALUES = {
    'color': ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow'],
    'material': ['rubber', 'metal'],
    'shape': ['cube', 'sphere', 'cylinder'],
    'size': ['large', 'small'],
}
SIMPLE_FNS = ('scene', 'unique')

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


def main():
    argv = sys.argv
    argv = argv[argv.index('--') + 1:] if '--' in argv else []
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)

    with open(args.scenes) as f:
        scenes = {s['image_filename']: s for s in json.load(f)['scenes']}
    with open(args.questions) as f:
        questions = json.load(f)['questions']

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

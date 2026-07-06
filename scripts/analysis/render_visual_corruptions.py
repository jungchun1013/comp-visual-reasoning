"""Render visually corrupted CLEVR images for causal intervention.

Changes a grounded object's queried attribute in the image while keeping
the question unchanged. This creates (clean_image, corrupt_image) pairs
where the visual content differs but the question is identical.

Run via Blender:
    BLENDER=SteerViT-legacy/tools/blender/blender
    $BLENDER --background --python scripts/analysis/render_visual_corruptions.py -- \
        --corruption-type color --num-pairs 200

Scene JSON provides full object specs (color, shape, material, size, 3d_coords,
rotation) so we can re-render exact scenes with one attribute changed.
"""

from __future__ import print_function
import sys, os, json, math, argparse, random
from pathlib import Path

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError:
    INSIDE_BLENDER = False

# CLEVR generation utilities
CLEVR_GEN_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'SteerViT-legacy', 'tools', 'clevr-dataset-gen', 'image_generation'))
sys.path.insert(0, CLEVR_GEN_DIR)

if INSIDE_BLENDER:
    import utils

# ── CLI ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--scenes', default='/home/jungchun/data/clevr/CLEVR_v1.0/scenes/CLEVR_val_scenes.json')
parser.add_argument('--questions', default='/home/jungchun/data/clevr/CLEVR_v1.0/questions/CLEVR_val_questions.json')
parser.add_argument('--output-dir', default='outputs/analysis/visual_corruptions')
parser.add_argument('--corruption-type', default='color',
                    choices=['color', 'material', 'shape', 'size'])
parser.add_argument('--num-pairs', default=200, type=int)
parser.add_argument('--width', default=480, type=int)
parser.add_argument('--height', default=320, type=int)
parser.add_argument('--render-samples', default=128, type=int)
parser.add_argument('--base-scene',
                    default=os.path.join(CLEVR_GEN_DIR, 'data', 'base_scene.blend'))
parser.add_argument('--properties-json',
                    default=os.path.join(CLEVR_GEN_DIR, 'data', 'properties.json'))
parser.add_argument('--shape-dir',
                    default=os.path.join(CLEVR_GEN_DIR, 'data', 'shapes'))
parser.add_argument('--material-dir',
                    default=os.path.join(CLEVR_GEN_DIR, 'data', 'materials'))
parser.add_argument('--seed', default=42, type=int)

# ── Swap tables ──────────────────────────────────────────────────

COLOR_SWAPS = {
    "red": ["blue", "green"], "blue": ["red", "green"], "green": ["red", "blue"],
    "gray": ["brown"], "brown": ["gray"],
    "purple": ["cyan", "yellow"], "cyan": ["purple", "yellow"], "yellow": ["purple", "cyan"],
}

MATERIAL_MAP = {"rubber": "Rubber", "metal": "MyMetal"}
MATERIAL_SWAPS = {"rubber": ["metal"], "metal": ["rubber"]}

SHAPE_MAP = {"cube": "SmoothCube_v2", "sphere": "Sphere", "cylinder": "SmoothCylinder"}
SHAPE_SWAPS = {"cube": ["sphere", "cylinder"], "sphere": ["cube", "cylinder"], "cylinder": ["cube", "sphere"]}

SIZE_MAP = {"large": 0.7, "small": 0.35}
SIZE_SWAPS = {"large": ["small"], "small": ["large"]}


# ── Blender helpers ──────────────────────────────────────────────

def load_properties(args):
    with open(args.properties_json, 'r') as f:
        props = json.load(f)
    color_to_rgba = {}
    for name, rgb in props['colors'].items():
        color_to_rgba[name] = [float(c) / 255.0 for c in rgb] + [1.0]
    return color_to_rgba


def setup_scene(args):
    bpy.ops.wm.open_mainfile(filepath=args.base_scene)
    render = bpy.context.scene.render
    render.engine = 'CYCLES'
    render.resolution_x = args.width
    render.resolution_y = args.height
    render.resolution_percentage = 100
    bpy.context.scene.cycles.samples = args.render_samples
    bpy.context.scene.cycles.max_bounces = 8
    bpy.context.scene.cycles.min_bounces = 8
    if bpy.context.scene.cycles.device == 'GPU':
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    utils.load_materials(args.material_dir)


def clear_objects():
    for obj in bpy.data.objects:
        if obj.name.startswith('Obj_'):
            bpy.data.objects.remove(obj, do_unlink=True)


def add_object_from_spec(args, color_to_rgba, obj_spec):
    """Add an object to the Blender scene from a scene JSON object spec."""
    shape_blend = SHAPE_MAP[obj_spec['shape']]
    material_blend = MATERIAL_MAP[obj_spec['material']]
    size = obj_spec['size']
    color = obj_spec['color']
    coords = obj_spec['3d_coords']
    rotation = obj_spec['rotation']

    r = SIZE_MAP[size]
    if shape_blend == 'SmoothCube_v2':
        r /= math.sqrt(2)

    utils.add_object(args.shape_dir, shape_blend, r, (coords[0], coords[1]), theta=rotation)
    obj = bpy.context.object
    obj.name = f'Obj_{obj_spec["shape"]}_{color}'
    rgba = color_to_rgba[color]
    utils.add_material(material_blend, Color=rgba)
    return obj


def render_to_file(filepath):
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


# ── Grounding ────────────────────────────────────────────────────

def execute_program(program, objects):
    """Execute a CLEVR functional program on a list of objects.

    Returns the final result (object index, attribute value, bool, or int).
    """
    results = {}
    for i, step in enumerate(program):
        fn = step['function']
        inputs = [results[j] for j in step['inputs']]
        vals = step.get('value_inputs', [])

        if fn == 'scene':
            results[i] = list(range(len(objects)))
        elif fn.startswith('filter_'):
            attr = fn.replace('filter_', '')
            val = vals[0]
            obj_set = inputs[0]
            results[i] = [idx for idx in obj_set if objects[idx].get(attr) == val]
        elif fn == 'unique':
            results[i] = inputs[0][0] if len(inputs[0]) == 1 else inputs[0][0]
        elif fn == 'relate':
            # Not needed for simple queries
            results[i] = inputs[0]
        elif fn.startswith('query_'):
            attr = fn.replace('query_', '')
            obj_idx = inputs[0]
            if isinstance(obj_idx, list):
                obj_idx = obj_idx[0]
            results[i] = objects[obj_idx].get(attr)
        elif fn in ('exist', 'count', 'equal_integer', 'greater_than', 'less_than',
                     'equal_color', 'equal_shape', 'equal_size', 'equal_material',
                     'same_color', 'same_shape', 'same_size', 'same_material',
                     'intersect', 'union'):
            results[i] = inputs[0]  # simplified
        else:
            results[i] = inputs[0] if inputs else None

    return results.get(len(program) - 1)


def find_target_object(program, objects):
    """Find the index of the grounded object from a query program.

    For query_X programs, the target is the object being queried.
    """
    results = {}
    for i, step in enumerate(program):
        fn = step['function']
        inputs = [results[j] for j in step['inputs']]
        vals = step.get('value_inputs', [])

        if fn == 'scene':
            results[i] = list(range(len(objects)))
        elif fn.startswith('filter_'):
            attr = fn.replace('filter_', '')
            val = vals[0]
            obj_set = inputs[0]
            if not isinstance(obj_set, list):
                obj_set = [obj_set] if obj_set is not None else []
            results[i] = [idx for idx in obj_set if objects[idx].get(attr) == val]
        elif fn == 'unique':
            obj_set = inputs[0]
            if isinstance(obj_set, list):
                results[i] = obj_set[0] if obj_set else None
            else:
                results[i] = obj_set
        elif fn.startswith('query_'):
            obj_idx = inputs[0]
            if isinstance(obj_idx, list):
                obj_idx = obj_idx[0] if obj_idx else None
            return obj_idx
        elif fn == 'relate':
            # Spatial relation — return related object indices
            obj_idx = inputs[0]
            if isinstance(obj_idx, list):
                obj_idx = obj_idx[0] if obj_idx else 0
            rel = vals[0] if vals else 'left'
            scene_rels = None
            # We don't have scene relationships here, skip
            results[i] = list(range(len(objects)))
        else:
            results[i] = inputs[0] if inputs else None

    return None


# ── Main ─────────────────────────────────────────────────────────

def main():
    # Parse args after '--' (Blender passes its own args before --)
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)

    random.seed(args.seed)

    # Load data
    print(f"Loading scenes: {args.scenes}", flush=True)
    with open(args.scenes) as f:
        scenes_data = json.load(f)
    scenes = {s['image_filename']: s for s in scenes_data['scenes']}

    print(f"Loading questions: {args.questions}", flush=True)
    with open(args.questions) as f:
        questions_data = json.load(f)

    # Filter questions: only query_<corruption_type> with unique grounding
    ctype = args.corruption_type
    query_fn = f'query_{ctype}'

    swap_table = {
        'color': COLOR_SWAPS,
        'material': MATERIAL_SWAPS,
        'shape': SHAPE_SWAPS,
        'size': SIZE_SWAPS,
    }[ctype]

    eligible = []
    for q in questions_data['questions']:
        prog = q.get('program', [])
        if not prog:
            continue
        # Must end with query_<attr>
        if prog[-1]['function'] != query_fn:
            continue
        # Must have scene
        img = q['image_filename']
        if img not in scenes:
            continue
        scene = scenes[img]
        # Find target object
        target_idx = find_target_object(prog, scene['objects'])
        if target_idx is None:
            continue
        target_obj = scene['objects'][target_idx]
        # Must have a swap available
        attr_val = target_obj[ctype]
        if attr_val not in swap_table:
            continue

        eligible.append({
            'question_index': q['question_index'],
            'question': q['question'],
            'answer': q['answer'],
            'image_filename': img,
            'target_obj_idx': target_idx,
            'target_attr': attr_val,
        })

    print(f"Eligible questions for {ctype}: {len(eligible)}", flush=True)
    random.shuffle(eligible)
    selected = eligible[:args.num_pairs]
    print(f"Selected: {len(selected)}", flush=True)

    # Setup output
    out_dir = Path(args.output_dir) / ctype
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    color_to_rgba = load_properties(args)

    # Render pairs
    pairs = []
    for i, sample in enumerate(selected):
        scene = scenes[sample['image_filename']]
        objects = scene['objects']
        target_idx = sample['target_obj_idx']
        original_val = sample['target_attr']
        corrupt_val = random.choice(swap_table[original_val])

        # Render corrupt image: same scene, but target object has swapped attr
        setup_scene(args)
        clear_objects()

        for oi, obj in enumerate(objects):
            spec = dict(obj)  # copy
            if oi == target_idx:
                spec[ctype] = corrupt_val
            add_object_from_spec(args, color_to_rgba, spec)

        corrupt_filename = f'vis_{ctype}_{i:04d}.png'
        corrupt_path = str(img_dir / corrupt_filename)
        render_to_file(corrupt_path)

        pairs.append({
            'pair_idx': i,
            'original_image': sample['image_filename'],
            'corrupt_image': corrupt_filename,
            'question': sample['question'],
            'answer': sample['answer'],
            'target_obj_idx': target_idx,
            'original_attr': original_val,
            'corrupt_attr': corrupt_val,
            'corruption_type': ctype,
        })

        if (i + 1) % 10 == 0:
            print(f'  Rendered {i+1}/{len(selected)}', flush=True)
            # Incremental save
            meta_path = out_dir / 'pairs.json'
            with open(str(meta_path), 'w') as f:
                json.dump(pairs, f, indent=2)

    # Final save
    meta_path = out_dir / 'pairs.json'
    with open(str(meta_path), 'w') as f:
        json.dump(pairs, f, indent=2)
    print(f"\nSaved {len(pairs)} pairs to {meta_path}")
    print("Done.")


if __name__ == '__main__':
    main()

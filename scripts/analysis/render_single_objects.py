"""Render single-object CLEVR scenes for DINOv2 representation analysis.

Generates images of every possible CLEVR object type (96 combinations of
color × shape × material × size), each with random position and rotation.

Run via Blender:
    BLENDER=SteerViT-legacy/tools/blender/blender
    $BLENDER --background --python scripts/analysis/render_single_objects.py -- \
        --output-dir outputs/analysis/single_objects \
        --n-per-type 30 --render-samples 64 --seed 42
"""

from __future__ import print_function
import sys, os, json, math, argparse, random, itertools
from pathlib import Path

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError:
    INSIDE_BLENDER = False

# CLEVR generation utilities — the single remaining legacy runtime dependency
# (docs/legacy-reference.md §6). Override with BLENDER_TOOLS_ROOT.
_BLENDER_TOOLS_ROOT = os.environ.get(
    'BLENDER_TOOLS_ROOT',
    os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'SteerViT-legacy', 'tools', 'clevr-dataset-gen')))
CLEVR_GEN_DIR = os.path.join(_BLENDER_TOOLS_ROOT, 'image_generation')
sys.path.insert(0, CLEVR_GEN_DIR)

if INSIDE_BLENDER:
    import utils

# ── CLI ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', default='outputs/analysis/single_objects')
parser.add_argument('--n-per-type', default=30, type=int,
                    help='Number of images per object type')
parser.add_argument('--width', default=480, type=int)
parser.add_argument('--height', default=320, type=int)
parser.add_argument('--render-samples', default=64, type=int)
parser.add_argument('--base-scene',
                    default=os.path.join(CLEVR_GEN_DIR, 'data', 'base_scene.blend'))
parser.add_argument('--properties-json',
                    default=os.path.join(CLEVR_GEN_DIR, 'data', 'properties.json'))
parser.add_argument('--shape-dir',
                    default=os.path.join(CLEVR_GEN_DIR, 'data', 'shapes'))
parser.add_argument('--material-dir',
                    default=os.path.join(CLEVR_GEN_DIR, 'data', 'materials'))
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--max-images', default=0, type=int,
                    help='Stop after this many images (0 = all); for smoke tests')
parser.add_argument('--num-distractors', default=0, type=int,
                    help='Random distractor objects added per scene. Each '
                         'distractor differs from the target in >=2 attributes '
                         'so a 3-attribute description of the target stays '
                         'unique for every query attribute.')
parser.add_argument('--target-size', default=None, choices=['large', 'small'],
                    help='Fix the target size (restricts the enumerated types)')
parser.add_argument('--distractor-size', default=None, choices=['large', 'small'],
                    help='Fix every distractor size (e.g. one-large-one-small '
                         'scenes: --target-size large --distractor-size small)')
parser.add_argument('--distinct-colors', action='store_true',
                    help='Skip gray targets; every distractor colour is non-gray '
                         'and distinct from the target and the other distractors')

# ── Constants ────────────────────────────────────────────────────

SHAPE_MAP = {"cube": "SmoothCube_v2", "sphere": "Sphere", "cylinder": "SmoothCylinder"}
MATERIAL_MAP = {"rubber": "Rubber", "metal": "MyMetal"}
SIZE_MAP = {"large": 0.7, "small": 0.35}

COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
SHAPES = ["cube", "sphere", "cylinder"]
MATERIALS = ["rubber", "metal"]
SIZES = ["large", "small"]

# Position range — CLEVR default ground plane
POS_MIN, POS_MAX = -3.0, 3.0


def sample_distractor(target_attrs, rng, size=None, exclude_colors=()):
    """Distractor attrs differing from the target in >=2 of the 4 attributes."""
    while True:
        attrs = (rng.choice(COLORS), rng.choice(SHAPES),
                 rng.choice(MATERIALS), size or rng.choice(SIZES))
        if attrs[0] in exclude_colors:
            continue
        if sum(a != b for a, b in zip(attrs, target_attrs)) >= 2:
            return attrs


def sample_position(existing, r, rng, min_margin=0.4, max_tries=100):
    """Position with no overlap against existing [(x, y, r), ...]."""
    for _ in range(max_tries):
        x = rng.uniform(POS_MIN, POS_MAX)
        y = rng.uniform(POS_MIN, POS_MAX)
        if all(math.hypot(x - ex, y - ey) >= r + er + min_margin
               for ex, ey, er in existing):
            return x, y
    raise RuntimeError("could not place object without overlap")


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


def add_single_object(args, color_to_rgba, color, shape, material, size, x, y, rotation):
    """Add a single CLEVR object to the scene."""
    shape_blend = SHAPE_MAP[shape]
    material_blend = MATERIAL_MAP[material]

    r = SIZE_MAP[size]
    if shape_blend == 'SmoothCube_v2':
        r /= math.sqrt(2)

    utils.add_object(args.shape_dir, shape_blend, r, (x, y), theta=rotation)
    obj = bpy.context.object
    obj.name = f'Obj_{shape}_{color}'
    rgba = color_to_rgba[color]
    utils.add_material(material_blend, Color=rgba)
    return obj


def render_to_file(filepath):
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


# ── Main ─────────────────────────────────────────────────────────

def main():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)

    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    color_to_rgba = load_properties(args)

    # Enumerate all 96 object types
    all_types = list(itertools.product(COLORS, SHAPES, MATERIALS, SIZES))
    if args.target_size:
        all_types = [t for t in all_types if t[3] == args.target_size]
    if args.distinct_colors:
        all_types = [t for t in all_types if t[0] != 'gray']
    total = len(all_types) * args.n_per_type
    print(f"Rendering {len(all_types)} object types × {args.n_per_type} = {total} images")

    metadata = []
    idx = 0

    for color, shape, material, size in all_types:
        for rep in range(args.n_per_type):
            # Target: random position and rotation
            placed = []
            x, y = sample_position(placed, SIZE_MAP[size], random)
            placed.append((x, y, SIZE_MAP[size]))
            rotation = random.uniform(0, 2 * math.pi)

            # Distractors: attrs >=2 away from target, non-overlapping
            distractors = []
            for _ in range(args.num_distractors):
                d_color, d_shape, d_material, d_size = sample_distractor(
                    (color, shape, material, size), random,
                    size=args.distractor_size,
                    exclude_colors=(color, 'gray', *[d['color'] for d in distractors])
                    if args.distinct_colors else ())
                dx, dy = sample_position(placed, SIZE_MAP[d_size], random)
                placed.append((dx, dy, SIZE_MAP[d_size]))
                distractors.append({
                    'color': d_color, 'shape': d_shape,
                    'material': d_material, 'size': d_size,
                    'x': round(dx, 4), 'y': round(dy, 4),
                    'rotation': round(random.uniform(0, 2 * math.pi), 4),
                })

            # Setup fresh scene, add objects, render
            setup_scene(args)
            clear_objects()
            add_single_object(args, color_to_rgba, color, shape, material, size, x, y, rotation)
            for d in distractors:
                add_single_object(args, color_to_rgba, d['color'], d['shape'],
                                  d['material'], d['size'], d['x'], d['y'],
                                  d['rotation'])

            fname = f'obj_{idx:04d}.png'
            render_to_file(str(img_dir / fname))

            entry = {
                'index': idx,
                'filename': fname,
                'color': color,
                'shape': shape,
                'material': material,
                'size': size,
                'x': round(x, 4),
                'y': round(y, 4),
                'rotation': round(rotation, 4),
            }
            if distractors:
                entry['distractors'] = distractors
            metadata.append(entry)

            idx += 1
            if idx % 100 == 0:
                print(f"  [{idx}/{total}] rendered")
            if args.max_images and idx >= args.max_images:
                break
        if args.max_images and idx >= args.max_images:
            break

    # Save metadata
    meta_path = out_dir / 'metadata.json'
    with open(str(meta_path), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Done. {idx} images rendered to {img_dir}")
    print(f"Metadata: {meta_path}")


if __name__ == '__main__':
    if not INSIDE_BLENDER:
        print("ERROR: This script must be run inside Blender.")
        print("Usage: blender --background --python render_single_objects.py -- [args]")
        sys.exit(1)
    main()

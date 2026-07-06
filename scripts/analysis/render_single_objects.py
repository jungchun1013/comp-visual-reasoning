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
    total = len(all_types) * args.n_per_type
    print(f"Rendering {len(all_types)} object types × {args.n_per_type} = {total} images")

    metadata = []
    idx = 0

    for color, shape, material, size in all_types:
        for rep in range(args.n_per_type):
            # Random position and rotation
            x = random.uniform(POS_MIN, POS_MAX)
            y = random.uniform(POS_MIN, POS_MAX)
            rotation = random.uniform(0, 2 * math.pi)

            # Setup fresh scene, add object, render
            setup_scene(args)
            clear_objects()
            add_single_object(args, color_to_rgba, color, shape, material, size, x, y, rotation)

            fname = f'obj_{idx:04d}.png'
            render_to_file(str(img_dir / fname))

            metadata.append({
                'index': idx,
                'filename': fname,
                'color': color,
                'shape': shape,
                'material': material,
                'size': size,
                'x': round(x, 4),
                'y': round(y, 4),
                'rotation': round(rotation, 4),
            })

            idx += 1
            if idx % 100 == 0:
                print(f"  [{idx}/{total}] rendered")

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

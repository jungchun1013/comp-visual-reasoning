"""Render single-object CLEVR-like images using pyrender for t-SNE analysis.

Matches CLEVR's camera angle, object sizes, exact color palette, and
rubber/metal material distinction as closely as pyrender (rasterization) allows.
"""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import json
import random
import argparse
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image

# Exact CLEVR color palette (from clevr-dataset-gen/image_generation/data/properties.json)
COLORS = {
    "gray":   [87/255, 87/255, 87/255],
    "red":    [173/255, 35/255, 35/255],
    "blue":   [42/255, 75/255, 215/255],
    "green":  [29/255, 105/255, 20/255],
    "brown":  [129/255, 74/255, 25/255],
    "purple": [129/255, 38/255, 192/255],
    "cyan":   [41/255, 208/255, 208/255],
    "yellow": [255/255, 238/255, 51/255],
}

SHAPES = ["sphere", "cube", "cylinder"]
MATERIALS = ["rubber", "metal"]
# CLEVR: small=0.35 Blender units, large=0.7
SIZES = {"small": 0.35, "large": 0.7}

IMG_W, IMG_H = 480, 320


def make_mesh(shape, radius, color_rgb, material):
    if shape == "sphere":
        mesh_tri = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    elif shape == "cube":
        mesh_tri = trimesh.creation.box(extents=[radius * 2 * 0.707] * 3)
        rot = trimesh.transformations.rotation_matrix(np.pi / 6, [0, 0, 1])
        mesh_tri.apply_transform(rot)
    elif shape == "cylinder":
        mesh_tri = trimesh.creation.cylinder(
            radius=radius, height=radius * 2, sections=64
        )

    if material == "metal":
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color_rgb + [1.0],
            metallicFactor=1.0,
            roughnessFactor=0.05,
        )
    else:
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color_rgb + [1.0],
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
    return pyrender.Mesh.from_trimesh(mesh_tri, material=mat, smooth=True)


def _mat44(R=None, t=None):
    m = np.eye(4)
    if R is not None:
        m[:3, :3] = R
    if t is not None:
        m[:3, 3] = t
    return m


def _look_at(eye, target, up):
    eye, target, up = [np.array(v, float) for v in (eye, target, up)]
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0] = r
    m[:3, 1] = u
    m[:3, 2] = -f
    m[:3, 3] = eye
    return m


def build_scene(obj_attrs, position):
    scene = pyrender.Scene(
        bg_color=[0.54, 0.57, 0.62, 1.0],
        ambient_light=[0.08, 0.08, 0.08],
    )

    # Ground plane — CLEVR-like gray
    ground = trimesh.creation.box(extents=[20, 20, 0.02])
    ground_mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.40, 0.40, 0.40, 1.0],
        metallicFactor=0.0, roughnessFactor=0.95,
    )
    scene.add(pyrender.Mesh.from_trimesh(ground, material=ground_mat),
              pose=_mat44(t=[0, 0, -0.01]))

    # Object
    r = SIZES[obj_attrs["size"]]
    obj_mesh = make_mesh(obj_attrs["shape"], r, COLORS[obj_attrs["color"]], obj_attrs["material"])
    obj_z = r
    scene.add(obj_mesh, pose=_mat44(t=[position[0], position[1], obj_z]))

    # Camera — CLEVR-like elevated angle
    cam = pyrender.PerspectiveCamera(yfov=np.radians(25))
    cam_pose = _look_at(eye=[7.36, -6.93, 5.07], target=[0, 0, 0], up=[0, 0, 1])
    scene.add(cam, pose=cam_pose)

    # Key light — warm, from upper right
    key_pose = _look_at(eye=[4, -4, 8], target=[0, 0, 0], up=[0, 0, 1])
    scene.add(pyrender.DirectionalLight(color=[1.0, 0.95, 0.90], intensity=2.5),
              pose=key_pose)

    # Fill light — cool, from left
    fill_pose = _look_at(eye=[-6, -2, 4], target=[0, 0, 0], up=[0, 0, 1])
    scene.add(pyrender.DirectionalLight(color=[0.85, 0.90, 1.0], intensity=1.2),
              pose=fill_pose)

    # Back/rim light
    back_pose = _look_at(eye=[-1, 6, 5], target=[0, 0, 0], up=[0, 0, 1])
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8),
              pose=back_pose)

    return scene


def render_dataset(out_dir, n_images=500, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    renderer = pyrender.OffscreenRenderer(IMG_W, IMG_H)
    scenes_meta = []

    for i in range(n_images):
        attrs = {
            "color": random.choice(list(COLORS.keys())),
            "shape": random.choice(SHAPES),
            "material": random.choice(MATERIALS),
            "size": random.choice(list(SIZES.keys())),
        }
        x = random.uniform(-2.5, 2.5)
        y = random.uniform(-2.0, 2.0)

        scene = build_scene(attrs, [x, y])
        color_img, _ = renderer.render(scene)

        fname = f"single_{i:06d}.png"
        Image.fromarray(color_img).save(out_dir / "images" / fname)

        scenes_meta.append({
            "image_index": i,
            "image_filename": fname,
            "objects": [{**attrs, "3d_coords": [x, y, 0],
                         "pixel_coords": [IMG_W // 2, IMG_H // 2, 8.0]}],
        })

        if (i + 1) % 100 == 0:
            print(f"  Rendered {i + 1}/{n_images}", flush=True)

    renderer.delete()

    with open(out_dir / "scenes.json", "w") as f:
        json.dump({"scenes": scenes_meta}, f, indent=2)
    print(f"Done: {n_images} images -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/clevr_single_object")
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    render_dataset(args.out_dir, args.n_images, args.seed)

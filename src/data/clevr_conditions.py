"""CLEVR condition hierarchy checking.

Ported from SteerViT-legacy/exp_vqa/analysis/utils/clevr_condition_checker.py

Direct (5 conditions):
  0: Feature extraction — each described attr exists independently
  1: Feature binding — all described attrs co-occur on one object
  2: Object grounding — full 4-attr object exists
  3: Position indexing — 4-attr match + pixel overlap
  4: Answer match — same answer from program execution

Spatial (7 conditions):
  0: Anchor feature binding
  1: Target feature binding
  2: Anchor object grounding
  3: Target object grounding
  4: Anchor position indexing
  5: Target position indexing
  6: Answer match
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from data.clevr_programs import evaluate_answer

ATTR_KEYS = ("color", "shape", "material", "size")

_tab10 = plt.cm.tab10.colors

ALL_COND_NAMES = [
    "Feature extraction", "Feature binding",
    "Object grounding", "Position ID", "Answer match",
]
COND_COLORS = [_tab10[2], _tab10[0], _tab10[1], _tab10[4], _tab10[3]]

SPATIAL_COND_NAMES = [
    "Anchor feature binding", "Target feature binding",
    "Anchor object grounding", "Target object grounding",
    "Anchor position ID", "Target position ID", "Answer match",
]
SPATIAL_COND_COLORS = [
    _tab10[0], _tab10[9], _tab10[1], _tab10[6],
    _tab10[4], _tab10[5], _tab10[3],
]

FAMILY_SUBCATEGORY = {
    86: "attr_query_direct", 87: "attr_query_direct",
    88: "attr_query_direct", 89: "attr_query_direct",
    52: "attr_query_same", 53: "attr_query_same", 54: "attr_query_same",
    55: "attr_query_same", 56: "attr_query_same", 57: "attr_query_same",
    58: "attr_query_same", 59: "attr_query_same", 60: "attr_query_same",
    61: "attr_query_same", 62: "attr_query_same", 63: "attr_query_same",
    76: "attr_query_spatial", 74: "attr_query_spatial",
    75: "attr_query_spatial", 77: "attr_query_spatial",
    80: "attr_query_spatial", 81: "attr_query_spatial",
}

SPATIAL_SUBCATEGORIES = {"attr_query_same", "attr_query_spatial"}


# ── Pixel overlap ────────────────────────────────────────────────

_FOCAL_PX = 35.0 / 32.0 * 480
_SIZE_RADIUS_3D = {"large": 0.7, "small": 0.35}

def _position_overlap(obj_coords, obj_size, ref_coords, ref_size):
    r_obj = _FOCAL_PX * _SIZE_RADIUS_3D.get(obj_size, 0.5) / obj_coords[2]
    r_ref = _FOCAL_PX * _SIZE_RADIUS_3D.get(ref_size, 0.5) / ref_coords[2]
    dist_sq = (obj_coords[0] - ref_coords[0]) ** 2 + (obj_coords[1] - ref_coords[1]) ** 2
    threshold = (r_obj + r_ref) / 2
    return dist_sq <= threshold ** 2


# ── Individual conditions ────────────────────────────────────────

def _feature_extraction(db_objs, all_filter_attrs):
    return all(
        any(o.get(attr_type) == val for o in db_objs)
        for attr_type, val in all_filter_attrs
    )

def _feature_binding(db_objs, described_attrs):
    if not described_attrs:
        return False
    return any(
        all(o.get(k) == v for k, v in described_attrs.items())
        for o in db_objs
    )

def _object_grounding(db_objs, attrs_4):
    if attrs_4 is None:
        return False
    return any(
        all(o.get(k) == attrs_4.get(k) for k in ATTR_KEYS)
        for o in db_objs
    )

def _position_indexing(db_objs, anchor_obj):
    if anchor_obj is None:
        return False
    ref_coords = anchor_obj.get("pixel_coords")
    ref_size = anchor_obj.get("size")
    if ref_coords is None or ref_size is None:
        return False
    for o in db_objs:
        if not all(o.get(k) == anchor_obj.get(k) for k in ATTR_KEYS):
            continue
        oc = o.get("pixel_coords")
        os = o.get("size")
        if oc is not None and os is not None and _position_overlap(oc, os, ref_coords, ref_size):
            return True
    return False

def _answer_match(db_objs, program, gt_answer):
    ans = evaluate_answer(db_objs, program)
    if ans is None:
        return False
    return str(ans).lower() == str(gt_answer).lower()


# ── Public API ───────────────────────────────────────────────────

def check_all_conditions(db_objs, all_filter_attrs, described_attrs,
                         anchor_4attrs, anchor_obj, program, gt_answer):
    """Check 5 conditions (direct). Returns list of 5 bools."""
    return [
        _feature_extraction(db_objs, all_filter_attrs),
        _feature_binding(db_objs, described_attrs),
        _object_grounding(db_objs, anchor_4attrs),
        _position_indexing(db_objs, anchor_obj),
        _answer_match(db_objs, program, gt_answer),
    ]

def check_spatial_conditions(db_objs, all_filter_attrs,
                             anchor_described, anchor_4attrs, anchor_obj,
                             target_described, target_4attrs, target_obj,
                             program, gt_answer):
    """Check 7 conditions (spatial/same). Returns list of 7 bools."""
    return [
        _feature_binding(db_objs, anchor_described),
        _feature_binding(db_objs, target_described),
        _object_grounding(db_objs, anchor_4attrs),
        _object_grounding(db_objs, target_4attrs),
        _position_indexing(db_objs, anchor_obj),
        _position_indexing(db_objs, target_obj),
        _answer_match(db_objs, program, gt_answer),
    ]


def check_pos_only(db_objs, query_info):
    """Position-overlap-only condition for anchor and target (NO attribute match).

    For each ref in (anchor_obj, target_obj), returns True iff ANY db_obj's
    (pixel_coords, size) overlaps the ref's per ``_position_overlap`` — pure
    geometry, no attribute comparison. This is the pos_only RDM used to test
    position-as-indexing vs a position shortcut (experiment 三): a curve that
    rises from position overlap alone, without any attribute binding.

    Returns [anchor_pos_only, target_pos_only]; False for a ref that is None
    or lacks pixel_coords/size.
    """
    out = []
    for ref in (query_info.get("anchor_obj"), query_info.get("target_obj")):
        if ref is None:
            out.append(False)
            continue
        ref_coords = ref.get("pixel_coords")
        ref_size = ref.get("size")
        if ref_coords is None or ref_size is None:
            out.append(False)
            continue
        hit = False
        for o in db_objs:
            oc = o.get("pixel_coords")
            os = o.get("size")
            if oc is not None and os is not None and \
                    _position_overlap(oc, os, ref_coords, ref_size):
                hit = True
                break
        out.append(hit)
    return out


def extract_query_info(query_scene_objs, program, gt_answer):
    """Extract all info needed for condition checking from a query."""
    from data.clevr_programs import (
        extract_all_filter_attrs, extract_described_attrs,
        find_anchor, find_target,
    )
    all_filter_attrs = extract_all_filter_attrs(program)
    described_attrs = extract_described_attrs(program)
    anchor_obj, anchor_described = find_anchor(query_scene_objs, program)
    anchor_4attrs = {k: anchor_obj.get(k) for k in ATTR_KEYS} if anchor_obj else None
    target_obj, target_described = find_target(query_scene_objs, program)
    target_4attrs = {k: target_obj.get(k) for k in ATTR_KEYS} if target_obj else None
    return {
        "all_filter_attrs": all_filter_attrs,
        "described_attrs": described_attrs,
        "anchor_obj": anchor_obj, "anchor_4attrs": anchor_4attrs,
        "anchor_described": anchor_described,
        "target_obj": target_obj, "target_4attrs": target_4attrs,
        "target_described": target_described,
        "program": program, "gt_answer": gt_answer,
    }


def label_db_image(db_scene_objs, query_info, subcategory):
    """Label a DB image. Returns (conditions, highest_index)."""
    is_spatial = subcategory in SPATIAL_SUBCATEGORIES
    if is_spatial:
        conditions = check_spatial_conditions(
            db_scene_objs, query_info["all_filter_attrs"],
            query_info["anchor_described"], query_info["anchor_4attrs"],
            query_info["anchor_obj"],
            query_info["target_described"], query_info["target_4attrs"],
            query_info["target_obj"],
            query_info["program"], query_info["gt_answer"],
        )
    else:
        conditions = check_all_conditions(
            db_scene_objs, query_info["all_filter_attrs"],
            query_info["described_attrs"], query_info["anchor_4attrs"],
            query_info["anchor_obj"],
            query_info["program"], query_info["gt_answer"],
        )
    highest = max((i for i in range(len(conditions)) if conditions[i]), default=-1)
    return conditions, highest

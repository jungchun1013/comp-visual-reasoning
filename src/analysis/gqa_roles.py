"""GQA scene graph -> role records for X23 (CLEVR mechanism observations on real
images). CPU only; no model is loaded here.

For one question type (direct / spatial / same) this builds the same record schema
as prepare_relational() in scripts/analysis/patch_language_condition.py, so the
extraction, transplant and head-ablation code can run on GQA unchanged:

  record["objects"]  = [anchor, target, third object] dicts (name, color, attributes,
                       box_xywh in original pixels, patches)
  record["A"], ["T"], ["D"] = 0, 1, 2   (direct: objects = [referent, distractor],
                                          A = T = 0, D = 1)
  record["questions"] = {"c1": original GQA question, "c2": counterfactual[, "c3"]}
  record["referent_words"], ["answers"], ["centroids_row_col"], ["n_patches"],
  record["bg_sample"] (patches inside no annotated box = background (b)),
  record["other_obj_patches"] (patches of non-role annotated objects = background (a))
  owners[i] : (grid*grid,) int8, 1 = anchor, 2 = target, 3 = third object, 0 = rest.

Counterfactual questions (c2, c3) are NATURAL GQA questions taken from the
`<split>_all_questions.json` pool of the same image (spatial c2: same anchor,
opposite relation, same class word and query, other answer; spatial c3: same
relation / class / query, same answer object, other anchor; direct c2: other
referent, same query, other answer; same-as c2: roles swapped). Uniqueness of the
answer is then guaranteed by GQA's generator and the wording is in-distribution.
Only same-as falls back to a rewritten question when no natural one exists
(record["c2_source"] = "constructed").

Filtering follows docs/gqa_relational_experiment_design.md appendix A. Every rule
is applied in order and the number of items it removes is recorded (funnel); the
first failing rule of every rejected question is kept in funnel.json. Nothing here
depends on a model output: the population is S_eligible.
"""
from __future__ import annotations

import collections
import html
import json
import re
from pathlib import Path

import numpy as np

LR_RELATIONS = {"to the left of": "left", "to the right of": "right"}
LR_OPPOSITE = {"to the left of": "to the right of", "to the right of": "to the left of"}
# GQA colour attributes (attribute ontology, colour group); compound names first so
# that "light blue" is not read as "blue".
COLOR_WORDS = ["light blue", "dark blue", "light brown", "dark brown", "cream colored",
               "white", "black", "blue", "red", "green", "brown", "gray", "yellow", "orange",
               "pink", "purple", "silver", "tan", "gold", "beige", "cream", "maroon", "teal",
               "khaki", "blond", "brunette"]
MATERIAL_WORDS = {"wood", "wooden", "metal", "plastic", "glass", "leather", "concrete", "brick",
                  "stone", "paper", "cloth", "fabric", "steel", "ceramic", "cardboard", "rubber",
                  "denim", "wool", "cotton", "porcelain", "marble", "iron", "silver", "gold"}
SIZE_WORDS = {"small", "large", "little", "big", "tiny", "huge", "short", "tall", "long"}
ATTRIBUTE_QUERIES = {"color", "material", "size"}
CLEVR_COLORS = {"red", "blue", "green", "yellow", "gray", "brown", "purple", "cyan"}

GEOMETRY_PRESETS = {
    # cover: fraction of a patch cell a box must cover for the patch to count as the
    # object's; min_patches: per role; max_frac: box area / image area; iou_tol: max
    # IoU between role boxes.
    "strict": dict(cover=0.5, min_patches=4, max_frac=0.30, iou_tol=0.0),
    "relaxed": dict(cover=0.3, min_patches=2, max_frac=0.50, iou_tol=0.05),
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_questions(root, split="val"):
    with open(Path(root) / "questions" / f"{split}_balanced_questions.json") as f:
        return json.load(f)


def load_scene_graphs(root, split="val"):
    with open(Path(root) / f"{split}_sceneGraphs.json") as f:
        return json.load(f)


def answer_vocab(root, cache_dir):
    """Model answer vocabulary (top-1500 train answers), cached as json."""
    cache = Path(cache_dir) / "answer_vocab.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    from tasks.decoder import build_gqa_decoder_vocab
    vocab = build_gqa_decoder_vocab(root)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(vocab, f)
    return vocab


_ARG_ID = re.compile(r"\((\d+)\)")


def parse_program(q):
    """Flat view of a GQA program: ops, select (name, id), relate (class, relation,
    target id), filters, query argument."""
    ops = [s["operation"] for s in q["semantic"]]
    out = {"ops": ops, "select": None, "relate": None, "filters": [], "query": None}
    for s in q["semantic"]:
        op, arg = s["operation"], s["argument"]
        if op == "select" and out["select"] is None:
            m = _ARG_ID.search(arg)
            out["select"] = (arg.split(" (")[0].strip(), m.group(1) if m else None)
        elif op == "relate" and out["relate"] is None:
            m = _ARG_ID.search(arg)
            parts = arg.split(" (")[0].split(",")
            if len(parts) == 3:
                out["relate"] = (parts[0].strip(), parts[1].strip(), m.group(1) if m else None)
        elif op.startswith("filter"):
            out["filters"].append((op, arg))
        elif op == "query":
            out["query"] = arg
    return out


def referent_word(q, oid):
    """The word of the question text that names object `oid` (head noun = last word of
    the annotated span), from GQA's annotations["question"] ({"3": id} or {"3:5": id});
    falls back to the select argument's last word. Used for the GCA attention-to-
    referent measurement, so it must be a word that occurs in the question."""
    words = q["question"].rstrip("?").split()
    spans = [k for k, v in q.get("annotations", {}).get("question", {}).items() if v == oid]
    for k in spans:
        a, b = (int(x) for x in k.split(":")) if ":" in k else (int(k), int(k) + 1)
        if 0 <= a < b <= len(words):
            return words[b - 1].strip("?,.'\"")        # keep the question's casing (TV, SUV)
    p = parse_program(q)
    return (p["select"][0] if p["select"] else "object").split()[-1]


def _compact(qid, q, p):
    return {"qid": qid, "image": q["imageId"], "question": q["question"], "answer": q["answer"],
            "ops": "|".join(p["ops"]), "select_id": p["select"][1] if p["select"] else None,
            "select_name": p["select"][0] if p["select"] else None,
            "ref_word": referent_word(q, p["select"][1]) if p["select"] else None,
            "rel_cls": p["relate"][0] if p["relate"] else None, "rel": p["relate"][1] if p["relate"] else None,
            "rel_tid": p["relate"][2] if p["relate"] else None, "query": p["query"],
            "detailed": q["types"]["detailed"]}


def question_pool(root, split, cache_path):
    """All query-type, filter-free questions of `<split>_all_questions.json` as compact
    dicts grouped by image (the natural-counterfactual pool). Cached (the raw file is
    1.7 GB and takes ~3 min to parse)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            rows = json.load(f)
    else:
        with open(Path(root) / "questions" / f"{split}_all_questions.json") as f:
            qa = json.load(f)
        rows = []
        for qid, q in qa.items():
            if q["types"]["structural"] != "query":
                continue
            p = parse_program(q)
            if p["filters"] or p["query"] is None or p["select"] is None:
                continue
            rows.append(_compact(qid, q, p))
        del qa
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(rows, f)
    pool = collections.defaultdict(list)
    for r in rows:
        pool[r["image"]].append(r)
    return pool


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def object_color(obj):
    attrs = [a.lower() for a in obj.get("attributes", [])]
    for c in COLOR_WORDS:
        if c in attrs:
            return c
    return None


def _value(obj, query):
    """The queried value of an object as the model must answer it."""
    if query == "name":
        return obj["name"]
    if query == "color":
        return object_color(obj)
    attrs = [a.lower() for a in obj.get("attributes", [])]
    if query == "material":
        return next((a for a in attrs if a in MATERIAL_WORDS), None)
    if query == "size":
        return next((a for a in attrs if a in SIZE_WORDS), None)
    return None


def patch_cover(box, W, H, grid):
    """(grid*grid,) fraction of each patch cell covered by the box (original pixels;
    the model resizes to a square, so cell = W/grid x H/grid)."""
    x, y, w, h = box
    px, py = W / grid, H / grid
    cols = np.arange(grid)
    ix = np.clip(np.minimum(x + w, (cols + 1) * px) - np.maximum(x, cols * px), 0, None) / px
    rows = np.arange(grid)
    iy = np.clip(np.minimum(y + h, (rows + 1) * py) - np.maximum(y, rows * py), 0, None) / py
    return (iy[:, None] * ix[None, :]).reshape(-1)


def box_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def centre_patch_units(box, W, H, grid):
    """Box centre in patch units (col, row)."""
    x, y, w, h = box
    return (x + w / 2) / (W / grid), (y + h / 2) / (H / grid)


def lr_holds(cand_box, anchor_box, relation, W, H, grid, margin):
    """`cand` is to the left/right of `anchor` by at least `margin` patches (2-D
    image-plane criterion on box centres)."""
    cx, _ = centre_patch_units(cand_box, W, H, grid)
    ax, _ = centre_patch_units(anchor_box, W, H, grid)
    d = ax - cx if LR_RELATIONS[relation] == "left" else cx - ax
    return d >= margin


def _box(obj):
    return (float(obj["x"]), float(obj["y"]), float(obj["w"]), float(obj["h"]))


def _name_unique(sg, name):
    return sum(1 for o in sg["objects"].values() if o["name"] == name) == 1


def _object_record(oid, obj, W, H, grid, cover):
    cov = patch_cover(_box(obj), W, H, grid)
    cx, cy = centre_patch_units(_box(obj), W, H, grid)
    return {"id": oid, "name": obj["name"], "color": object_color(obj),
            "attributes": list(obj.get("attributes", [])), "box_xywh": list(_box(obj)),
            "n_patches_cov": int((cov >= cover).sum()), "centre_col_row": [cx, cy]}


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

class Funnel:
    def __init__(self, n0):
        self.steps = collections.OrderedDict([("0 program shape", n0)])
        self.reason = {}

    def drop(self, qid, rule):
        self.steps[rule] = self.steps.get(rule, 0) + 1
        self.reason[qid] = rule


class Geometry:
    def __init__(self, W, H, grid, cover, min_patches, max_frac, iou_tol):
        self.W, self.H, self.grid = W, H, grid
        self.cover, self.min_patches, self.max_frac, self.iou_tol = cover, min_patches, max_frac, iou_tol

    def role_ok(self, box):
        cov = patch_cover(box, self.W, self.H, self.grid)
        return int((cov >= self.cover).sum()) >= self.min_patches and (box[2] * box[3]) / (self.W * self.H) <= self.max_frac

    def separate(self, a, b):
        return box_iou(a, b) <= self.iou_tol

    def lr(self, cand, anchor, rel, margin):
        return lr_holds(cand, anchor, rel, self.W, self.H, self.grid, margin)


def _candidates(mode, questions):
    """Balanced-val questions whose program has the right shape for the mode (rule 0)."""
    out = {}
    for qid, q in questions.items():
        p = parse_program(q)
        if q["types"]["structural"] != "query" or p["query"] is None or p["filters"] or p["select"] is None:
            continue
        if mode == "direct":
            if p["ops"] == ["select", "query"] and p["query"] in ATTRIBUTE_QUERIES:
                out[qid] = (q, p)
        elif mode == "spatial":
            if (p["ops"] == ["select", "relate", "query"] and p["relate"] is not None
                    and p["relate"][1] in LR_RELATIONS and p["query"] in ({"name"} | ATTRIBUTE_QUERIES)):
                out[qid] = (q, p)
        elif mode == "same":
            if (p["ops"] == ["select", "relate", "query"] and p["relate"] is not None
                    and p["relate"][1] == "same color" and p["query"] == "name"):
                out[qid] = (q, p)
    return out


def _in_vocab(vals, vocab):
    return all(v is not None and v in vocab for v in vals)


def _rename(question, old, new):
    """Replace 'the <old>' by 'the <new>'; None unless exactly one occurrence."""
    pat = re.compile(r"\bthe " + re.escape(old) + r"\b")
    if len(pat.findall(question)) != 1:
        return None
    return pat.sub("the " + new, question, count=1)


def build_records(mode, questions, scene_graphs, pool, vocab, *, grid=16, margin=2.0,
                  geometry="strict", bg_per_image=64, seed=42, n2c=None):
    """S_eligible records for one mode. Returns (records, owners, funnel)."""
    gp = GEOMETRY_PRESETS[geometry]
    cands = _candidates(mode, questions)
    fun = Funnel(len(cands))
    records, owners = [], []
    for qid in sorted(cands):
        q, p = cands[qid]
        sg = scene_graphs.get(q["imageId"])
        if sg is None:
            fun.drop(qid, "1 no scene graph"); continue
        W, H = sg["width"], sg["height"]
        objs = sg["objects"]
        geo = Geometry(W, H, grid, **gp)
        anchor = objs.get(p["select"][1])
        if anchor is None:
            fun.drop(qid, "1 anchor id missing"); continue
        if not _name_unique(sg, anchor["name"]):
            fun.drop(qid, "1 anchor name not unique"); continue
        others = [r for r in pool.get(q["imageId"], []) if r["qid"] != qid]
        if mode == "direct":
            rec = _direct_record(qid, q, p, sg, objs, anchor, others, vocab, fun, geo)
        elif mode == "spatial":
            rec = _spatial_record(qid, q, p, sg, objs, anchor, others, vocab, fun, geo, margin)
        else:
            rec = _same_record(qid, q, p, sg, objs, anchor, others, vocab, fun, geo, n2c)
        if rec is None:
            continue
        role_ids = [o["id"] for o in rec["objects"]]
        role_boxes = [tuple(o["box_xywh"]) for o in rec["objects"]]
        covs = np.stack([patch_cover(b, W, H, grid) for b in role_boxes])
        owner = np.zeros(grid * grid, np.int8)
        best, top = covs.argmax(0), covs.max(0)
        second = np.sort(covs, 0)[-2] if len(role_boxes) > 1 else np.zeros_like(top)
        sel = (top >= gp["cover"]) & (top > second)
        owner[sel] = best[sel] + 1
        if any(int((owner == k + 1).sum()) < gp["min_patches"] for k in range(len(role_boxes))):
            fun.drop(qid, "2 role < min patches after owner assignment"); continue
        any_cov = np.zeros(grid * grid, np.float32)
        other = np.zeros(grid * grid, bool)
        for oid, o in objs.items():
            cov = patch_cover(_box(o), W, H, grid)
            any_cov = np.maximum(any_cov, cov)
            if oid not in role_ids:
                other |= cov >= gp["cover"]
        # background (b): patches that no annotated box covers by at least `cover`
        # (the same threshold that makes a patch an object's); (a) = the non-role
        # object patches; the two partition the non-role patches.
        bg_b = np.nonzero((any_cov < gp["cover"]) & (owner == 0))[0]
        if len(bg_b) < 8:
            fun.drop(qid, "2 fewer than 8 background (b) patches"); continue
        rng = np.random.RandomState(seed + int(qid) % 100000)
        bg_sample = sorted(int(v) for v in rng.choice(bg_b, min(bg_per_image, len(bg_b)), replace=False))
        cent = [[float((np.nonzero(owner == k + 1)[0] // grid).mean()), float((np.nonzero(owner == k + 1)[0] % grid).mean())]
                for k in range(len(role_boxes))]
        rec.update(question_id=qid, image_id=q["imageId"], filename=f"{q['imageId']}.jpg",
                   image_wh=[W, H], mode=mode, grid=grid, margin_patches=margin, geometry=geometry,
                   centroids_row_col=cent, n_patches=[int((owner == k + 1).sum()) for k in range(len(role_boxes))],
                   bg_sample=bg_sample, other_obj_patches=[int(v) for v in np.nonzero(other)[0]],
                   n_background_b=int(len(bg_b)), gqa_types=q["types"],
                   clevr_answerable=all(v in CLEVR_COLORS for v in rec["answers"].values()))
        records.append(rec)
        owners.append(owner)
    fun.steps["final S_eligible"] = len(records)
    return records, owners, fun


def _direct_record(qid, q, p, sg, objs, anchor, others, vocab, fun, geo):
    query = p["query"]
    a_box = _box(anchor)
    if not geo.role_ok(a_box):
        fun.drop(qid, "2 referent box geometry"); return None
    if _value(anchor, query) != q["answer"]:
        fun.drop(qid, "3 scene-graph value != GQA answer"); return None
    partner = None
    for r in sorted(others, key=lambda r: r["qid"]):
        if r["ops"] != "select|query" or r["query"] != query or r["select_id"] == p["select"][1] or r["answer"] == q["answer"]:
            continue
        o2 = objs.get(r["select_id"])
        if o2 is None or not _name_unique(sg, o2["name"]) or o2["name"] == anchor["name"]:
            continue
        b2 = _box(o2)
        if not geo.separate(a_box, b2) or not geo.role_ok(b2) or _value(o2, query) != r["answer"]:
            continue
        partner = (r, o2)
        break
    if partner is None:
        fun.drop(qid, "4 no natural pair"); return None
    r2, o2 = partner
    if not _in_vocab([q["answer"], r2["answer"]], vocab):
        fun.drop(qid, "3 answer not in vocab"); return None
    return {"objects": [_object_record(p["select"][1], anchor, geo.W, geo.H, geo.grid, geo.cover),
                        _object_record(r2["select_id"], o2, geo.W, geo.H, geo.grid, geo.cover)],
            "A": 0, "T": 0, "D": 1, "queried": query, "pair_question_id": r2["qid"], "c2_source": "natural",
            "questions": {"c1": q["question"], "c2": r2["question"]},
            "referent_words": {"c1": referent_word(q, p["select"][1]), "c2": r2["ref_word"]},
            "answers": {"c1": q["answer"], "c2": r2["answer"]}}


def _spatial_record(qid, q, p, sg, objs, anchor, others, vocab, fun, geo, margin):
    cls, rel, t_id = p["relate"]
    query = p["query"]
    target = objs.get(t_id)
    if target is None:
        fun.drop(qid, "1 target id missing"); return None
    if target["name"] == anchor["name"]:
        fun.drop(qid, "1 anchor and target share a name"); return None
    a_box, t_box = _box(anchor), _box(target)
    if not geo.separate(a_box, t_box):
        fun.drop(qid, "2 anchor/target boxes overlap"); return None
    if not (geo.role_ok(a_box) and geo.role_ok(t_box)):
        fun.drop(qid, "2 role box geometry"); return None
    if not geo.lr(t_box, a_box, rel, 0.0):
        fun.drop(qid, "7 scene-graph relation disagrees with 2-D box centres"); return None
    if not geo.lr(t_box, a_box, rel, margin):
        fun.drop(qid, f"8 centre distance < {margin:g} patches"); return None
    if _value(target, query) != q["answer"]:
        fun.drop(qid, "3 scene-graph value != GQA answer"); return None
    opp = LR_OPPOSITE[rel]
    # natural c2: same anchor, opposite relation, same class word and query, other answer
    c2 = None
    for r in sorted(others, key=lambda r: r["qid"]):
        if (r["ops"] != "select|relate|query" or r["query"] != query or r["select_id"] != p["select"][1]
                or r["rel_cls"] != cls or r["rel"] != opp or r["answer"] == q["answer"]):
            continue
        o3 = objs.get(r["rel_tid"])
        if o3 is None or o3["name"] == anchor["name"]:
            continue
        b3 = _box(o3)
        if not (geo.separate(b3, a_box) and geo.separate(b3, t_box) and geo.role_ok(b3)):
            continue
        if not geo.lr(b3, a_box, opp, margin) or _value(o3, query) != r["answer"]:
            continue
        c2 = (r, o3)
        break
    # natural c3: same relation / class / query, same answer object, other anchor
    c3 = None
    for r in sorted(others, key=lambda r: r["qid"]):
        if (r["ops"] != "select|relate|query" or r["query"] != query or r["rel_tid"] != t_id
                or r["rel_cls"] != cls or r["rel"] != rel or r["select_id"] == p["select"][1] or r["answer"] != q["answer"]):
            continue
        o4 = objs.get(r["select_id"])
        if o4 is None or o4["name"] in (anchor["name"], target["name"]) or not _name_unique(sg, o4["name"]):
            continue
        b4 = _box(o4)
        if not (geo.separate(b4, a_box) and geo.separate(b4, t_box) and geo.role_ok(b4)):
            continue
        if not geo.lr(t_box, b4, rel, margin):
            continue
        if c2 is not None and r["select_id"] != c2[0]["rel_tid"] and c3 is not None:
            continue
        c3 = (r, o4)
        if c2 is not None and r["select_id"] == c2[0]["rel_tid"]:
            break
    if c2 is None and c3 is None:
        fun.drop(qid, "9 no natural c2 or c3 question"); return None
    # the third object: c2's answer object when c2 exists, else c3's anchor. c3 is kept
    # only when its anchor is that same object (three roles, as in CLEVR).
    if c2 is not None:
        oid3, o3 = c2[0]["rel_tid"], c2[1]
        if c3 is not None and c3[0]["select_id"] != oid3:
            c3 = None
    else:
        oid3, o3 = c3[0]["select_id"], c3[1]
    # referent words as they appear in the question text (GQA's select argument), not
    # the scene-graph name ("freezer" in the question, "refrigerator" in the graph)
    questions, answers, words = {"c1": q["question"]}, {"c1": q["answer"]}, {"c1": referent_word(q, p["select"][1])}
    if c2 is not None:
        questions["c2"], answers["c2"], words["c2"] = c2[0]["question"], c2[0]["answer"], c2[0]["ref_word"]
    if c3 is not None:
        questions["c3"], answers["c3"], words["c3"] = c3[0]["question"], c3[0]["answer"], c3[0]["ref_word"]
    if not _in_vocab(list(answers.values()), vocab):
        fun.drop(qid, "3 answer not in vocab"); return None
    tx = centre_patch_units(t_box, geo.W, geo.H, geo.grid)[0]
    return {"objects": [_object_record(p["select"][1], anchor, geo.W, geo.H, geo.grid, geo.cover),
                        _object_record(t_id, target, geo.W, geo.H, geo.grid, geo.cover),
                        _object_record(oid3, o3, geo.W, geo.H, geo.grid, geo.cover)],
            "A": 0, "T": 1, "D": 2, "queried": query, "relation": rel, "opposite": opp, "class_word": cls,
            "axis": "column", "has_c2": c2 is not None, "has_c3": c3 is not None,
            "c2_question_id": c2[0]["qid"] if c2 else None, "c3_question_id": c3[0]["qid"] if c3 else None,
            "dissociated": bool((tx - geo.grid / 2) * (1 if LR_RELATIONS[rel] == "right" else -1) < 0),
            "questions": questions, "referent_words": words, "answers": answers}


def _member(name, cls, n2c):
    """True / False / None (unknown) membership of a scene-graph name in a class word."""
    if name == cls:
        return True
    classes = n2c.get(name)
    if not classes:
        return None
    return cls in classes


def class_members(questions, scene_graphs, cache_path=None):
    """name -> set(classes), read from relate arguments of the GQA questions
    ("vegetable,to the right of,s (id)": the object id's scene-graph name is a
    vegetable). Cached as json."""
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            return {k: set(v) for k, v in json.load(f).items()}
    n2c = collections.defaultdict(set)
    for q in questions.values():
        sg = scene_graphs.get(q["imageId"])
        if sg is None:
            continue
        for s in q["semantic"]:
            if s["operation"] != "relate":
                continue
            m = _ARG_ID.search(s["argument"])
            parts = s["argument"].split(" (")[0].split(",")
            if not m or len(parts) != 3 or parts[0].strip() in ("_", ""):
                continue
            obj = sg["objects"].get(m.group(1))
            if obj is not None:
                n2c[obj["name"]].add(parts[0].strip())
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({k: sorted(v) for k, v in n2c.items()}, f)
    return n2c


def _same_record(qid, q, p, sg, objs, anchor, others, vocab, fun, geo, n2c):
    cls, _, t_id = p["relate"]
    target = objs.get(t_id)
    if target is None:
        fun.drop(qid, "1 target id missing"); return None
    if target["name"] == anchor["name"]:
        fun.drop(qid, "1 anchor and target share a name"); return None
    a_col = object_color(anchor)
    if a_col is None or object_color(target) != a_col:
        fun.drop(qid, "11 anchor colour missing or target colour differs"); return None
    if re.search(r"\b" + re.escape(a_col) + r"\b", q["question"]):
        fun.drop(qid, "11 anchor colour stated in the question"); return None
    a_box, t_box = _box(anchor), _box(target)
    if not geo.separate(a_box, t_box):
        fun.drop(qid, "2 anchor/target boxes overlap"); return None
    if not (geo.role_ok(a_box) and geo.role_ok(t_box)):
        fun.drop(qid, "2 role box geometry"); return None
    if target["name"] != q["answer"]:
        fun.drop(qid, "3 scene-graph value != GQA answer"); return None
    same_col = [oid for oid, o in objs.items() if oid != p["select"][1] and object_color(o) == a_col]
    if same_col != [t_id]:
        fun.drop(qid, "11 anchor colour shared by other objects"); return None
    third = None
    for oid, o in objs.items():
        if oid in (p["select"][1], t_id) or object_color(o) in (None, a_col):
            continue
        ob = _box(o)
        if not (geo.separate(ob, a_box) and geo.separate(ob, t_box) and geo.role_ok(ob)):
            continue
        if o["name"] not in vocab or o["name"] in (anchor["name"], target["name"]) or not _name_unique(sg, o["name"]):
            continue
        third = (oid, o)
        break
    if third is None:
        fun.drop(qid, "11 no third object"); return None
    c2, source, w2 = None, None, None
    for r in sorted(others, key=lambda r: r["qid"]):
        if (r["ops"] == "select|relate|query" and r["query"] == "name" and r["rel"] == "same color"
                and r["select_id"] == t_id and r["rel_tid"] == p["select"][1] and r["answer"] == anchor["name"]):
            c2, source, w2 = r["question"], "natural", r["ref_word"]
            break
    if c2 is None and _member(anchor["name"], cls, n2c) is True:
        # constructed c2 ("... same color as the <target>?" -> anchor) is well-formed only
        # when the anchor belongs to the class word of the question (known from the GQA
        # question corpus)
        old = referent_word(q, p["select"][1])
        c2, source, w2 = _rename(q["question"], old, target["name"]), "constructed", target["name"].split()[-1]
    if c2 is None:
        # no counterfactual: the item stays in S_eligible for the observational analyses
        # (c0 / c1 only) and is excluded from the interventions (has_c2 = False)
        source = "none"
    if not _in_vocab([q["answer"], anchor["name"]], vocab):
        fun.drop(qid, "3 answer not in vocab"); return None
    questions = {"c1": q["question"]}
    words, answers = {"c1": referent_word(q, p["select"][1])}, {"c1": q["answer"]}
    if c2 is not None:
        questions["c2"], words["c2"], answers["c2"] = c2, w2, anchor["name"]
    return {"objects": [_object_record(p["select"][1], anchor, geo.W, geo.H, geo.grid, geo.cover),
                        _object_record(t_id, target, geo.W, geo.H, geo.grid, geo.cover),
                        _object_record(third[0], third[1], geo.W, geo.H, geo.grid, geo.cover)],
            "A": 0, "T": 1, "D": 2, "queried": "name", "attribute": "color", "shared_color": a_col,
            "class_word": cls, "c2_source": source, "has_c2": c2 is not None,
            "questions": questions, "referent_words": words, "answers": answers}


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def write_outputs(out_dir, mode, records, owners, fun, tag=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"relational_records{tag}.json", "w") as f:
        json.dump(records, f, indent=1)
    if owners:
        np.save(out_dir / f"owner{tag}.npy", np.stack(owners))
    with open(out_dir / f"funnel{tag}.json", "w") as f:
        json.dump({"mode": mode, "steps": fun.steps, "exclusion_reason": fun.reason}, f, indent=1)
    lines = ["| rule | removed | remaining |", "|---|---|---|"]
    remaining = fun.steps["0 program shape"]
    order = sorted(fun.steps, key=lambda k: (k == "final S_eligible", k))     # by rule number, final last
    for k in order:
        v = fun.steps[k]
        if k in ("0 program shape", "final S_eligible"):
            lines.append(f"| {k} | — | {v} |")
            continue
        remaining -= v
        lines.append(f"| {k} | {v} | {remaining} |")
    with open(out_dir / f"funnel{tag}.md", "w") as f:
        f.write(f"# funnel — {mode}{tag}\n\n" + "\n".join(lines) + "\n")
    print(f"[{mode}{tag}] " + "; ".join(f"{k}: {v}" for k, v in fun.steps.items()))


# ---------------------------------------------------------------------------
# audit page (blind: no model output anywhere)
# ---------------------------------------------------------------------------

ROLE_DRAW = {"A": ((148, 102, 189), "anchor"), "T": ((44, 160, 44), "target"), "D": ((31, 119, 180), "third object")}


def audit_page(out_dir, records, images_root, n_items, seed=42, max_width=560):
    """Write audit/index.html with box-overlaid images and a checklist per item;
    the page saves answers as a JSON download. Shows GQA ground truth only."""
    from PIL import Image, ImageDraw
    out_dir = Path(out_dir)
    img_dir = out_dir / "audit" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    idx = sorted(rng.choice(len(records), min(n_items, len(records)), replace=False).tolist())
    items = []
    for i in idx:
        r = records[i]
        im = Image.open(Path(images_root) / r["filename"]).convert("RGB")
        scale = min(1.0, max_width / im.width)
        im = im.resize((int(im.width * scale), int(im.height * scale)))
        dr = ImageDraw.Draw(im)
        roles = [("A", r["A"]), ("T", r["T"]), ("D", r["D"])] if r["mode"] != "direct" else [("T", r["T"]), ("D", r["D"])]
        for role, j in roles:
            x, y, w, h = r["objects"][j]["box_xywh"]
            col, lab = ROLE_DRAW[role]
            dr.rectangle([x * scale, y * scale, (x + w) * scale, (y + h) * scale], outline=col, width=3)
            dr.text((x * scale + 3, y * scale + 3), f"{lab}: {r['objects'][j]['name']}", fill=col)
        fn = f"{r['question_id']}.jpg"
        im.save(img_dir / fn, quality=85)
        items.append((r, fn))
    checks = [("roles", "role boxes point at the right objects (anchor / target / third object)"),
              ("boxes", "boxes are tight enough (no other object dominates a box)"),
              ("relation", "the stated relation / shared colour is true in the image"),
              ("unique", "the referent is unique for its name; the answer object is unique for the question"),
              ("c2", "the counterfactual question (c2 / c3) is grammatical and has the listed answer")]
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>X23 audit</title>",
             "<style>body{font-family:sans-serif;max-width:1100px;margin:auto}"
             ".item{border-top:1px solid #ccc;padding:12px 0}.q{margin:2px 0}"
             "label{display:block}textarea{width:100%}</style></head><body>",
             f"<h2>X23 audit — {records[0]['mode']} ({len(items)} items)</h2>",
             "<p>For each item tick what is TRUE. Leave unticked = false. Add a note if needed. "
             "Press 'Download answers' at the end (a JSON file).</p>"]
    for k, (r, fn) in enumerate(items):
        parts.append(f"<div class='item' id='it{k}'><b>{k + 1}. question {r['question_id']} / image {r['image_id']}</b>")
        parts.append(f"<div><img src='img/{fn}'></div>")
        for c in ("c1", "c2", "c3"):
            if c in r["questions"]:
                parts.append(f"<div class='q'><b>{c}</b>: {html.escape(r['questions'][c])} &rarr; "
                             f"<i>{html.escape(str(r['answers'].get(c)))}</i></div>")
        for key, text in checks:
            parts.append(f"<label><input type='checkbox' data-q='{r['question_id']}' data-k='{key}'> {text}</label>")
        parts.append(f"<textarea rows='1' data-q='{r['question_id']}' data-k='note' placeholder='note'></textarea></div>")
    parts.append("<p><button onclick='dl()'>Download answers</button></p>"
                 "<script>function dl(){const o={};document.querySelectorAll('[data-q]').forEach(e=>{"
                 "const q=e.dataset.q;o[q]=o[q]||{};o[q][e.dataset.k]=e.type==='checkbox'?e.checked:e.value;});"
                 "const b=new Blob([JSON.stringify(o,null,1)],{type:'application/json'});"
                 "const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='audit_answers.json';a.click();}"
                 "</script></body></html>")
    with open(out_dir / "audit" / "index.html", "w") as f:
        f.write("\n".join(parts))
    with open(out_dir / "audit" / "sample.json", "w") as f:
        json.dump([r["question_id"] for r, _ in items], f)
    print(f"Saved: {out_dir / 'audit' / 'index.html'} ({len(items)} items)")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_filter(args, out_dir):
    """--gqa-filter MODE: S_eligible for MODE under both geometry presets and (spatial)
    margins 1 / 2 / 3; the untagged files use --gqa-geometry / --gqa-margin; audit page
    on the untagged set."""
    out_dir = Path(out_dir)
    meta_dir = Path(args.gqa_meta_dir)
    mode = args.gqa_filter
    questions = load_questions(args.gqa_root, args.gqa_split)
    sgs = load_scene_graphs(args.gqa_root, args.gqa_split)
    vocab = answer_vocab(args.gqa_root, meta_dir)
    n2c = class_members(questions, sgs, meta_dir / f"class_members_{args.gqa_split}.json") if mode == "same" else None
    pool = question_pool(args.gqa_root, args.gqa_split, meta_dir / f"question_pool_{args.gqa_split}_all_refword2.json")
    print(f"GQA {args.gqa_split}: {len(questions)} balanced questions, {len(sgs)} scene graphs, "
          f"natural-counterfactual pool {sum(len(v) for v in pool.values())} questions on {len(pool)} images; "
          f"answer vocab {len(vocab)}")
    variants = [(g, m) for g in GEOMETRY_PRESETS for m in ([1.0, 2.0, 3.0] if mode == "spatial" else [float(args.gqa_margin)])]
    main = None
    for g, m in variants:
        records, owners, fun = build_records(mode, questions, sgs, pool, vocab, grid=args.grid, margin=m,
                                            geometry=g, bg_per_image=args.bg_per_image, seed=args.seed, n2c=n2c)
        is_main = g == args.gqa_geometry and m == float(args.gqa_margin)
        tag = "" if is_main else f"_{g}" + (f"_margin{int(m)}" if mode == "spatial" else "")
        write_outputs(out_dir, mode, records, owners, fun, tag=tag)
        if is_main:
            main = records
    r = main or []
    if r:
        print(f"S_eligible ({mode}, geometry {args.gqa_geometry}, margin {args.gqa_margin:g}): {len(r)} questions on "
              f"{len({x['image_id'] for x in r})} images; CLEVR-answerable {sum(x['clevr_answerable'] for x in r)}")
        if mode == "spatial":
            print(f"  with c2 {sum(x['has_c2'] for x in r)}, with c3 {sum(x['has_c3'] for x in r)}, "
                  f"both {sum(x['has_c2'] and x['has_c3'] for x in r)}; dissociated {sum(x['dissociated'] for x in r)}; "
                  f"queried {dict(collections.Counter(x['queried'] for x in r))}")
        if mode == "same":
            print(f"  c2 source {dict(collections.Counter(x['c2_source'] for x in r))}")
        if args.gqa_audit_n:
            audit_page(out_dir, r, Path(args.gqa_root) / "images", args.gqa_audit_n, seed=args.seed)
    return main

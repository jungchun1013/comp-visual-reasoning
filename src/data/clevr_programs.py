"""Utilities for parsing CLEVR functional programs."""

from __future__ import annotations


def compute_program_depth(program: list[dict]) -> int:
    """Compute the reasoning depth (longest path) in the program DAG.

    Each program step has an 'inputs' list of indices pointing to earlier steps.
    Depth is the longest chain from any root (no inputs / 'scene') to the final step.
    """
    if not program:
        return 0

    n = len(program)
    depths = [0] * n

    for i, step in enumerate(program):
        inputs = step.get("inputs", [])
        if not inputs:
            depths[i] = 1
        else:
            depths[i] = max(depths[j] for j in inputs) + 1

    return depths[-1]


def classify_question_type(program: list[dict]) -> str:
    """Classify question by the final program step's function type.

    Returns one of: exist, count, equal_integer, less_than, greater_than,
    equal_material, equal_color, equal_shape, equal_size,
    query_color, query_shape, query_material, query_size.

    For simpler analysis, use `coarse_question_type()` to map to 6 categories.
    """
    if not program:
        return "unknown"
    return program[-1].get("function", program[-1].get("type", "unknown"))


# Mapping from fine-grained CLEVR program types to coarse categories
_COARSE_TYPE_MAP = {
    "exist": "exist",
    "count": "count",
    "equal_integer": "compare_integer",
    "less_than": "compare_integer",
    "greater_than": "compare_integer",
    "equal_material": "equal_attribute",
    "equal_color": "equal_attribute",
    "equal_shape": "equal_attribute",
    "equal_size": "equal_attribute",
    "query_color": "query_attribute",
    "query_shape": "query_attribute",
    "query_material": "query_attribute",
    "query_size": "query_attribute",
}


def coarse_question_type(program: list[dict]) -> str:
    """Map to 6 coarse categories: exist, count, compare_integer,
    query_attribute, equal_attribute, unknown."""
    fine = classify_question_type(program)
    return _COARSE_TYPE_MAP.get(fine, "unknown")


def extract_oracle_prompt(scene: dict, program: list[dict]) -> str:
    """Execute the CLEVR program on the scene to find the target object(s),
    then return a text description for oracle steering.

    For query/exist questions, returns the description of the primary referred object.
    For counting questions, returns the filter description (e.g., "red metal cubes").
    For comparison questions, returns descriptions of both compared objects.
    """
    objects = scene.get("objects", [])
    if not objects or not program:
        return ""

    # Execute program to get intermediate object sets
    results = _execute_program(objects, program)

    # Build oracle prompt based on question type
    final_type = program[-1].get("function", program[-1].get("type", ""))

    if final_type.startswith("query_"):
        # The input to query is a single object — describe it
        input_idx = program[-1]["inputs"][0]
        obj_set = results[input_idx]
        if obj_set:
            return _describe_object(obj_set[0])
        return ""

    elif final_type == "exist" or final_type == "count":
        # Describe the filtered set
        input_idx = program[-1]["inputs"][0]
        obj_set = results[input_idx]
        if obj_set:
            return _describe_object(obj_set[0])
        # If empty set, describe the filter chain
        return _describe_filter_chain(program, program[-1]["inputs"][0])

    elif final_type in ("equal_integer", "less_than", "greater_than"):
        # Two sets being compared — describe first object of each
        parts = []
        for inp_idx in program[-1]["inputs"]:
            obj_set = results[inp_idx]
            if obj_set:
                parts.append(_describe_object(obj_set[0]))
        return " and ".join(parts) if parts else ""

    elif final_type.startswith("equal_"):
        # Two objects being compared
        parts = []
        for inp_idx in program[-1]["inputs"]:
            obj_set = results[inp_idx]
            if obj_set:
                parts.append(_describe_object(obj_set[0]))
        return " and ".join(parts) if parts else ""

    return ""


def _describe_object(obj: dict) -> str:
    """Create a text description like 'large red metal cube'."""
    parts = []
    for attr in ("size", "color", "material", "shape"):
        val = obj.get(attr, "")
        if val:
            parts.append(val)
    return " ".join(parts)


def _describe_filter_chain(program: list[dict], step_idx: int) -> str:
    """Walk backward through filter steps to build a text description."""
    parts = []
    idx = step_idx
    while idx >= 0 and idx < len(program):
        step = program[idx]
        stype = step.get("function", step.get("type", ""))
        if stype.startswith("filter_"):
            value_inputs = step.get("value_inputs", [])
            if value_inputs:
                parts.append(value_inputs[0])
            idx = step["inputs"][0] if step.get("inputs") else -1
        else:
            break
    parts.reverse()
    return " ".join(parts) if parts else ""


def _execute_program(objects: list[dict], program: list[dict]) -> list[list[dict]]:
    """Execute CLEVR functional program on scene objects.

    Returns a list of results (one per program step). Each result is a list of
    object dicts (the working set at that step).
    """
    results: list[list[dict]] = []

    for step in program:
        stype = step.get("function", step.get("type", ""))
        inputs = step.get("inputs", [])
        value_inputs = step.get("value_inputs", [])

        if stype == "scene":
            results.append(list(objects))

        elif stype.startswith("filter_"):
            attr = stype[len("filter_"):]  # e.g., "color", "shape", "size", "material"
            val = value_inputs[0] if value_inputs else ""
            parent_set = results[inputs[0]] if inputs else objects
            filtered = [o for o in parent_set if o.get(attr, "") == val]
            results.append(filtered)

        elif stype == "unique":
            parent_set = results[inputs[0]] if inputs else []
            results.append(parent_set[:1])

        elif stype == "relate":
            # Spatial relation: find objects related to the input object
            direction = value_inputs[0] if value_inputs else ""
            parent_set = results[inputs[0]] if inputs else []
            if parent_set:
                anchor = parent_set[0]
                related = _get_related_objects(anchor, direction, objects)
                results.append(related)
            else:
                results.append([])

        elif stype == "intersect":
            set_a = set(id(o) for o in results[inputs[0]]) if inputs else set()
            set_b = set(id(o) for o in results[inputs[1]]) if len(inputs) > 1 else set()
            intersection_ids = set_a & set_b
            all_objs = results[inputs[0]] + results[inputs[1]] if len(inputs) > 1 else []
            seen = set()
            intersected = []
            for o in all_objs:
                if id(o) in intersection_ids and id(o) not in seen:
                    intersected.append(o)
                    seen.add(id(o))
            results.append(intersected)

        elif stype == "union":
            set_a = results[inputs[0]] if inputs else []
            set_b = results[inputs[1]] if len(inputs) > 1 else []
            seen = set()
            union = []
            for o in set_a + set_b:
                if id(o) not in seen:
                    union.append(o)
                    seen.add(id(o))
            results.append(union)

        elif stype in ("same_color", "same_shape", "same_size", "same_material"):
            attr = stype[len("same_"):]
            parent_set = results[inputs[0]] if inputs else []
            if parent_set:
                val = parent_set[0].get(attr, "")
                same = [o for o in objects if o.get(attr) == val and o is not parent_set[0]]
                results.append(same)
            else:
                results.append([])

        else:
            # Terminal operations (query_*, exist, count, equal_*, less_than, greater_than)
            # Just pass through the input set for oracle prompt extraction
            parent_set = results[inputs[0]] if inputs else []
            results.append(parent_set)

    return results


def _get_related_objects(
    anchor: dict, direction: str, all_objects: list[dict]
) -> list[dict]:
    """Get objects spatially related to anchor in the given direction.

    CLEVR uses 3D coordinates. Relations are based on 2D projected positions:
    - left/right: x-axis
    - front/behind: y-axis (lower y = front in CLEVR's coordinate system)
    """
    ax, ay = anchor.get("3d_coords", [0, 0, 0])[:2]
    related = []
    for obj in all_objects:
        if obj is anchor:
            continue
        ox, oy = obj.get("3d_coords", [0, 0, 0])[:2]
        if direction == "left" and ox < ax:
            related.append(obj)
        elif direction == "right" and ox > ax:
            related.append(obj)
        elif direction == "front" and oy < ay:
            related.append(obj)
        elif direction == "behind" and oy > ay:
            related.append(obj)
    return related


# ── Public API for retrieval condition checking ──────────────────────


def evaluate_answer(objects: list[dict], program: list[dict]) -> str | None:
    """Execute program on scene objects and return the answer string.

    Returns None if execution fails (e.g., empty set at a unique step).
    """
    if not program or not objects:
        return None

    results = _execute_program(objects, program)
    final = program[-1]
    fn = final.get("function", final.get("type", ""))
    inputs = final.get("inputs", [])

    if fn.startswith("query_"):
        attr = fn[len("query_"):]
        obj_set = results[inputs[0]] if inputs else []
        return obj_set[0].get(attr) if obj_set else None

    if fn == "count":
        obj_set = results[inputs[0]] if inputs else []
        return str(len(obj_set))

    if fn == "exist":
        obj_set = results[inputs[0]] if inputs else []
        return "yes" if obj_set else "no"

    if fn.startswith("equal_") and fn != "equal_integer":
        attr = fn[len("equal_"):]
        if len(inputs) >= 2:
            a, b = results[inputs[0]], results[inputs[1]]
            if a and b:
                return "yes" if a[0].get(attr) == b[0].get(attr) else "no"
        return None

    if fn == "equal_integer":
        if len(inputs) >= 2:
            return "yes" if len(results[inputs[0]]) == len(results[inputs[1]]) else "no"
        return None

    if fn == "less_than":
        if len(inputs) >= 2:
            return "yes" if len(results[inputs[0]]) < len(results[inputs[1]]) else "no"
        return None

    if fn == "greater_than":
        if len(inputs) >= 2:
            return "yes" if len(results[inputs[0]]) > len(results[inputs[1]]) else "no"
        return None

    return None


def find_anchor(objects: list[dict], program: list[dict]) -> tuple[dict | None, dict]:
    """Execute program on scene, return (anchor_object, described_attrs_dict).

    Anchor = result of the first ``unique`` step.
    Described attrs = {attr_type: value} from filter steps leading to that unique.

    Returns (None, {}) if no unique step or the unique step yields an empty set.
    """
    if not program or not objects:
        return None, {}

    results = _execute_program(objects, program)
    described = extract_described_attrs(program)

    for i, step in enumerate(program):
        if step.get("function", step.get("type", "")) == "unique":
            obj_set = results[i]
            if obj_set:
                return obj_set[0], described
            return None, described

    return None, described


def extract_described_attrs(program: list[dict]) -> dict:
    """Extract {attr_type: value} from filter steps before the first ``unique``.

    Example: ``filter_color(brown) → filter_size(large) → unique``
    returns ``{"color": "brown", "size": "large"}``.
    """
    attrs = {}
    for step in program:
        fn = step.get("function", step.get("type", ""))
        if fn == "unique":
            break
        if fn.startswith("filter_"):
            value_inputs = step.get("value_inputs", [])
            if value_inputs:
                attrs[fn[len("filter_"):]] = value_inputs[0]
    return attrs


def find_target(objects: list[dict], program: list[dict]) -> tuple[dict | None, dict]:
    """Find target object after ``relate`` or ``same_*`` step.

    Walks the program past the first ``relate`` or ``same_*`` step, collects
    subsequent ``filter_*`` steps as target described attrs, and returns the
    first matching target object from the executed results.

    Returns (target_obj, target_described_attrs) or (None, {}).
    """
    if not program or not objects:
        return None, {}

    results = _execute_program(objects, program)

    # Find first relate or same_* step
    pivot_idx = None
    for i, step in enumerate(program):
        fn = step.get("function", step.get("type", ""))
        if fn == "relate" or fn.startswith("same_"):
            pivot_idx = i
            break
    if pivot_idx is None:
        return None, {}

    # Collect filters after pivot, before terminal operation
    target_attrs: dict = {}
    last_result_idx = pivot_idx
    for i in range(pivot_idx + 1, len(program)):
        fn = program[i].get("function", program[i].get("type", ""))
        if fn.startswith("filter_"):
            attr = fn[len("filter_"):]
            value_inputs = program[i].get("value_inputs", [])
            if value_inputs:
                target_attrs[attr] = value_inputs[0]
            last_result_idx = i
        elif fn == "unique":
            last_result_idx = i
        else:
            break  # terminal operation (exist, count, query_X)

    target_objs = results[last_result_idx]
    if target_objs:
        return target_objs[0], target_attrs
    return None, target_attrs


def extract_all_filter_attrs(program: list[dict]) -> list[tuple[str, str]]:
    """Return all (attr_type, value) pairs from every filter step in the program.

    Unlike ``extract_described_attrs``, this covers the entire program — useful
    for feature-extraction conditions in comparison/counting programs that
    mention multiple objects.
    """
    attrs = []
    for step in program:
        fn = step.get("function", step.get("type", ""))
        if fn.startswith("filter_"):
            value_inputs = step.get("value_inputs", [])
            if value_inputs:
                attrs.append((fn[len("filter_"):], value_inputs[0]))
    return attrs

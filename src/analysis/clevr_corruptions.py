"""CLEVR question corruption generator for activation patching.

Two-level taxonomy:
  Coarse (type):  attribute, spatial, attribute_query, quantifier
  Fine (fine_type): color, material, size, shape,
                    what_color, what_material, what_size, what_shape,
                    spatial, quantifier
"""

from __future__ import annotations

import random
import re

# ── Attribute swaps (within-type) ─────────────────────────────────

_COLOR_SWAPS = {
    "red": ["green", "blue"], "green": ["blue", "red"], "blue": ["red", "green"],
    "gray": ["brown"], "brown": ["gray"],
    "purple": ["cyan", "yellow"], "cyan": ["yellow", "purple"], "yellow": ["purple", "cyan"],
}

_MATERIAL_SWAPS = {
    "rubber": ["metal", "shiny"], "metal": ["rubber", "matte"],
    "shiny": ["matte", "rubber"], "matte": ["shiny", "metal"],
}

_SIZE_SWAPS = {
    "large": ["small", "tiny"], "small": ["large", "big"],
    "big": ["tiny", "small"], "tiny": ["large", "big"],
}

_SHAPE_ALTERNATIVES = {
    "cube": ["sphere", "cylinder"],
    "sphere": ["cube", "cylinder"],
    "cylinder": ["cube", "sphere"],
    "block": ["ball"],
    "ball": ["block"],
}

# ── Spatial swaps ─────────────────────────────────────────────────

SPATIAL_SWAPS = {
    "left": ["right"], "right": ["left"],
    "behind": ["in front of"], "in front of": ["behind"],
}

# ── Quantifier swaps ─────────────────────────────────────────────

QUANTIFIER_SWAPS = {
    "how many": ["is there a", "are there any", "is there any", ],
    "what number of": ["is there a", "are there any", "is there any", ],
    "is there a": ["how many", "what number of"],
    "are there any": ["how many", "what number of"],
    "is there any": ["how many", "what number of"],
}

# ── Attribute query phrases ──────────────────────────────────────

_QUERY_PHRASES = ["what color", "what material", "what shape", "what size"]


# ── Helpers ───────────────────────────────────────────────────────

def _try_swap(question: str, swap_table: dict,
              coarse_type: str, fine_type: str) -> list[dict]:
    """Try swaps from a table. Values are lists of alternatives (randomly chosen)."""
    results = []
    for original, alts in sorted(swap_table.items(), key=lambda x: -len(x[0])):
        if isinstance(alts, str):
            alts = [alts]
        pattern = re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE)
        if pattern.search(question):
            replacement = random.choice(alts)
            corrupted = pattern.sub(replacement, question, count=1)
            if corrupted != question:
                results.append({
                    "type": coarse_type,
                    "fine_type": fine_type,
                    "original_word": original,
                    "corrupted_word": replacement,
                    "corrupted_question": corrupted,
                })
    return results


def _try_query_swap(question: str) -> list[dict]:
    """Swap attribute query type: 'What color' → 'What shape', etc.

    Handles both sentence-initial ("What color is...") and
    mid-sentence ("The X has what color?") patterns.
    """
    q_lower = question.lower()
    results = []
    for phrase in _QUERY_PHRASES:
        # Match anywhere in the question (not just startswith)
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        match = pattern.search(q_lower)
        if match:
            attr = phrase.split()[1]  # "color", "material", "shape", "size"
            for other_phrase in _QUERY_PHRASES:
                if other_phrase == phrase:
                    continue
                # Preserve original casing of "What"/"what"
                orig_text = question[match.start():match.end()]
                if orig_text[0].isupper():
                    replacement = other_phrase.capitalize()
                else:
                    replacement = other_phrase
                corrupted = question[:match.start()] + replacement + question[match.end():]
                results.append({
                    "type": "attribute_query",
                    "fine_type": f"what_{attr}",
                    "original_word": phrase,
                    "corrupted_word": other_phrase,
                    "corrupted_question": corrupted,
                })
            break
    return results


# ── Public API ────────────────────────────────────────────────────

def generate_corruptions(question: str) -> list[dict]:
    """Generate corrupted versions of a CLEVR question.

    Returns list of dicts, each with:
        type: "attribute" | "spatial" | "attribute_query" | "quantifier"
        fine_type: "color" | "material" | "size" | "shape" |
                   "what_color" | "what_material" | "what_size" | "what_shape" |
                   "spatial" | "quantifier"
        original_word: str
        corrupted_word: str
        corrupted_question: str
    """
    corruptions = []
    # Attribute (within-type swap)
    corruptions.extend(_try_swap(question, _COLOR_SWAPS, "attribute", "color"))
    corruptions.extend(_try_swap(question, _MATERIAL_SWAPS, "attribute", "material"))
    corruptions.extend(_try_swap(question, _SIZE_SWAPS, "attribute", "size"))
    corruptions.extend(_try_swap(question, _SHAPE_ALTERNATIVES, "attribute", "shape"))
    # Spatial
    corruptions.extend(_try_swap(question, SPATIAL_SWAPS, "spatial", "spatial"))
    # Attribute query (cross-type swap)
    corruptions.extend(_try_query_swap(question))
    # Quantifier
    corruptions.extend(_try_swap(question, QUANTIFIER_SWAPS, "quantifier", "quantifier"))
    return corruptions

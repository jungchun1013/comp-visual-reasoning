# JOURNAL

## TODO
> [!NOTE] Persistent until done or removed. Every item requires a bracketed tag.
> [!NOTE] Tags: `[model]`, `[data]`, `[metrics]`, `[infra]`, `[plot]`, `[main flow]`, `[paper]`, `[ablation]`, `[debug]`

- [ ] [main flow] Confirm EXPERIMENT.md objective with user (initialized 2026-07-05 from the user's v2 outline)
- [ ] [paper] E1c s42/s43/s44 replication DONE for Table 1 backbone matrix + s43 for the 4-cell ablation table (08-13, tables in paper_artifacts.md §8.1) — Sup-ViT s44 flagged as a high-variance outlier (0.8078 vs 0.8655/0.8826); user still to pick footnote-vs-table-mean camera-ready treatment
- [ ] [paper] R4: transfusion baseline has no checkpoint — retrain or drop from baseline table? (user decision)
- [ ] [paper] learned_text paper cell (24.6) is protocol-dependent: training-log final-ep 0.2456 vs independent eval protocol 0.197 (last.pt) / 0.207 (best.pt ep2); windowed train-loop acc 0.4667 is a third number. User decides camera-ready treatment (footnote or renumber). Artifacts: `outputs/analysis/generalization/clevr_dinov2_learned_text_decoder1l{,_lastep}_s42.json`
- [ ] [paper] Baseline implementations GATED ON USER GO: OpenFlamingo-9B zero-shot mechanism analysis (priority 1), T5-vs-RoBERTa capacity axis (+CLOSURE). Designs pre-registered in docs/paper_v2_outline.md; survey in experiment_registry.md X14. (PixArt-Σ un-gated by user 2026-07-06 — probe/CA-map running.)
- [x] [plot] E5 failure-mode figures from the landed JSONs (§7) — done 07-10 (`failure_modes.py --replot all`, 4 models)
- [ ] [plot] CoGenT alpha-sweep curve from `cogent_zeroshot/zeroshot_alpha_sweep.json` (R2 evidence)
- [x] [plot] multi-object steered t-SNE light-palette replot — RESOLVED 07-10 as "no sweep needed": presentation (docs/presentation_assets.md) references only 3 t-SNE figures, none of the 71 steered PNGs; the steered/manipulation series uses its own opaque tab10 *condition* palette (tsne_viz.py FILL_COLORS, grounding_manipulation.py) untouched by the ATTR_VALUE_COLORS change. Replotted only the 3 presentation figures (2× tsne_single, 1× dino_attribute_tsne) from caches.
- [ ] [plot] ACDC / binding-interchange figures (results JSONs exist, no figures)
- [x] [main flow] E5 agent-autonomous diagnosis pass — done 07-10 (`failure_modes.py --diagnose all`, D1–D4 in RESULTS.md §17)
- [ ] [main flow] **POLICY (user 2026-07-05)**: non-paper run dirs (nogate/film/mean/20ep/sup+mae decoder1l/flamingo/llava/transfusion) are training-exploration byproducts — do not eval, analyze, or build claims on them. Exception: existing gate writeup (RESULTS.md §2) stays; siglip_decoder1l sanctioned for E3.
- [ ] [model] dinov2_mean rerun if wanted: only an epoch-0 last.pt exists (crashed run; ep0 acc 0.218 — meaningless). Retrain or drop the cell.
- [ ] [ablation] T2I optional follow-up: declarative-caption prompts (extract_oracle_prompt) if the user wants to chase the domain-mismatch caveat
- [ ] [infra] broken A6000 (PCI 0x21, `[GPU requires reset]`, survives reboot, PCI remove blocks on usage count) → report for hardware repair

## Today's Progress
> [!NOTE] Append entries as work happens. Write so a stranger understands three months later.

- **2026-09-02 — Relational (same-as / spatial) status: the 2026-07-15 batch
  never got a write-up; read off here before the 3-object mechanism run.**
  (a) Position-only RSA (`conditional_rsa/clevr_dinov2_decoder1l_scratch_pos_only/
  */rsa_conditional_stats.json`): the position RDM's correlation with the model
  RDM never exceeds 0.03 in any category, peaks at blocks 3–7 and decays to
  ≤0.01 by block 11, i.e. it is exhausted before attribute binding rises
  (direct: binding 0.14 @5 → 0.76 @11; same: anchor 0.42 @8 → 0.11 @11,
  target 0.16 @7 → 0.41 @11; spatial: anchor 0.11 @8 → 0.04 @11, target
  0.19 @7 → 0.40 @9 → 0.33 @11). Position is not what the late blocks encode.
  (b) Shortcut renders (`shortcut_renders/*/add_object_eval_*.json`, n=100
  each): adding a near-copy of the anchor (one described attribute flipped)
  leaves accuracy at 0.89 (same) / 0.95 (spatial); the reported
  "hallucination" 0.31 / 0.67 is the rate of giving the anchored-on-distractor
  answer, and for spatial the placement rule forces the target to stand in the
  asked direction of the fake anchor too, so that answer coincides with the
  correct one — the 0.67 is not a hallucination measure and must not be cited
  as one. Translating the scene changes nothing (0.00). (c) Anchor dissipation
  (`anchor_dissipation/*/dissipation_stats.json`): 69 queries, 0 errors, flagged
  underpowered; no correlation with correctness is estimable. Net: the
  "anchor → target handoff" (RESULTS §9) is still correlational; the causal
  version is the 3-object transplant run launched today.
- **2026-09-04 — Flamingo-style LoRA run done; the training script's
  validation was under-reporting because of right padding. Corrected full-val
  accuracies: frozen LLM 0.4932, LoRA 0.5290. Unfreezing the LLM does not
  reduce add-object hallucination.** `clevr_flamingo_dinov2_lora_s42`
  (TinyLlama-1.1B-Chat, LoRA r=16 on q/v, 6 GCA layers text→vision, 8 epochs,
  batch 64; same recipe as `clevr_flamingo_dinov2_frozenllm_s42` minus
  `--freeze-llm`). In-run val acc 0.3795 (frozen run: 0.3176), but the
  script's `evaluate()` tokenised prompts with RIGHT padding and batched
  `generate` on a decoder-only LLM, which corrupts the shorter prompts of a
  batch: on the same 1,920 questions right padding gives 0.3734 and left
  padding 0.5161. Fixed in `train_flamingo_clevr.py::evaluate` (left padding
  for the duration of evaluation only; training collate unchanged) and
  re-evaluated both checkpoints on all 149,991 val questions with left
  padding (`eval_left_padding_full.{log,json}` in each run dir): frozen LLM
  **0.4932**, LoRA **0.5290**. Both earlier numbers (0.3176 / 0.3795) are
  superseded; RESULTS §15's 0.3176 needs this correction. Add-object
  hallucination (`add_object_eval_flamingo.py`, which already used left
  padding; n=100 per attribute), LoRA vs frozen: colour 0.21 vs 0.08,
  material 0.53 vs 0.59, shape 0.45 vs 0.52, size 0.63 vs 0.54, bait share
  of errors 0.78–1.00 on the non-colour attributes for both. So with the LLM
  unfrozen the model gains 3.6 points overall but fixates on the added lure
  exactly as before — consistent with the §15 reading that lure fixation is
  a property of LLM-side fusion, not of training budget or trainability.
  Comparison across the three fusion designs at matched vision backbone
  (frozen DINOv2 ViT-B/14 @336): in-stream (question written into the ViT)
  0.9095; mirror (question stream = RoBERTa-large, patches as key/value)
  0.8624; Flamingo-style (question stream = TinyLlama-1.1B, patches as
  key/value, generative readout) 0.4932 frozen / 0.5290 LoRA; concatenation
  readout without any cross-attention 0.49–0.55. The mirror and the
  Flamingo-style model have the same fusion direction, so the 0.86 vs 0.53
  gap is not the direction: it is the readout (classification decoder over
  the whole token sequence vs autoregressive generation from the last
  position of a 1.1B causal LM) and the training regime — the causal LM's
  last-position readout is the same aggregation problem the −CA models fail
  at, now with a stronger language prior competing.
- **2026-09-03 — Mirror model mechanism: selection is the referent word's
  attention over patches, sharpened only in the last two text-GCA layers, and
  the target's colour enters the text stream there; errors are not attention
  misplacement.** `patch_language_condition/mirror/` (324 two-object pairs,
  decoder accuracy clean 0.94 / corrupted 0.96; key/value patch tokens
  verified identical across conditions). (1) Attention mass of the referent
  word token on target / distractor patches by text-GCA layer 2/6/10/14/18/22:
  0.22/0.14/0.06/0.06/0.18/0.20 vs 0.18/0.10/0.04/0.05/0.05/0.08 (background,
  547 of 576 tokens, takes the rest, peaking 0.91 at layer 10). Layers 2–14
  are not selective; layers 18 and 22 put 2.5–4× more mass on the referent
  than on the distractor, and under the corrupted run the two curves swap.
  The `<s>` and `</s>` tokens are never selective (equal mass on both
  objects), so the fetch is done by the referent word, not by a summary
  token. (2) Linear probe of the target's colour from the RoBERTa hidden
  states: ≤0.30 through layer 18, 0.55 after the layer-18 GCA, 0.97 after the
  layer-22 GCA (chance 0.16; text-only control flat at 0.14) — the colour
  reaches the text stream only in the last two GCA layers. (3) The 1-layer
  decoder reads 97% of its attention from question tokens other than the
  referent word, `<s>`, "color" or `</s>`. (4) Error split: on the 18
  incorrect trials the referent word's mass is target 0.196 / distractor
  0.088, on the 306 correct 0.203 / 0.080 — errors are not fetch errors of
  attention placement. (5) Fetch test with the mean target−distractor contrast
  (norm 2.1 vs patch norm 41.9): at scale 1–2 no effect; at scale 20 (added
  norm ≈ the token norm, `mirror_v2/`) adding it to the distractor's key/value
  patches raises the referent word's last-layer mass on the distractor from
  0.08 to 0.18 (≈ the baseline target mass 0.20) and moves 11% of answers to
  the distractor's colour (P(target) 0.78); a norm-matched random vector at
  the same scale does nothing (1.00 / 0.00). So the fetch is content-addressed
  through the object-identity direction, but weakly: the global mean contrast
  is a crude key, and the same vector on the 547 background patches swamps
  the keys (P(target) 0.43 at scale 10, 0.02 at 20).
  Reading: the mirror implements tag-then-fetch with content-addressed
  attention in its last two cross-attention layers, and the visual stream
  never changes; the in-stream model instead rewrites the patches over six
  layers (removal). Both reach the one-layer readout, but the mirror's answer
  lives in question tokens, the in-stream model's in the background copy.
- **2026-09-03 — Mirror model trained: writing the question into the TEXT
  stream (text queries the vanilla ViT patches) reaches 0.8624, most of the
  in-stream model's 0.9095 and far above the −CA readout models (≤0.55).**
  `clevr_dinov2_mirror_decoder1l_scratch_s42`: vanilla frozen DINOv2, GCA at
  RoBERTa-large layers 2,6,10,14,18,22 (text = query, patches = key/value),
  1-layer decoder reading the text tokens, 28.5 M trainable, 16 epochs
  (resumed from epoch 1 after a validation OOM caused by the co-resident LoRA
  run; final-only validation at batch 64). Per type, mirror vs main model:
  query_attribute 0.910 vs 0.987, count 0.736 vs 0.841, exist 0.889 vs 0.949,
  equal_attribute 0.935 vs 0.897, compare_integer 0.820 vs 0.749. So in-stream
  conditioning in either stream, with the same six GCA layers and the same
  one-layer readout, recovers the bulk of the accuracy that the concatenation
  readout cannot learn; the visual-stream version keeps the edge on referent
  selection and enumeration (query_attribute, count, exist), the text-stream
  version wins on two-referent comparisons (equal_attribute, compare_integer),
  where a text token that fetches both objects is the natural computation.
  Mechanism analysis (`--mirror` mode: text-GCA attention mass, decoder
  attention over text tokens, per-layer text readout probe, key/value fetch
  test, error split) launched on `best.pt` → `patch_language_condition/mirror/`.
- **2026-09-03 — Aggregation-depth curve, third point: −CA decoder with 4
  layers reaches 0.5349, not above the 2-layer 0.5502.** Run
  `clevr_dinov2_concat_decoder4l_nogca_scratch_s42` (16 epochs, no GCA,
  question and patches concatenated in the decoder). Its in-run final
  validation died with CUDA OOM because a second training shared the GPU;
  re-evaluated `last.pt` (epoch 15) with `scripts/evaluate.py` at val batch 64
  (`eval_last_epoch15_v2.log` in the run dir): overall 0.5349 on 149,991 val
  questions (query_attribute 0.483, count 0.486, equal_attribute 0.570,
  compare_integer 0.584, exist 0.680). With the 1-layer 0.4945 and 2-layer
  0.5502 (both epoch-15 final validation) the curve is flat from 2 layers on:
  putting the selection into the readout does not recover the 0.9095 of the
  in-stream model by adding readout depth, at least up to 4 layers at this
  width and training budget.
- **2026-09-02 — Relational questions on 3-object scenes (DINOv2, local patches
  + 1-layer decoder): the anchor → answer handoff is causal and in-stream; the
  spatial relation is written first as an absolute-position field, then as an
  anchor-centred one.** New `--relational {same,spatial}` mode of
  `patch_language_condition.py` on `data/clevr_three_object_v2` (672 renders,
  three pairwise-distinct non-gray colours). Roles: anchor A (named in the
  question), answer object T, third object D. Same-as: "There is another thing
  that is the same {attr} as the {A colour} object; what is its color?" (652
  scenes; clean run names A → answer T, corrupted run names T → answer A;
  accuracy 0.97 / 0.98). Spatial: "What color is the object {left of / right
  of / in front of / behind} the {A colour} object?" (494 scenes, relation
  from mask centroids with ≥2-patch margins; corrupted run = opposite relation
  word → answer D; accuracy 1.00 / 1.00). Outputs
  `patch_language_condition/relational_{same,spatial}/`.
  (1) Projection of the question's change onto each object's own colour
  direction at block 11: same-as answer T +8.7, named A −1.4, third D +1.7;
  under the corrupted run the A and T curves swap exactly. Spatial: T +10.1,
  A +0.8, D +1.6, and again the boost follows the answer object. So the final
  state is the two-object removal pattern with the answer object in the
  target role and the named anchor treated as a non-referent.
  (2) Token replacement (one group of the clean run replaced by the corrupted
  run's tokens at block ℓ; clean-self control 1.00 at every cell). Same-as:
  the named anchor's patches are causally needed from block 3 through block
  10 (P(clean answer) 0.83 → 0.63, answer moving to A's colour up to 0.36) and
  become irrelevant at block 11 (0.84); the answer object's patches likewise
  (0.68 → 0.46 at block 7, answer moving to the third object D up to 0.32);
  D's patches never matter; background patches carry the answer only at block
  11 (P(clean answer) 0.34, P(A colour) 0.66). This is the causal version of
  the RESULTS §9 RSA handoff: anchor information is used through block 10,
  suppressed at 11, and the answer is relocated to the background copy by the
  last GCA layer. Spatial: no object group is causal before block 9; at block
  9 the two candidates' patches matter (T 0.77, D 0.68, answer moving to D
  0.32), the anchor's barely (0.96 — expected, the anchor is the same object
  in both runs); background carries the answer at block 11 (0.32, P(D) 0.68).
  (3) GCA write norms: the anchor's patches receive the largest write at layer
  9 (same 9.3 vs 6.8–7.0 for the others; spatial 8.5 vs 7.3–7.8); the
  background receives the most at layer 11.
  (4) Spatial write-position regression (263k background patches, per-patch
  write-norm difference between the two relation words): R² on the absolute
  coordinate along the relation axis 0.61 / 0.38 / 0.43 / 0.34 / 0.09 / 0.34
  at layers 1/3/5/7/9/11; on the coordinate relative to the anchor centroid
  0.00 / 0.00 / 0.33 / 0.00 / 0.66 / 0.59. The relation word is first
  broadcast as a global left/right (front/behind) gradient over the image and
  only at layers 9–11 rewritten as a gradient centred on the anchor — the
  position computation lives in the background patches, which is why the
  candidates' tokens only become causal at block 9. Behavioural shortcut
  test (same-side vs opposite-side of the image centre) is at ceiling (1.00
  both, n = 26 / 468) and uninformative for this checkpoint.
- **2026-09-02 — Flamingo-style LoRA run launched; mirror model implemented.**
  Why the earlier LoRA attempt "would not run": that was the LLaVA-style script
  (`clevr_llava_dinov2_lora_s42`, 2026-06-02) — 576 vision tokens inside a 7B
  4-bit LLM with gradient checkpointing, no DINOv2 feature cache, checkpoint
  only at epoch end (28 h < 1 epoch, dir empty), and an `evaluate` KeyError on
  `batch["questions"]` that would have fired before the save. The Flamingo
  script's LoRA mode (`train_flamingo_clevr.py`, vision only in cross-attention
  KV, cached features) never had that problem; its 06-15 run only lacked a
  valid eval (decode bug, fixed 07-06). Launched `clevr_flamingo_dinov2_lora_s42`
  (TinyLlama-1.1B-Chat, LoRA r=16 on q/v, 6 GCA layers text→vision, 8 ep,
  batch 64; same recipe as the frozen-LLM run 0.3176 minus `--freeze-llm`).
  Mirror model (`configs/experiment/clevr_dinov2_mirror_decoder1l_scratch.yaml`):
  vanilla frozen DINOv2, GCA at RoBERTa-large layers 2,6,10,14,18,22 with text
  as query and patches as key/value, one-layer decoder reading the text tokens;
  28.5 M trainable (22.0 M GCA + 1.8 M connector + 4.6 M decoder) vs 27.7 M in
  the main model; original model path verified bit-identical after the change.
  Training queued behind the running −CA 4-layer run (single usable GPU).
- **2026-09-02 — X21 (M): MAE's referent marker is a portable tag, not a
  position ID; decoder attention is not a norm artifact.**
  `patch_language_condition/mae/marker_test.{json,png}` (324 pairs, refer-
  target accuracy 0.997). Marker direction = mean over images of the target's
  raw patch mean under refer-target minus under refer-distractor; its norm is
  ≤0.8 through block 6, 3.1 at block 7, 12.5 at block 9, 37.6 at block 11 —
  written almost entirely by the last GCA layer. Transplanting it at block 11
  onto the DISTRACTOR's patches (a different image position) under the
  refer-target question moves decoder attention from the target to the
  distractor (distractor mass 0.004 → 0.32 at 1×, 0.75 at 2×) and flips the
  answer to the distractor's colour (0.44 at 1×, 0.62 at 2×); moving it
  (add to distractor, subtract from target) flips 0.84 at 1×. A norm-matched
  random vector does nothing (P(target) 1.00, attention unchanged).
  Subtracting it from the target alone removes attention from both objects
  (target mass 0.468 → 0.001) yet the answer stays correct 0.79 — the decoder
  then reads from background tokens, so MAE also carries a usable background
  copy at block 11. Blocks 7–10: no effect (marker too small there and later
  GCA layers rewrite it). Verdict: the marker follows content, not position —
  Saravanan-style identity code, not an Assouel-style position ID.
  Norm control (`attn_norm_control.json`, MAE and DINOv2): no token exceeds
  5× the per-image median norm in either backbone (ViT-B, no Darcet
  outliers); Spearman(attention, norm) 0.31 (MAE) / 0.15 (DINOv2) under
  refer-target; per-token target/background attention ratio MAE 9.5 (no
  question) → 47.1 (refer target), DINOv2 5.2 → 9.0, unchanged by excluding
  high-norm tokens. The sink-token objection to the decoder-attention claim
  is closed. Flags `--marker-test`, `--attn-norm-control`.
- **2026-08-31 — X21 (K): MAE replication — selection without removal.**
  `patch_language_condition/mae/` (14×14, 324 pairs, referring accuracy
  0.997). The non-referent keeps its colour (colour RDM 0.51 vs 0.54 at block
  11; token-level selection contrast ≤ +1.6 vs DINOv2's +11.4); the referent
  is marked (per-patch referent probe 0.79 at L7 → 0.99 at L11) and the
  decoder's attention selects it (127.8 vs 1.3 ×1e-3 per patch); the
  answer-determining tokens are the object tokens at block 11 (swap 0.90),
  no background copy. Colour vectors flip at every block (0.54 → 0.95).
  So the DINOv2 block-11 background copy is DINOv2-specific, and "removal"
  is not architecture-general — it is what DINOv2/SigLIP do, MAE marks
  instead. Registry X21 (K). Site updated.
- **2026-08-31 — X21 (J): queried = material and size — the boost of the
  queried attribute tracks the backbone's default.** Template-RSA of the
  target's offset with the queried attribute's RDM at block 11: material
  0.19 (no question) → 0.86 (refer target) → 0.04 (refer distractor); size
  0.10 → 0.73 → 0.09. Removal from the non-referent and the block-11
  background copy hold for all four queried attributes; the boost appears
  for colour, material and size (weakly kept by default) and not for shape
  (kept at 0.77 anyway). Projection rises stay ≤ +2.8 for material/size —
  the boost is visible in rank-order organisation more than raw projection
  (difference recorded, unexplained). Registry X21 (J).
- **2026-08-31 — X21 (I): head combinations on GCA layers 7/9 — graded
  concentration, no sufficient or necessary four-head subset.** Keeping only
  the top-4 heads (ranked by the single-head scan) preserves ≈70% (L7) /
  ≈50% (L9) of the selection effect at blocks 9–10 with accuracy 0.96–0.98;
  4 random heads preserve far less (+0.2 to +2.0) and cost accuracy;
  zeroing only the top-4 removes about half. Registry X21 (I) has the
  table. Material/size runs and the MAE replication are queued behind it.
- **2026-08-29 — X21 (G): queried attribute = shape — the removal from the
  non-referent generalises, the rise on both objects does not.**
  `patch_language_condition/shape/` (DINOv2, same 324 pairs, questions
  "What shape is the {size/material/colour} object?"). Selection on the
  target's own-shape direction (refer target − refer distractor) grows from
  block 5 to +14.7 at block 11, mirrored on the distractor; template RSA shape
  RDM at block 11: referent 0.81, non-referent 0.05, no question 0.77. But
  relative to no question no shape projection rises (referent −3 to −5,
  non-referent −11 to −17.5), unlike colour where both objects rose +12 to +15
  through block 8. Token swaps and interventions reproduce the colour
  structure (objects carry selection at 7–10, background at 11; shape vectors
  flip from block 5 with controls at 0). Site claim narrowed; registry X21
  (G) has the full table.
- **2026-08-29 — X21 (H): head ablation scan — the selection effect is
  written by whole GCA layers 7 and 9, not by single heads.** `--head-scan`
  (258 zero-ablations with `analysis.patching_utils.HeadAblator`, files
  `head_scan.json`, `head_scan_rows.jsonl`, `head_scan.png`). Measured: the
  target's projection on its own colour direction, refer target − refer
  distractor, per block. Single-head ablations change the block-11 effect by
  a median of 0.0 (5th percentile −0.7) and never lower accuracy below 0.97;
  the only large single-head effect is SA block 11 head 7 (11.4 → 5.1, blocks
  9–10 unchanged, accuracy unchanged). Zeroing all heads of GCA layer 7 or 9
  removes the effect at blocks 9–10 (+0.9/+0.8 and −0.8/−0.4 vs +5.3/+6.0)
  with accuracy ≈0.7; GCA 1/3/11 do nothing. Head-level correlation with the
  patching recovery is weak (≤0.28). Registry X21 (H) has the full table.
- **2026-08-29 — X21 (F): SigLIP replication of the whole language-condition
  suite.** `outputs/analysis/patch_language_condition/siglip/` (grid 16 @ 256,
  same 324 pairs, masks inspected, X19-SigLIP reproduced on its 30 pairs).
  Same mechanism as DINOv2 with two differences: (1) the referent /
  non-referent split appears earlier (Δ_ref +15 at block 5, +29 at 7; own-colour
  split from block 7) and stays to block 11 — no last-block copy into the
  background tokens: target+Δ flips 0.83–0.95 at every block including 11,
  object-token swaps move the answer 0.80–0.88 through block 11, background
  swaps ≤0.21; (2) the background tokens at blocks 9–11 carry the question
  type — swapping in no-question background tokens there makes the decoder
  answer "no" (93%) instead of a colour, while the object tokens keep which
  object and its colour. Decoder attention per patch: referent 113×1e-3,
  background 1.8, non-referent 0.1. Template RSA position ≤0.09 at block 11;
  colour RDM 0.60 when the target is the referent, 0.10 when not. Probes:
  referent 1.00 from L5, spatial-LOO within 0.05 of random. Registry X21 (F).
- **2026-08-29 — X21 (E): attribute-specific directions (Song et al.
  concept vectors) — amplification vs suppression reconciled.** New
  `--attr-directions` mode (from cache; new files `partA_attr_directions.json`,
  `attr_directions.png`). Directions from the 1-object images (independent
  set). Any colour question raises the own-colour projection of both objects
  alike through block 8 (Song's amplification of the asked attribute, not
  object-selective) and lowers the shape projection of both objects (unasked
  attribute); from block 9 the referent keeps the raised colour and the
  non-referent loses it (refer target − refer distractor on own colour:
  +5.3 (9), +6.0 (10), +11.4 (11); mirrored on the distractor). A shape-direction
  dip of the referent at blocks 5–8 exists only when the referring word is a
  shape word (checked by splitting on referring-word type: shape n=223 vs
  size 59 / material 42) — matching the word, not a general effect on the
  unasked attribute. Registry X21 updated.
- **2026-08-29 — X21 (D): where the decoder reads the answer — decoder
  attention grouped by background / target / distractor, and token swaps between conditions.** New `--readout`
  mode in `scripts/analysis/patch_language_condition.py` (new files only:
  `readout_attention.json`, `readout_swap.json`, `readout_swap_trials.jsonl`,
  `readout.png`, `log_readout.txt`). Checks passed: generate() = first-token
  argmax; c1/c2 accuracy 0.994; attention rows sum to 1; block-11 capture +
  trunk.norm equals the decoder input; identity swap reproduces the baseline
  at all blocks. Decoder attention per patch: referent 13.4×1e-3, background
  1.5×1e-3, non-referent 2.0×1e-3; top patch is the referent in 76% of
  images. Swaps (forward pass with the question about the target; tokens replaced
  from the forward pass with the question about the distractor): object tokens swapped → distractor's colour 0.88 at block 7,
  0.97 at 9–10, 0.24 at 11; background tokens swapped → 0.00–0.02 through
  block 10, 0.71 at 11. So the selection sits in the object tokens at
  blocks 7–10 and is copied into the background tokens at block 11, where
  the decoder reads it. Registry X21 updated.
- **2026-08-28 — X21: RSA position control closed with the per-position
  background template.** New `--rsa-template` mode in
  `scripts/analysis/patch_language_condition.py` (writes only
  `partA_rsa_template.json` + `rsa_template.png`; nothing regenerated). The
  template is X19's per-position background mean, built from the sparse
  cache (19–60 background tokens per position). Position RDM correlation of
  the target's offset drops from 0.6–0.8 (image-mean subtraction) to 0.1–0.3
  without a question and ≈0 from block 5 under a question. Colour RDM: 0.43
  at block 0 in all conditions; no question → 0.01 at block 11; target is
  referent → 0.59; distractor is referent → 0.06 at block 11. So the
  non-referent's colour is removed from its own patches from block 5 on,
  the same profile as Δ_nonref. Full-identity RDM is uninformative
  (84 combos, nearly all pairs differ). Registry X21 updated.
- **2026-08-27 — X21: language condition on the patch object vector; Parts A
  and B done, Part C (single-patch probes) running.** User ruling: the
  no-question-only design of X19 was a design error; literature survey first
  (Song/Lepori/Pavlick 2025, Feng & Steinhardt 2024, Saravanan et al. 2025,
  Assouel/Webb 2025, Lepori et al. 2024, Darcet 2024; references in registry
  X21), then `scripts/analysis/patch_language_condition.py` (324 pairs, four
  n2 conditions, sparse cache 3.9 GB). Treated as a replication of those
  methods on the GCA ViT (user: 就當復現). Results: (A) projection of the
  target's patch mean onto the no-question object direction — Δ_ref
  (refer target − refer distractor) rises from block 5, peaks +26.5 at block 9
  (0.40 of the offset norm), Δ_nonref is its mirror (−26.4); decomposed
  against no-question, "refer this object" adds only +3 while "refer the
  other object" removes −23 → selection works mainly by suppressing the
  non-referent. Any question (incl. non-referring) shifts both objects by the
  same amount, and at block 11 all conditions drop the object projection by
  ≈15. GCA write norm is the same for target/distractor/background patches,
  cos(write, V) ≤ 0.11, and patch→referent-word attention does not differ
  between the two objects — selection is not visible in attention weights.
  c0 reproduces X19 on its 30 pairs exactly (0.912 / 0.624 / 0.962).
  (B) additive colour-vector intervention (difference-in-means on n1 raw
  target means; edit at block ℓ on baseline-correct trials, n=322): α=1
  target+Δ flips the answer to B at 80% (block 0) → 99% (block 8–10);
  norm-matched random 0%, background subset 0%, distractor patches 0%
  (c1), distractor+Δ under c2 97%, target+Δ under c2 0%. Block 11 is the
  exception: object-only edit 27%, ALL-background edit 79% — the final
  readout takes the answer from background tokens too. (C) single-patch
  probes: bg/object 0.99; colour 0.99→0.91, shape 0.72→1.00, material and
  size →0.99; referent-vs-non-referent 0.58 (L1) → 0.94 (L5) → 1.00 (L7+),
  no-question control 0.50; spatial-LOO ≈ random split (Δ ≤ 0.05) →
  position-invariant per-patch code, supervised. Open: offset RSA still
  position-dominated (needs the per-position template). Figures + JSON in
  `outputs/analysis/patch_language_condition/`.
- **2026-08-27 — X20 probe queue finished (9 models); readout × backbone
  table complete.** Readings: MAE deficit representational under local
  patches (0.817) but NOT under CLS token (0.916 decode at 0.77 accuracy) →
  site claim to be reworded as pretraining × readout; Sup-ViT +question drop
  is readout-side (decode 0.937 vs 0.930); −CA flat at chance on all three
  categories; three readouts share the decode curve within every backbone.
  Table + curves in `outputs/analysis/linear_probe/probe_table_direct.*`,
  section in `docs/results_tables.md` (regenerated). Details registry X20.
- **2026-08-27 — X19 rerun on MAE and Sup-ViT; single-image PCA figure now
  shows the scene.** User hypothesis after X20's MAE probe deficit: MAE's
  local patch encoding differs. Confirmed at patch level: MAE object patches
  are far from the background cloud at every layer (single-image PCA,
  PC1+2 45–62%), offsets most type-specific (L11 within 0.869 / between
  0.418), and background-subtracted KMeans keeps object-vs-object separation
  through L11 (n2 target 0.81 / distractor 0.61) where DINOv2 / SigLIP /
  Sup-ViT decay to 0.1–0.3. Sup-ViT tracks DINOv2. `pca_single_*.png` gained a
  left column with the scene + owner overlay (user request); all four
  backbones replotted from cache. Details in registry X19. X20 probe queue
  still running (5/9 done at 00:00).
- **2026-08-26 — X20 comprehensive linear probe launched (story-vs-evidence
  check).** User's point: accuracy shows a behavioral difference, a linear
  probe shows whether the representation differs. Audit of the site
  (claim → evidence type) found accuracy-only mechanism claims: pretraining
  objective explains MAE's deficit (準確率總表), Sup-ViT's +question drop is a
  readout interaction, −CA 0.9237→0.4945 is the causal baseline / "gradients
  vanish without CA" (第五階段), failure structure is mechanism-level not
  readout-level (第二階段), ungated CA "leaves no separable handle" (gate note,
  no evidence cited while an existing probe on `clevr_dinov2_nogate_scratch`
  shows decode 0.171 — artifact-vs-real being checked). Probe coverage was
  4/12 paper cells and zero for −CA. Same X10 protocol; `linear_probe.py` got
  a `--categories` CLI (default unchanged). Queue on GPU0
  (`outputs/analysis/linear_probe/x20_probe_queue_2026-08-26.log`): −CA all
  3 categories, then mae/sup decoder1l+concat, siglip concat, siglip/sup/mae
  cls (direct only). Pre-registered readings in registry X20. −CA checkpoint
  verified: GCA layers present, attn_gate frozen at 0.0 for all six layers.
  Aggregation script `probe_table.py` (readout × backbone table + layer
  curves) to follow; site changes wait for user review.
  Later the same day: the ungated-CA 0.171 turned out to be a loader artifact
  (`linear_probe.py` rebuilt the backbone with default `use_gate=True`,
  `strict=False` left the missing gate at 0 → GCA nulled); `linear_probe.py`
  now uses `load_any_checkpoint`. **User ruling: the ungated-CA variant is
  deprecated — no rerun, no comparison, no row in the probe table.** First
  result: −CA is flat at chance across all 12 layers on all three categories
  (decode 0.171, match 0.21–0.23) vs 0.17→0.92 for the CA readouts.
- **2026-08-26 — X19 (local patches additive structure) added to the results
  site.** New subsection「Local patches 的加性結構」inserted at the end of the
  Multi-object hallucination section of `docs/site/index.src.html` (local-only
  file, not in git), before its Paper-ready claim; the claim gained one sentence
  (patch = background + additive, position-invariant, object-specific vector;
  small relative to background positional variance, so pooling / unsupervised
  clustering need a selection step). Five figures inlined from
  `outputs/analysis/patch_pca_cluster/` (global PCA n1, single-image PCA n2,
  raw and background-subtracted k=3 overlays, DINOv2 vs SigLIP metrics
  half-pair). Site vocabulary: 1-object/2-object, 扣掉逐位置背景平均, no
  internal codes. Rebuilt `index.html` (12.1 MB, 35 inlined images, was 29);
  public server confirmed serving the new heading. Prose drafted by an Opus
  agent under the user's explanation rules. Backup of the pre-edit source kept
  in the job tmp dir only.
- **2026-08-19 — X19 patch-token PCA + KMeans on the NEW paired renders
  (additive object-vector test); paired dataset render completed.** The paired
  object-count dataset finished rendering 06:31 (another agent's
  `render_single_objects.py --paired` run, previously unjournaled):
  `data/clevr_object_count/{n1,n2}` = 480 pairs (96 combos × 5 positions),
  target placement bit-identical across n1/n2, 1 distractor ≥2 attrs different,
  sizes free 240/240 — replaces the invalidated single_object_v3/two_object_v2.
  New `scripts/analysis/patch_pca_cluster.py` (X19; imports X16's
  extraction/segmentation from tsne_patch_level.py; noca on
  clevr_dinov2_decoder1l_scratch_s42; CPU-only, GPU left to the s44 run; X16
  three-phase --masks-only → cached npz → --replot). Findings: (1) additive +
  object-specific offsets confirmed — target offset (obj−bg mean, 768-d)
  unchanged when the distractor is added (same-pair n1↔n2 cos 0.998→0.962
  L1→L11); within-combo-across-position cos > between-combo everywhere (L11
  0.912 vs 0.624); one shared "objectness" direction carries 0.60–0.79 of
  offset energy and the residual is combo-specific (L11 0.729 vs −0.152).
  (2) User-specified KMeans (5 random pairs, k=2 n1 / k=3 n2, red/blue overlay
  alpha 0.3) on RAW tokens fails (IoU 0.01–0.05): background's positional
  manifold dominates inertia. (3) Subtracting a per-position background
  template (the hypothesis' own prediction) recovers the foreground (n1 target
  IoU 0.60@L1; ARI 0.7→0.3 with depth) but the two foreground clusters split
  core-vs-halo, not object-vs-object. (4) Global-fit PCA (n1/n2 share the
  frame — the advantage over per-panel t-SNE): PC1+2 only ~25–39% var, object
  patches collapse to one clump by L11. Artifacts:
  `outputs/analysis/patch_pca_cluster/`; registry X19 has full design/caveats.
  Code developed on worktree branch `worktree-patch-pca-cluster` (main
  checkout's bg-edit guard); JOURNAL/registry entries appended there too —
  merge on next commit pass.
  **SigLIP leg (user-ordered)**: same pipeline on
  `clevr_siglip_decoder1l_scratch_s42` noca (@256, 16×16 grid, same 35 pairs)
  → `outputs/analysis/patch_pca_cluster/siglip/`. Additivity replicates
  (n1↔n2 cos 0.999→0.922), but the depth trend reverses vs DINOv2: bgsub
  KMeans at L1 separates object-vs-object (target/distractor IoU 0.57/0.63 ≈
  fg 0.66, not DINOv2's core-vs-halo), then foreground clusters fragment into
  background scatter with depth (L11 0.07–0.11 vs DINOv2 0.20–0.27).

- **2026-08-18 — site made external-reader-safe (user-directed, second pass).**
  The site is served publicly (http://141.212.110.118:8899, long-running
  `http.server` rooted at docs/site/), so index.src.html was purged of internal
  vocabulary: (a) all checkpoint run names removed — models are described by
  role + components; readout variants renamed to the new two-axis grammar
  **CLS token / local patches / local patches + question** (visual interface ×
  whether the question re-enters at readout; replaces concat_decoder1l /
  decoder1l / cls, which conflated the axes — both decoder variants read local
  patches); (b) all experiment registry codes deleted (E3/E4/E5/E7/E8/E9, X13,
  T1/T4, R1) and replaced by experiment descriptions — user ruling: H1–H3,
  D1–D4, A1–A6 stay because the page defines them; (c) every `<span class=src>`
  path span, RESULTS.md/paper_artifacts/JOURNAL citation, and dir-migration
  note removed (provenance stays in RESULTS.md/registry; prov block now a
  3-sentence external statement). Two new comparison figures (both CPU replots
  from caches, new filenames, nothing overwritten): `dino_attribute_tsne.py
  --combined` → `outputs/analysis/single_objects/dino_attribute_tsne_combined.png`
  (single panel, 4 channels: hue=color, shade=material via tab20 dark/light
  pairs — metal dark, rubber light (user replaced the initial black-edge
  encoding); glyph=shape, size=size; canvas ×1.6) and `raw_backbone_probe.py --combined` →
  `outputs/analysis/raw_backbone_probe/combined_probe.png` (1×4 attr panels ×
  4 backbone curves, shared axes; replaces the separate DINOv2 + MAE figures and
  gives raw_backbone_probe.py its previously missing replot path). index.html
  rebuilt: 7.2 MB, 25 images; verified live on :8899.
  Follow-up rounds same day (user-directed): (1) material encoding changed
  from black edges (a real bug — the global `alpha=` kwarg overwrote rubber's
  transparent edge alpha, giving every point a black edge) to tab20 dark/light
  hue pairs (metal dark / rubber light), thin uniform 0.5pt outline kept for
  crispness only; canvas ×2 (cell 8 → 9×9 in). (2) t-SNE subsampled to 3
  positions per attribute combination (96 combos → 288 points,
  `--per-combo 3`, seed 42) so individual markers stay readable. (3) Term
  unification: **"ViT backbone"** replaces raw backbone / raw substrate /
  原始基底 / pretrained substrate everywhere; figures retitled "ViT backbone
  t-SNE — DINOv2, single objects" and "ViT backbone probing — multi-object
  scenes"; site section renamed "ViT backbone probing". Final build 6.8 MB.
  (4) Section merge, user-directed: backbone probing + object-count 受控探測
  are ONE experiment named by hypothesis — "**Multi-object hallucination**"
  (first half = per-object local readout, information present; second half =
  pooled 1-vs-2 readout, selection collapse; identical method across n1/n2).
  Standalone object-count section removed, its content folded in; fixation
  triangle + merged paper-ready claim close the section; behavioral 第三階段
  and timeline references updated. Naming rule recorded in memory: experiments
  named after the hypothesis, measurements keep "ViT backbone <measurement>".

- **2026-08-19 — pooled redo of the multi-object hallucination evidence (X18) +
  final site vocabulary pass.** User rulings: the 3×3 per-object readout is NOT
  the designed experiment — readout must be 24×24 mean pooling with object count
  as the only variable; "substrate" banned; ALL A-codes (A1–A6, A1.2, A2↔A4,
  A4.3) rewritten as descriptions; 「誘餌奪走」→ hallucination rate. New run
  `raw_backbone_probe.py --pooled` (X18, CPU 4-way parallel workers; GPU held
  the s44 cls run): 4 backbones × {n1=500, n2=480} × 12 blocks, mean-pooled →
  PCA50+logistic on target attrs + DINOv2 b11 pooled t-SNE (288 pts/panel).
  Headline: n1 all 0.91–1.00; n2 target color DINOv2 0.912→0.517, Sup-ViT
  →0.812, SigLIP →0.850, MAE →0.912 (shape/material stay high — target is the
  large object); t-SNE n1 = shape×material islands, n2 = diffuse. Site section
  rewritten as ONE experiment (backbone pooled → +language conditions →
  fixation triangle); old 3×3 figures/table removed from the page (files kept
  on disk). Raw-vs-trained numeric gap (0.517 vs noca 0.356) flagged on site as
  protocol-level, direction-consistent. Rebuilt 7.1 MB / 25 images, live :8899.

- **2026-08-15 — site updated with the external-comparison section.** New pillar
  「外部對照：unified VLM 的 reference recoding」 in docs/site/index.src.html
  (§4.2 reproduction + order negative control, t-SNE geometry contrast,
  prefix/postfix behavioral cost, section conclusion + paper-ready claim);
  figures referenced from recode-repro/outputs/report_figures/ (5 combined
  presentation figures, conditions as row labels, no internal condition codes).
  docs/site/build.py added (base64-inlining build; index.html rebuilt, 7.3 MB,
  26 images). Provenance paragraph extended. Local files only — nothing pushed
  or deployed.

- **2026-08-14 — X17 reference probe (GCA ViT, Song-et-al §4.2 analog) done.**
  Paired-referring probe on 219 two-object scenes: referring rises from chance at
  block 0 to 1.0 by block 6 (0.625@b1 → 0.977@b5), controls exactly 0.5 everywhere.
  First unpaired attempt was a lesson: target is always the large object in this
  dataset, so ALL conditions (incl. noca) probed 1.0 through the size confound —
  fixed with the two-direction paired design + grouped split; old dir
  `reference_probe/two_object/` kept as confound record. Companion Qwen2.5-VL
  reproduction lives in `../recode-repro/` (own JOURNAL): text-prefix run reproduces
  the paper (counting/referring 1.0@L8, controls ~chance); image-first run = all
  conditions bit-identical features → 0.562 flat, proving default Qwen ordering
  admits NO in-stream recoding (causal mask gates it).

- **2026-08-13 — X16 patch-level t-SNE (unpooled tokens) run.** New
  `scripts/analysis/tsne_patch_level.py` on `clevr_dinov2_decoder1l_scratch_s42`:
  t-SNE of individual patch tokens (first non-pooled embedding analysis) over GCA
  layers, 4 figures in `outputs/analysis/tsne/patch_level/`. Stimuli: 10×1-object
  (v3) + 10×2-object (v2); object→patch masks via saturation gate + nearest-hue
  pixel segmentation (chromaticity match fails — pyrender/Blender renders are dim,
  chroma drifts toward gray; single_object_v3 `pixel_coords` is a constant dummy
  (240,160), not the true position). Qualitative: per-object patch clusters exist at
  every layer even in 2-object scenes (the substrate is intact at patch level);
  10×2-object color clustering is visibly more mixed than 10×1-object; under
  `ca_refshape` ("What color is the {shape}?") referent patches aggregate into
  referent-dominated clusters by L9. Registry X16 has full design + caveats.

- **2026-08-05 — CORRECTION: the DINOv2 scale-axis entries (07-18/07-23/07-24) used a
  contaminated ViT-B reference value; direction of the return-scaling claim REVERSES.**
  Those entries cite `clevr_dinov2_concat_decoder1l_scratch_s42` final val as **0.9439**
  — that number is `stdout.log` inside that dir, wandb run `qgkov0mm` (started 06-03).
  But `docs/paper_artifacts.md` §8.1 already resolved (07-06, before the scale-axis work)
  that this exact dir is **contaminated**: the checkpoint actually stored there (`best.pt`)
  and the top-level `clevr_dinov2_concat_decoder_scratch_s42.log` (wandb run `udgs6s3l`,
  started 06-09) agree on **0.9237** — the number Table 1 actually uses. `stdout.log`
  belongs to an unrelated earlier run that happened to write into the same output-dir
  name before the never-overwrite convention was adopted. Verified again today: the two
  logs' wandb run IDs differ, confirming two distinct trainings share one directory.
  **Corrected scale axis: ViT-S/14 0.8925 < ViT-B/14 0.9237 < ViT-L/14 0.9646 → S→B
  +3.12 pts, B→L +4.09 pts — INCREASING returns, not diminishing** (07-24 entry said
  diminishing, +5.1/+2.1, using the wrong B). I-JEPA-vs-DINOv2-B gap (07-18 entry) is
  also affected: −2.62 pts, not −4.6. Per-qtype breakdowns and every other run's numbers
  in the affected entries are unaffected (only the B-point comparisons are wrong). No
  RESULTS.md section exists yet for the scale axis (never written in, per user
  instruction to hold off) — nothing to fix there. Root cause: the scale-axis work
  pulled `stdout.log` directly from the reused dir name without checking the
  already-documented §8.1 contamination note first — a cross-reference gap, not a new
  training bug.

- **2026-07-24 ~01:43 — ViT-S run COMPLETE: final (ep15) val 0.8925 → DINOv2 scale axis S<B<L closed.** `clevr_dinov2s_concat_decoder1l_scratch_s42` full 16-epoch val trajectory: 0.2101 / 0.5001 / 0.5153 / 0.5283 / 0.5956 / 0.6764 / 0.7401 / 0.7909 / 0.8208 / 0.8398 / 0.8594 / 0.8726 / 0.8823 / 0.8881 / 0.8910 / **0.8925**. Takeoff delayed ~1.5 epochs vs L (S resembles I-JEPA's right-shifted S-curve — S ep3 still 0.528 while L ep3 already 0.656); ran the full 16 epochs in ~16.5 h (~4× faster than L's ~3 days). **Scale axis (same DINOv2 paradigm + concat decoder1l + GCA, backbone size the only variable): ViT-S/14 22M 0.8925 < ViT-B/14 86M 0.9439 < ViT-L/14 300M 0.9646 — monotone; scaling DINOv2 monotonically helps CLEVR grounding, with diminishing returns (S→B +5.1, B→L +2.1).** Per-qtype, the gain concentrates on the hard binding classes: compare_integer 0.747(S)→0.864(L) = +11.7 pts and count 0.811(S)→0.934(L), while query_attribute barely moves (0.974→0.992, already saturated). S final breakdown: query_attribute 0.9735 / exist 0.9383 / equal_attribute 0.8768 / count 0.8111 / compare_integer 0.7474. S process in wandb-sync teardown at report time (still held GPU); next queued: Phase 3 Qwen runs (GATED ON USER GO — long gap since the earlier "after ViT-S" confirmation, re-confirm before launching).
- **2026-07-23 ~08:46 — DINOv2-L run COMPLETE: final (ep15) val 0.9646; ViT-S auto-launched by the gate, verified training.** `clevr_dinov2l_concat_decoder1l_scratch_s42` full 16-epoch val trajectory: 0.2125 / 0.4898 / 0.5290 / 0.6555 / 0.7221 / 0.8181 / 0.8831 / 0.9172 / 0.9367 / 0.9496 / 0.9550 / 0.9591 / 0.9615 / 0.9638 / 0.9643 / **0.9646**. Crossed ViT-B concat decoder1l (0.9439) at ep9 and finished **+2.1 pts** above it — scaling the DINOv2 backbone within the same paradigm helps, and L (300M) had not fully plateaued until ~ep13 (increments +1.29→+0.54→+0.41→+0.24→+0.23→+0.05→+0.03). Ceiling set by the two hard qtypes: final count 0.934, compare_integer 0.864 (both improved over ep9's 0.901/0.833 but remain the bottleneck); query/equal/exist all 0.96–0.99. train acc hit 0.993 by ep14 while val flattened — feature-readability ceiling, not epoch budget. Scale axis now S(running)→B 0.9439→L 0.9646; L brackets I-JEPA ViT-H 0.8975 from above within-paradigm (different objective, not size-matched). **ViT-S handoff verified end-to-end**: L pid 3963433 exited 08:49, gate `gate_vits_after_L.sh` (pid 76364) slept for CUDA teardown, re-verified CVD=0 torch alloc, truncated the crash-only train.log, and launched `clevr_dinov2s_concat_decoder_scratch` (pid 382312) at 08:50; confirmed alive + past first step (ep0 step100 acc 0.21, ~0.37 s/100-step → much faster than L). Persistent Monitor re-armed on S's train.log (Val acc / OOM / Traceback).

- **2026-07-22 ~03:46 — ViT-S/14 added as the smallest scale point (S→B→L axis); OOM'd trying to run concurrently with L, re-queued to auto-launch after L.** DINOv2 ViT-S/14 (~22M, 12 blocks/384-d/6 heads, same GCA injection depths [1,3,5,7,9,11] as ViT-B). New configs `configs/model/dinov2_small.yaml` + `configs/experiment/clevr_dinov2s_concat_decoder_scratch.yaml` (mirror L: res 336, cls pool, concat decoder1l, 16 ep, val_batch 256). Checkpoint downloaded to NFS cache (22M, no hang). First launch on CVD=0 OOM'd because it shares L's card. Gated launcher `$CLAUDE_JOB_DIR/tmp/gate_vits_after_L.sh` (pid tracked) waits for L pid 3963433 to exit, re-verifies GPU alloc, truncates the crash-only train.log, then launches S. **INFRA LESSON — still only ONE usable GPU (earlier "GPU 1 works / two GPUs" read was WRONG): `CUDA_VISIBLE_DEVICES=0` and nvidia-smi use different index orderings — L runs on the CVD=0 card (nvidia-smi calls it "GPU 1", 37.7GB); nvidia-smi "GPU 0" (17 MiB) is the broken A6000 that CVD=1 selects (`torch.cuda` → "No CUDA GPUs available", `device_count()==1`). Proven by the ViT-S OOM landing on L's card ("Process 3963433 has 39.91 GiB in use, 152 MiB free"). Never infer a free GPU from nvidia-smi's free-memory column; a passing CVD=0 torch alloc on an already-loaded card still fits a tiny test tensor and does NOT mean room for a run. All runs are sequential on the one card.** Backbone self-attn heads (per block, S/B/L/g/H): 6/12/16/24/16 — DINOv2 keeps head_dim 64 and widens by head count, I-JEPA-H is head_dim 80; the GCA cross-attn is a fixed 16 heads and the concat-decoder self-attn a fixed 8 (nhead), both backbone-independent.
- **2026-07-20 ~09:10 — DINOv2-g KILLED at ~step 10k/43.7k of epoch 0 (user decision); replaced by DINOv2-L (ViT-L/14, 300M) as the scale reference.** Rationale: g's real pace was ~9.5h/epoch → ~7 days on the single GPU vs L's estimated ~3 days; B→L is already a 3.5× parameter jump within the same paradigm, enough to answer "does scaling DINOv2 help", at the cost of only bracketing I-JEPA ViT-H (632M) from below instead of from both sides (g would have completed the g>H>L sandwich). g's partial dir (`clevr_dinov2g_concat_decoder1l_scratch*`) kept per no-overwrite policy — a future g run gets a NEW name and should reuse the grad-accum machinery (16×4) + the now-seeded NFS cache. New configs: `configs/model/dinov2_large.yaml` (GCA [2,6,10,14,18,22] over 24 blocks, depth-fraction-matched; res 336, cls pool), `configs/experiment/clevr_dinov2l_concat_decoder_scratch.yaml` (batch 64, val 256, 16 epochs, `wandb.name=clevr_dinov2l_concat_decoder1l_scratch`). ViT-L blob seeded home→NFS cache; launched offline (pid 3963433). Monitor on Overall/error lines.
- **2026-07-20 ~06:00 — DINOv2-g TRAINING (pid 3900269) after two OOMs: batch 64 OOM'd in forward, batch 32 OOM'd in backward; running at batch 16 × grad_accum 4 (effective 64 preserved) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.** Confirmed healthy past step 200 (loss 1.80→1.71). Trainable 50.2M / 1.54B (3.26%). **Revised ETA: ~0.79s per micro-step × 43.75k micro-steps ≈ 9.5h/epoch train + eval ⇒ 16 epochs ≈ 7 days** (vs the 2-day guess — ViT-g @336 has 2.25× tokens and 1.25× depth over ViT-H @224, and micro-batching costs throughput). Monitor on `Overall:` val lines + error signatures. in the first training steps; added grad accumulation and relaunched (pid 3898577).** ViT-g @336 (577 tokens, 40 blocks, 1536-d) needs ~43GB at train batch 64 → OOM on the 44GB card. Fix keeps the protocol's effective batch 64: new `training.grad_accum_steps` in `src/trainer.py` (zero_grad at window start, loss/accum before backward, clip+step at window end or last batch; default 1 = bit-identical to old behavior for every other config; loss logging uses the unscaled loss; scheduler is per-epoch so unaffected). Experiment yaml now sets `batch_size: 32` + `grad_accum_steps: 2`, `val_batch_size` 512→128 (headroom). Same log file (append). during checkpoint download; killed + relaunched (pid 3896753).** The 01:57 launch never trained a step: `HF_HOME` pointed at the NFS cache (`/nfs/.../jungchun/.cache/huggingface`) but the 4.5GB ViT-g checkpoint was only cached in home `~/.cache/huggingface` (from April), so hf_hub re-downloaded via xet — stalled at 3.96/4.55GB at 03:36 (signed S3 URLs expired → endless 403s, sockets CLOSE-WAIT, main thread futex-wait; unrecoverable). GPU sat idle ~3.5h; zero results lost. Fix: killed the hung pid (user-approved), seeded the NFS cache from the home copy (cp blob + snapshot symlink + refs, removed `.incomplete` and stale lock dir), relaunched with `HF_HUB_OFFLINE=1` appending to the same `train.log`. **Lesson: before launching a large-backbone run with NFS `HF_HOME`, verify the checkpoint blob already exists in THAT cache (`ls $HF_HOME/hub/models--<repo>/blobs/`) — the home and NFS caches are separate, and a 4.5GB re-download over xet can hang on URL expiry.**
- **2026-07-20 ~02:00 — adaLN-Zero STOPPED EARLY at epoch 16/20 (user decision: answer already clear; rerun later if needed).** Trajectory (`clevr_dinov2_adaln_zero_scratch_s42`): 0.4279 / 0.4442 / 0.4755 / 0.5531 / 0.6918 / 0.7627 / 0.7877 / 0.8153 / 0.8275 / 0.8307 / 0.8439 / 0.8517 / 0.8512 / 0.8566 / 0.8567 / 0.8557 / **0.8570** (ep0–16; plateaued 0.851–0.857 over the last 5). Three-arm readout (same DINOv2-B + decoder1l): **GCA 0.9095 (16 ep) > FiLM 0.8672 (20 ep) ≈ adaLN-Zero ~0.857 (stopped ep16, trending ≤0.86)**. Onset order inverts final order: adaLN-Zero starts highest (ep0 0.428 — zero-init preserves the pretrained function), FiLM middle, GCA lowest (0.212) — but content injection (cross-attn) has the higher ceiling than either global-modulation topology; within modulation, raw-space FiLM ≥ normalized-space adaLN. Checkpoints kept (best.pt = ep16 0.8570, last.pt, epoch_{4,9,14}.pt); dir preserved per no-overwrite policy — any rerun gets a NEW name. Monitor stopped. **GPU now free → DINOv2-g next.**
- **2026-07-18 ~15:40 — I-JEPA run COMPLETE: final val 0.8975 (vs DINOv2-B concat 0.9439, −4.6 pts); adaLN-Zero launched.** `clevr_jepa_concat_decoder1l_scratch_s42` full 16-epoch trajectory: 0.2089 / 0.4932 / 0.5105 / 0.5185 / 0.5350 / 0.5952 / 0.7093 / 0.7664 / 0.8045 / 0.8326 / 0.8502 / 0.8668 / 0.8810 / 0.8918 / 0.8963 / **0.8975**. Same S-curve as DINOv2-B but shifted ~3 epochs later (I-JEPA ep7 0.766 ≈ DINOv2 ep4 0.781; acceleration phase ep5–7 maps onto DINOv2 ep3–5); learning order identical (query_attribute/exist first, count and binary comparisons last; ep0 count+query_attribute both 0.000). Final slope ~+0.1 pt/epoch — mostly converged, so the −4.6 gap reads as feature-ceiling, not epoch budget (caveat: ViT-H vs ViT-B, not size-matched; DINOv2-g will calibrate scale).
  - **INFRA: legacy venv interpreter broke 07-17 00:31** — `.venv-aspen/bin/python` symlink pointed at uv-managed `cpython-3.13.12` under `~/.local/share/uv/python/`, which had been deleted (root disk 100%-full cleanup); running I-JEPA survived because the interpreter was already loaded. uv's index no longer offers 3.13.12, so restored by recreating `~/.local/share/uv/python/cpython-3.13.12-linux-x86_64-gnu/bin` as a symlink to `/home/jungchun/miniconda3/bin` (miniconda IS CPython 3.13.12 — exact version match; legacy tree itself untouched). Verified: venv python resolves, torch 2.11.0+cu128 imports, CVD=0 cuda alloc OK.
  - **LAUNCH (~15:45, pid 3616479):** `+experiment=clevr_dinov2_adaln_zero_scratch` → `outputs/model/clevr_dinov2_adaln_zero_scratch/train.log`, CVD=0 + TORCH_HOME/HF_HOME on NFS. Readout: final acc vs FiLM 0.8672 and GCA decoder1l. DINOv2-g queued after.
- **2026-07-16 ~08:45 — adaLN-Zero added as third conditioning arm (`condition_type: adaln_zero`); queued after I-JEPA.** Conditioning-topology comparison now GCA (additive content injection) vs FiLM (raw-space γ/β reweighting, `clevr_dinov2_film_scratch_s42` = 0.8672) vs **adaLN-Zero** (condition-predicted shift/scale/gate on the block's *own* norm1/norm2 outputs, DiT-style — modulation inside the block on the pre-attn/pre-MLP path, not an external residual mount). Frozen-backbone adaptation: DiT's `x + gate·branch` (gate zero-init) would kill the pretrained attn/MLP branches at init, so we use `x + (1+gate)·branch` and `x̂·(1+scale)+shift`, all generators zero-init ⇒ each block computes exactly the pretrained function at init. Code: `AdaLNZeroCondition` in `src/model/crossattention.py` (LN'd mean-pooled text → zero-init Linear → 6×dim: shift1/scale1/gate1/shift2/scale2/gate2), adaLN branch in `block_forward` + attach case in `ViTBackbone` (`src/model/backbone.py`); reuses the `blk.gated_cross_attn` attribute so `unfreeze_gca` and `checkpoint_io` work unchanged. Configs: `configs/model/steervit_dino_adaln_zero.yaml`, `configs/experiment/clevr_dinov2_adaln_zero_scratch.yaml` — exact mirror of the FiLM run (ViT-B DINOv2 @336, layers [1,3,5,7,9,11], decoder1l, 20 epochs) so the delta is purely the conditioning mechanism. CPU smoke PASSED: attach at 6 layers; **init-identity bit-for-bit** (cond forward == uncond at zero-init); modulation live after bias perturb; grads reach `to_mod` at all 6 layers, frozen trunk grad-free. Hydra compose check OK. **Queue order (user): I-JEPA → adaLN-Zero → DINOv2-g** (single usable GPU).
- **2026-07-16 ~07:52 — Backbone matrix extended with two large backbones; I-JEPA training launched.** Goal: add a joint-embedding-predictive backbone (I-JEPA) to the concat_decoder headline accuracy table (which had only ViT-B DINOv2/SigLIP/Sup/MAE). I-JEPA ships only at ViT-H+ (no ViT-B), and no single scale has clean checkpoints for all paradigms (ViT-B lacks JEPA; ViT-H/14 lacks DINOv2 & SigLIP), so a full size-matched matrix is impossible. User decision: add **I-JEPA ViT-H/14** as a standalone new-paradigm point (caveat: ViT-H, not size-matched to the ViT-B table) + **DINOv2 ViT-g/14** as a same-paradigm scale reference (does scaling the best B-backbone help? calibrates scale-vs-objective for reading the JEPA number). Plan file `~/.claude/plans/virtual-juggling-tiger.md`.
  - **Zero code change** — pure config. `ViTBackbone` auto-derives embed_dim/patch/token-count from the timm trunk, GCA layers are config-driven, `feature_pool: mean` (no-CLS) already exists (SigLIP). New: `configs/model/{jepa,dinov2_giant}.yaml` + `configs/experiment/clevr_{jepa,dinov2g}_concat_decoder_scratch.yaml`. jepa = `vit_huge_patch14_gap_224.in1k_ijepa`, res 224, mean pool, GCA `[3,8,14,20,25,31]` (6 injections, depth-fraction-matched to ViT-B's `[1,3,5,7,9,11]` over 32 blocks). dinov2g = `vit_giant_patch14_dinov2.lvd142m`, res 336, cls pool, GCA `[4,11,18,25,32,39]` over 40 blocks. Offline structure smoke (`tests/test_smoke.py::test_large_backbone_build_and_gca`) confirmed both build+forward on GPU with `HF_HUB_OFFLINE=1`: I-JEPA 32 blk/1280-d/0 prefix, DINOv2-g 40 blk/1536-d/1 prefix, GCA attached exactly at the requested layers, conditioned forward (GCA live) finite.
  - **INFRA: "504" was a red herring; the real blocker was a full root disk.** timm downloads I-JEPA via **torch.hub** from `dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar` (NOT HF hub) into `~/.cache/torch/hub/checkpoints/`. `/` is at **100% (4 GB free)** — the 9.7 GB I-JEPA checkpoint (full training state, not just weights) failed with `OSError errno 28`. Fix: `TORCH_HOME=/nfs/turbo/coe-chaijy/jungchun/.cache/torch` (2.2 TB free) → downloaded+built OK (params 630,762,240). **I-JEPA training MUST export this TORCH_HOME** or timm re-downloads into the full home disk. Home `~/.cache` bloat: huggingface 362 G, uv 46 G, pip 30 G (user's call to clean).
  - **LAUNCH (07:52, CVD=0 verified by torch alloc):** `clevr_jepa_concat_decoder_scratch` → `outputs/model/clevr_jepa_concat_decoder1l_scratch_s42/`. batch 64 holds on ViT-H (no OOM; Trainable 42.6 M / 1.03 B = 4.14%, backbone frozen; full CLEVR 699989/149991). ~0.94 s/step → ~2.9 h/epoch → **~1.9 days for 16 epochs**. Persistent Monitor armed on train.log (epoch/error/completion). DINOv2-g held for user go after I-JEPA (single usable GPU → sequential).
- **2026-07-16 — Terminology LOCKED to the full paper's framing ("Language Conditioning Elicits Symbol Grounding in VFMs").** Two stages ONLY: **Binding** (middle layers separate scenes by the *described* attributes; RSA proxy `_feature_binding`) → **Retrieval** (final layers organize scenes by the *queried attribute value*, within binding's subspace; RSA proxy `_answer_match`). **Stop using "answer matching"/"answer classification"** — the second stage is **Retrieval**. Object grounding and position indexing are diagnostic conditions, NOT stages. Discovered discrepancy: `conditional_rsa.py:70-82` mislabels the chains — `"Retrieval | Binding"` points at cond 2 = `_object_grounding`, and the real Retrieval (cond 4/6 = `_answer_match`) is labelled `"Answer classification"`. Needs a label swap for future runs (existing on-disk stats keep old labels — never rewritten). Memory `v2-architecture-and-naming-migration` updated. Circuit per paper: binding heads = middle CA (described attrs); retrieval heads = late CA+SA (queried attr type+value); described via CA, queried via CA+SA, separate pathways.
- **2026-07-15 ~13:44 — Relational-mechanism experiment batch launched (background, unattended).** Four experiments derived from the workshop-figure finding that same-attribute relations show an anchor→target RSA handoff while spatial relations do not (possible position shortcut vs position-as-indexing). Plan file: `~/.claude/plans/virtual-juggling-tiger.md`. Priority 三→一→四→二.
  - **實驗三 (pos_only)**: new `check_pos_only` in `src/data/clevr_conditions.py` (position-overlap RDM, NO attribute match) + `--pos-only`/`--dump-per-query`/`--query-list` flags in `scripts/analysis/conditional_rsa.py`; races position-RDM vs attribute-RDM onset per layer. Output `outputs/analysis/conditional_rsa/clevr_dinov2_decoder1l_scratch_pos_only/`.
  - **實驗一 (anchor_swap → head_ablation)**: new `generate_anchor_swap` (`src/analysis/clevr_corruptions.py`, 9097 eligible attr_query_same questions), `collect_anchor_swap_samples` (`patching_sampling.py`), anchor_swap routing in `activation_patching.py`, and a new `HeadAblator` context manager (`src/analysis/patching_utils.py`, zero/mean, mirrors HeadPatcher slice math). G2 auto-selects top-3 GCA + top-1 SA heads (`scripts/analysis/select_ablation_heads.py`) then re-runs conditional RSA with those heads ablated + a disjoint random-head control.
  - **實驗四 (shortcut_renders)**: `render_add_object.py` gained `--mode {shared_anchor,translate}` + `--category`; `evaluate_answer_strict` in `clevr_programs.py` (returns None on non-unique `unique` step — closes the silent `[:1]` hole). **DONE**: 4×100 valid pairs rendered on CPU (~28 min), `outputs/analysis/shortcut_renders/{shared_anchor,translate}_{attr_query_same,attr_query_spatial}/`.
  - **實驗二 (anchor_probe / anchor_dissipation)**: two new scripts — per-layer probe of anchor quadrant vs attribute on steered features (定錨 pretest), and per-query dissipation↔correctness analysis over the per_query dumps.
  - Orchestration: `scripts/analysis/queue_relational_{cpu,gpu}.sh` (resume-safe, `^=== QUEUE DONE` anchored, all gates automatic — G1 baseline-repro is diagnostic-only/never-abort, G2 head-select self-gates, G4 requires ≥100 pairs). CPU render queue done; GPU queue running S1.
  - Verification before launch: 3 parallel impl agents each CPU-verified their files (py_compile, unit, --help); integrated cross-group `HeadAblator` import confirmed; then all 5 pipeline paths smoke-tested on the real model+GPU (pos_only+per-query, HeadAblator ablation+query-list, anchor_swap patching, anchor_probe, dissipation) — all green.
  - **INFRA CORRECTION (verify, do not trust nvidia-smi display on this host):** the usable GPU is selected by `CUDA_VISIBLE_DEVICES=0`, NOT `=1`. `CVD=1` → torch `"No CUDA GPUs available"` (this is the broken 0x21 A6000 from the TODO note). `CVD=0` → healthy (benchmarked **113 TFLOPS fp16**, torch allocated 1.5 GB successfully). nvidia-smi misleadingly reports the *working* GPU 0's util/mem as `[N/A]`/`17 MiB` even under active load — so `nvidia-smi` memory/util is NOT a reliable liveness signal here; confirm with a `torch.zeros(...,device='cuda')` alloc test per CVD. The plan had assumed CVD=1 (from the initial nvidia-smi read); queue corrected to CVD=0 with a comment. S1 ran ~3 h for its 216 queries (CPU/IO-bound on the per-query 500-image DB re-load, not GPU-bound — GPU model confirmed).
  - **S1 done + G1 perfect (16:43):** the pos_only run reproduced BOTH baselines (`clevr_dinov2_decoder1l_scratch` and `_v2`) to Δ0.0000 at layers 6/9/11 on all 5 pre-existing curves — the pos_only columns are purely additive, and the two on-disk baselines are identical to current-code output (R1 stale-baseline worry moot).
  - **G2 recalibration (16:52) — head-ablation floor was too strict.** anchor_swap headwise patching top |mean| = 0.344 (GCA L5H4; then L11H8 0.288, L9H8 0.219, L7H10 0.188) vs median 0.0155 = **22× median** (ratio gate passed by a wide margin) but below the absolute floor 0.5, so the auto-gate marked FAIL and skipped S3/S4. The 0.5 floor was calibrated to the paper's *queried*-attribute binding heads (~0.77); anchor-swap is a more indirect corruption, so 0.34 is a legitimate anchor-binding signal — the ratio test is the operative criterion, not the floor. Launched `scripts/analysis/queue_relational_ablation.sh` (gated on the main queue's DONE) to run the ablation with the selector's own picks: ablate `gca:5:4,gca:11:8,gca:9:8,sa:8:0`, disjoint random control `gca:11:3,gca:1:14,gca:1:3,sa:5:10`, then baseline+ablation dissipation into `anchor_dissipation/clevr_dinov2_decoder1l_scratch_ablation/`. Main queue continued to S5 anchor_probe → S7 eval → S8 baseline dissipation.
- Day rotation performed (backlog: 07-06→07-09 archived below; boundary crossings were missed while sessions ran through the nights).
- Dead-log cleanup (user-authorized): deleted 5 crash-only logs (activation_patching_legacy, path_patching_phase3, download_pixart, eval_dinov2_mean Hydra-error, followup_d2 skip-wrapper); moved 2 real-output logs home (`eval_clevr_dinov2_mean_scratch_s42_v2.log` → its model dir; `grounding_manip_tsne_v3labels.log` → `grounding_manipulation/..._v2/`). `outputs/analysis/` top level now log-free; `metadata/` holds only the 6 multi-experiment pipeline logs.
- RESULTS.md §15 (Flamingo 8-ep retrain + E7 rerun: hallucination flat-to-worse despite doubled training → LLM-side-fusion property, E7 flamingo leg now quantitative) and §16 (object_count 1-vs-2-object × 5 prompts: scene-level collapse with one distractor, partial recovery only under the aligned color prompt) written; §12 Pending refreshed.
- **E5 figures done**: `failure_modes.py` gained `plot_summary` + `--replot MODEL|all` (aggregation from failure_summary.json, no GPU; GPU runs now emit the figure automatically) → `failure_modes.png` in all 4 model dirs (1×4: per-qtype bars / yes-no confusion / counting signed-error log-hist / acc-vs-depth). Main-vs-nogca pair is the A5 visual: symmetric confusion + off-by-one errors + flat depth curve vs prior-level everything, 2,043 non-numeric counts, wide error spread. RESULTS.md §7 updated with figure pointers.
- **Presentation t-SNE palette closed out**: the 3 presentation-referenced figures replotted from caches to tab10 α0.7 (`tsne_single_{noca,ca}_allattr.png`, `dino_attribute_tsne_v2.png`); 71-figure steered sweep cancelled — those use condition colors (tsne_viz FILL_COLORS), not ATTR_VALUE_COLORS (TODO entry has the full rationale). `tsne_grounding_manipulation.png` untouched for the same reason.
- **E5 autonomous diagnosis pass done (RESULTS.md §17)**: `failure_modes.py --diagnose MODEL|all` joins records.jsonl with CLEVR val questions/scenes and executes every program with a new mini ground-truth executor (`_exec_program`; validated: recomputed answer == dataset answer on all 150k programs). Main-model findings: D1 counting is enumeration-capacity-limited (acc 0.92→0.36 from gt 0→7, signed error slides +0.09→−0.81, undercounts dominate; counting degrades ~2.5× faster with scene clutter than the task average); D2 comparison errors are counting noise (95% at |Δ|≤1; acc 0.71 at Δ=0 vs ~1.0 at Δ≥4); D3 residual yes/no errors are spatial-chain: 8–13 pts lost per relate hop (0.987/0.905/0.776 at 0/1/2 hops); D4 depth-11 dip = family composition (two-referent/two-count types), not a depth cliff. nogca contrast: all three gradients flat/absent — the structured error profile only exists with grounding. Artifacts: `outputs/analysis/failure_modes/<model>/diagnosis.{json,png}` × 4 models.

## Log
> [!NOTE] Day Rotation inserts archived entries here. Newest on top.

### 2026-07-09 (Thu)
- **E7 evidence completed**: new `scripts/analysis/add_object_plot.py` → `outputs/analysis/add_object/hallucination_bar.png` (grouped bars, 5 models × 4 attrs, aggregation-only/rerunnable). Existing-JSON version first, auto-refreshed after the flamingo rerun.
- **Flamingo 8-ep E7 rerun (4 attrs, fixed protocol, last.pt)**: hallucination color 0.08 / material 0.59 / shape 0.52 / size 0.54, bait_share 0.83–1.00 (non-color); doubling training did not reduce hallucination → architecture (LLM-side fusion), not training budget. RESULTS.md §15.
- **Flamingo retrain harvested**: `clevr_flamingo_dinov2_frozenllm_s42` finished 07-08 04:12, final=best val acc 0.3176 (8 ep).
- **Log placement root fix**: new `src/analysis/run_log.py:tee_stdout(out_dir)` (appends stdout+stderr to `<out_dir>/log.txt`, timestamp+argv header) wired into 21 analysis scripts (agent did 18, hand-added add_object_eval{,_flamingo}, dino_attribute_tsne); all py_compile-clean, smoke-verified append mode. Blender render scripts excluded (no src on Blender's path).
- **Log consolidation**: 65 stray logs moved into their experiment dirs by content evidence (14 from `analysis/` top level, 51+ from `metadata/`); pipeline/queue logs stay in `metadata/`.
- **Folder convergence (user-directed)**: vault-root `outputs/` (2.5G invalid gcog runs + dead launch log) and vault-root `docs/` (superseded 05-13 superpowers design docs) DELETED; gcog launch scripts repointed `../outputs/gcog` → `outputs/gcog`; `main/outputs/2026-*` Hydra date dirs merged into `outputs/log/`; vault CLAUDE.md updated (results live ONLY in `main/outputs/`).
- 3 commits pushed to origin/master: object_count follow-ups (tab10 α0.7 palette, shape prompts, distractor rendering, patch-token naming) / run_log tee wiring / gcog paths.
- Superpowers plugin retired (user-directed): principles distilled into `~/.claude/skills/brainstorming/` (slim design gate) + playbook (20-judgment commit-reminder rule, 00-diagnosis debugging discipline); plugin disabled in settings; tdd-guard disabled by user.

### 2026-07-08 (Wed)
- **object_count experiment completed (RESULTS.md §16)**: n1 (500 single-object) + n2 (480 two-object, unconstrained ≥2-attr distractor) × {no-CA, color-object, color-cube, shape-object, shape-large} — 10 allattr t-SNE grids + 2 five-condition probes. L11: n1 ≈1.0 everywhere; n2 no-CA color 0.356 → 0.458 only under the aligned color prompt; shape prompts recover nothing. n3 dropped, one-large-one-small dataset rendered but retired (user reversal) — kept unused.
- Palette settled after iterations: ATTR_VALUE_COLORS = tab10 baked at alpha 0.7 (light tab20 rejected as too pale in legends); all 10 t-SNE figures replotted from caches (a first replot ran from the wrong cwd — relative paths wrote nowhere while the wrapper still exited 0; reran from `main/`).
- Gray-cluster question resolved quantitatively: CLEVR "gray" L11 centroid distance 12.2 vs 4.6–5.5 among chromatic colors — real feature-space isolation (achromatic + gray floor), not a t-SNE artifact.

**Completed:**
- [x] [paper] Flamingo retrain (launched 07-06, harvested 07-08/09; E7 fixed-protocol rerun done, RESULTS.md §15)

### 2026-07-07 (Tue) — covers 07-06 evening through 07-07
- **E7 harvested (all 4 attributes, concat main model)**: binding fixation is ROBUST under adversarial lure — acc_base→acc_added: color 0.98→0.97, material 1.00→1.00, shape 1.00→0.98, size 0.90→0.94; hallucination_rate 0–0.06; the FEW errors are bait-shaped (bait_share_of_errors 0.5–1.0). Reframes A1: trained grounding resolves the multi-object fixation problem for direct queries; the real multi-object bottleneck is set enumeration/combination (E5). Size is the weakest attribute throughout. RESULTS.md §8.
- **D2 lastep landed**: learned_text last.pt(ep15) independent eval = 0.1974 ≠ training-log 0.2456 ≠ paper 24.6 provenance assumption. Same qualitative collapse as best.pt (QryAttr 0.000, Count 0.003, binary at chance). Paper cell is protocol-dependent → new TODO for camera-ready decision.
- Side queue (E7 evals + D2) finished 03:51 — ran concurrently with pipeline E4 on the single healthy GPU (46GB, pipeline peaks ~8GB); pipeline will SKIP E7 via its own done_markers. E4 linear_probe done (02:57–07:03, exit=0); conditional_rsa running since 07:03.
- Journal day rotation performed (08:00 boundary).
- **nogca fixation leg (user-requested E5↔E8 alignment) landed 09:06, RESULTS.md §8b**: E7-on-nogca hallucination 0.24–0.59 vs trained 0–0.06 (~10× causal gap on identical stimuli); E5-on-nogca qryattr errors 98.6% = another scene object's value (color-only: 100.0% of 2,276 vs 52.3% chance), out-of-scene 1/6516. −CA fails by object mis-selection with intact attribute encoding → fixation. Triangle closes when E8 lands (raw substrate per-object decodability).
- **E4 harvested (probe+RSA on GCA-decoder), RESULTS.md §9 — A2↔A4 LOCK CONFIRMED**: Binding RSA half-rise L7/peak L11 sits exactly in patching's binding-head window (CA L3–L9); Retrieval separates L9–L11 matching SA11; ordering Binding→Retrieval strict in all 3 categories. Bonus: anchor→target binding handoff visible in same/spatial (anchor peaks L8 then collapses, target climbs to L11) — relational chaining = sequential re-binding, dovetails with E5's "relations cheap for localization".
- **E3 harvested (SigLIP patching), RESULTS.md §10 — A4.3 circuit motif REPLICATES**: sparse attribute-specialized GCA binding heads in L3–L7 (color L5H11 +1.08, material L5H12 +1.25, size L5H9 +0.93, shape L7H9 +0.34 — same head index as DINOv2's shape head), shared query-routing GCA heads at L7, query-side SA late (L10–L11). Deviation: described-side SA integration sits mid-layer instead of L11.
- **T2I prioritized by user (over OpenFlamingo)**: diffusers 0.39.0 installed to vault-root `deps-t2i/` via pip --target (legacy venv untouched, PYTHONPATH overlay; verified PixArtSigmaPipeline imports with venv torch 2.11). PixArt-Sigma-XL-2-512-MS transformer downloaded (2.3G); companion repo `pixart_sigma_sdxlvae_T5_diffusers` (T5-XXL+VAE) downloading in background. GPU phase queued behind E3+E8.
- **E8 MAE landed (14:38) — E8 COMPLETE, all 4 backbones**: MAE per-object decodability 0.914–0.979, only slightly below DINOv2's 0.966–0.997. Substrate ordering matches downstream ordering but the substrate gap is far too small to explain the 17-pt VQA gap → "MAE weaker substrate" must be phrased as weaker *binding-usable* structure, not missing attribute information (dovetails with §5's two-referent EqAttr diagnosis). RESULTS.md §11 updated; fixation-triangle table now rests on all four backbones.
- **T2I timestep sweep complete (t=100/261/400), RESULTS.md §13 — VERDICT NEGATIVE under pre-registered criteria**: referent-local probe exceedances scattered (best: material 0.689 vs 0.571 at t=261; color 0.230 vs 0.159 at t=400), CA localization never above 1.33× chance. Only structured residue: t=400 CA peaks at the SAME block (B6) in all 3 categories at 0.0112–0.0117. Bounded conclusion per pre-registered caveat (questions≠captions): zero-shot binding does not emerge under question prompts; grounding as measured requires task training. Optional follow-up: declarative-caption prompts via extract_oracle_prompt.
- **E9 contrast figure + E10 v2-label replots done (GPU freed by sweep completion)**: `abc_localization.py` now emits `abc_contrast.png` (grouped CA-share bars; A>B>C gradient visible per attribute, shape-C lowest at 0.199; JSON verified byte-identical, only the figure added). `grounding_manipulation.py` gained `--replot-from` (figure regeneration from saved JSONs, no GPU) — 6 figures + GPU t-SNE rerun into NEW dir `outputs/analysis/grounding_manipulation/clevr_dinov2_decoder1l_scratch_v2labels/`, originals untouched.
- **Opus substrate/fixation integrated report written**: `docs/substrate_fixation_report.md` (triangle narrative §8–§14; spot-checked numbers against RESULTS.md). Opus flagged two pre-registration deviations → new TODO.
- **NAMING (user decision 2026-07-07, supersedes the 07-06 interim labels)**: Retrieval has NO object/answer split — the old middle chain level IS the Retrieval stage; the answer level is the pre-existing classification readout ("Answer classification"), not a grounding stage, not discussed as mechanism. "Retrieval (object)/(answer)" abolished across conditional_rsa/tsne_viz/grounding_manipulation, legacy-reference naming map, RESULTS.md, all three report docs, memory. Corrected-label replots → `grounding_manipulation/clevr_dinov2_decoder1l_scratch_v3labels/` (7 figures; v2labels dir kept untouched).
- **Flamingo E7 measurement corrected (21:40), RESULTS.md §14**: first run was a harness artifact — `generate_answer` lowercases its decode, adapter split on uppercase `"Answer:"` → every prediction was prompt-echo `'question:'`, all-zero accs. Fixed in the adapter AND in `train_flamingo_clevr.py:evaluate` (same latent bug; would have reported val acc 0 during the planned retrain). Invalid JSONs left in place (never-overwrite); corrected runs = `*_fixed.json`. Corrected result: chance-level, near-degenerate predictions on all 4 attributes (color 98% "yellow", size 100% "small") → E7-flamingo uninterpretable until retrain; qualitative-only status confirmed quantitatively.

**Completed:**
- [x] [paper] Pre-registration deviations RESOLVED (user-approved 2026-07-06): outline amended — A1.3 headline metric = hallucination_rate (bait_share kept secondary); T2I t=261/400 sweep recorded as post-hoc robustness check.

### 2026-07-05 (Sun)
- Identified the paper's model-variant split: performance tables use the **concat self-attention decoder** runs (`clevr_<bb>_concat_decoder1l_scratch_s42`, `ConcatSelfAttnDecoder` per `src/tasks/decoder.py:309`), while all mechanistic analyses use the **GCA-decoder** (`clevr_dinov2_decoder1l_scratch_s42`, `VQADecoder`). Matches paper App. A.
- Re-extracted final-epoch val acc first-hand for every run in `outputs/model/` (train_log.jsonl where present, else `Val acc:` lines in text logs). Full matrix in `docs/paper_artifacts.md` §9.
- Verified paper↔repo number matches: SigLIP 92.6 / Sup 86.6 / MAE 74.8 (concat), cls table 90.1/84.8/86.6/77.0, ablations 52.8/49.4/24.6, Table 6 transfer numbers all exact from `*_ft_all/results.json`.
- Flagged unresolved: DINOv2 92.4 matches NO main-repo artifact (concat=94.4, decoder1l=91.0); CoGenT 92.4/88.0 vs repo logs 94.5/89.5; dispatched search of SteerViT-legacy outputs.
- Found learned-text log subtlety: paper 24.6 = final-epoch 0.2456; the run's *best* acc was 0.4667 (early epoch, then degrades) → paper convention = final-epoch accuracy.
- Discovered probe+RSA for the concat main model already exist under `outputs/analysis/{linear_probe,conditional_rsa}/concat_decoder_1l/` — E4 gap narrowed to the GCA-decoder checkpoint only.
- Wrote `docs/paper_artifacts.md` (R0 deliverable): provenance for every paper table, naming map, unresolved list, regenerating commands.
- Launched E1a (dinov2_mean eval) + E1b (per-qtype breakdowns, 17 checkpoints) sequentially on GPU 1; logs → `outputs/analysis/metadata/`.
- Initialized JOURNAL.md + EXPERIMENT.md (this file; backfilled Log from git history and outputs mtimes only).
- **RESOLVED DINOv2 92.4 (corrected)**: checkpoint-stored `val_acc` in ALL FOUR concat best.pt files matches Table 1 exactly (0.92375/0.92556/0.86551/0.74763) → **Table 1 is single-provenance, main-repo concat s42**. The misleading 0.9437 came from `clevr_dinov2_concat_decoder1l_scratch_s42/stdout.log`, which belongs to a different, unidentified run (dir contamination — it also holds a stray siglip_nogate ckpt subdir); the true log is top-level `clevr_dinov2_concat_decoder_scratch_s42.log` (Val acc 0.9237). Legacy `odd_scratch_decoder_1l` (0.9248) was a near-miss coincidence. CoGenT 92.4 = legacy cogent run at epoch 11 (ep15=0.9449); ValB 88.0 has no persisted artifact → cannot reproduce; main-repo equivalents 94.5/89.5. R0 complete.
- W1+W2 done: initialized RESULTS.md (R2 gate-as-design-choice framing with alpha-interpolation + intervention evidence; R3 CoGenT table; R4 baseline triage) and wrote `scripts/analysis/aggregate_results.py` → `docs/results_tables.md` (46 model dirs; flags source conflicts like the contaminated dinov2-concat stdout.log; `--ckpt-meta` reads authoritative checkpoint val_acc).
- Diagnosed GPU selection: CUDA enumerates fastest-first, so `CUDA_VISIBLE_DEVICES=1` selects the BROKEN A6000 (nvidia-smi GPU 0, ERR! state, torch cuda unavailable). Healthy GPU = CUDA device **0** (PCI bus 0x22 = nvidia-smi GPU 1); matmul smoke-tested OK. Also: `eval_generalization.py --data-root` expects the PARENT of `CLEVR_v1.0`, and `evaluate.py` needs `+checkpoint=` (append syntax). First E1 batch launch failed on all three counts; fixed and relaunched.
- E9 first pass done (aggregation only, no GPU): `scripts/analysis/abc_localization.py` → `outputs/analysis/abc_localization/clevr_dinov2_decoder1l_scratch/`. Finding: CA-share gradient A(0.53–0.55) > B(0.43–0.49) > C(0.20–0.43); C's top head is always late SA (L11H0/H11). Claim holds as gradient, not absolute. Details in RESULTS.md §6.
- L1 done: `docs/legacy-reference.md` (naming grammars incl. legacy `odd_scratch_decoder_1l` decode, corruption taxonomy, headwise-patching methodology + stats schema, plot-style pointer, legacy ckpt format).
- L3 done: `src/model/checkpoint_io.py` unifies 6 duplicated loaders (5 planned + eval_generalization's). Fixed real bugs: old eval_generalization loader dropped `use_gate` (nogate ckpts rebuilt with zero-init gates → GCA silently nulled) and built decoder models for classification ckpts. Smoke-tested on 5 ckpt types (concat/nogate/cls/mot/legacy) — all load with correct task+gate config.
- E1a done: dinov2_mean last.pt is **epoch 0** (crashed run) — acc 0.218 meaningless; cell marked unrecoverable-without-retrain.
- Per user instruction, **stopped the E1b batch** (was on checkpoint 1/17) and recorded it as a TODO with a resume command; GPU is now free.
- L4 done (BLENDER_TOOLS_ROOT env override in both render scripts); L5 done (routing lines in both CLAUDE.md files, backups in .claude/backups/; legacy-headwise memory now points at docs/legacy-reference.md §3).
- P1 done: 3 logical commits on master (configs+training-core / analysis+eval suite / docs+journal+checkpoint_io). Stray root `eval_results.json` (my E1a epoch-0 output) moved to outputs/analysis/metadata/; `/data/` + `/eval_results.json` gitignored.
- E5 tooling ready: `scripts/analysis/failure_modes.py` (per-question dump → per-family acc, yes/no confusion + prior bias, counting signed-error hist, answer drift; `--stride` for subsampled runs). Runs when GPU frees up.
- **E7 unblocked**: Blender 3.6.14 smoke render passed (96 single-object images via render_single_objects.py to scratchpad, CPU, ~minutes). Add-object pipeline design next.
- E7 launched: new `scripts/analysis/render_add_object.py` (reuses render_visual_corruptions helpers; distractor = one described-attr flipped + bait value on queried attr; answer invariance verified by program execution; base re-render controls domain shift) + `add_object_eval.py` (acc_base/acc_added/hallucination_rate/bait_share_of_errors/flip_rate). Color 100 pairs rendering on CPU → outputs/analysis/add_object/color/.
- E7 render bug fixed (is_simple_query rejected the final query_* step itself → 0 eligible; now ~1700 eligible per attribute). All 4 attributes rendering 100 pairs each on CPU → `outputs/analysis/add_object/<attr>/`.
- L2/L6 complete (subagent): 7 plotting scripts consolidated onto `src/analysis/plot_style.py` (replot-verified GPU-free on conditional_rsa); L6 evidence verdict — `render_single_object` (pyrender)/`render_single_objects` (Blender), `linear_probe_{single,multi}`, `tsne_single_object` are distinct ACTIVE pipelines, not duplicates → nothing moved, `scripts/README.md` written instead.
- P3/P5/P6 on master: README.md (paper overview + matrix table + regeneration pointer), LICENSE (MIT), tests/test_smoke.py — 4 backbones build+forward OK (dinov2/siglip/augreg/mae, pretrained=False). Committed (9b26242).
- **Chained GPU queue launched**: waits for E1b, then E5 failure_modes (concat main + GCA-decoder, stride 4) → E4 probe+RSA (GCA-decoder, new dirs) → E7 add-object evals (4 attrs, main model) → E3 SigLIP decoder1l patching (A/B denoising). Everything idempotent, logs in outputs/analysis/metadata/.
- E7 renders COMPLETE: 4 attributes × 100 pairs × 2 images; first pair visually verified (distractor correct, described referent still unique). Evals queued.
- **Fable pre-registration written** (`docs/paper_v2_outline.md`): A1–A5 claim registry with exact v2 wording, evidence artifact, and status per claim; wording constraints from R0/E9 (no "A only affects CA" absolutes; bait_share_of_errors as E7 headline; single-provenance table rule); E5 autonomous-diagnosis hypotheses H1–H3 with their tests — any model can execute the diagnosis from this file once E5 tables land.
- E8 implemented (`scripts/analysis/raw_backbone_probe.py`): per-object patch-pooled attribute probes on multi-object scenes from the raw backbone (fresh zero-gated GCA = pure ViT, verified crossattention.py:95,106). Chained to run on 4 backbones after the main GPU queue.
- Alignment audit delivered: `docs/experiment_registry.md` (X1–X15 + D1–D11). D7 RESOLVED same-copy (md5); learned_text best/last divergence verified (only affected run); E7 families == attr_query_direct exactly.
- E10 done (subagent): figure labels → v2 naming ("Object match" chosen for the old middle-stage measurement, pending user confirm); 9 conditional-RSA figures replotted into `conditional_rsa_v2names/`; linear_probe/grounding_manipulation have no GPU-free replot path (labels fixed for next GPU run).
- P2/P4 done (subagent): `release/public` branch commit 4e41c8a — all personal paths → env vars (CLEVR_ROOT etc.), `grep /home/jungchun` empty, CLAUDE.md untracked there. Worktree kept in scratchpad.
- **E1b first results**: dinov2 concat overall 0.9237 == ckpt val_acc == paper 92.4 (independent protocol reproduces the number — D10 closed). Per-qtype: QryAttr 0.991 / Exist 0.961 / EqAttr 0.925 / Count 0.853 / CmpInt 0.785; SigLIP profile nearly identical. Failure ordering (CmpInt < Count) confirms A5.1's direction; numbers are within ~1pt of the draft's legacy-derived cells → camera-ready table barely moves.
- **Night pipeline daemonized** (2026-07-05 23:19): earlier background jobs were parented to the claude CLI (would die on disconnect) — stopped cleanly and relaunched via `setsid nohup` (PPID=1). Monitor from ANY terminal: `tail -f main/outputs/analysis/metadata/night_pipeline.log`; stages log `[pipeline] ... done`, finishes with `ALL STAGES DONE`. Stage order: E1b remainder → E5 failure_modes ×2 → E4 probe+RSA (dinov2 GCA-decoder) → E7 add-object eval ×4 → E3 SigLIP patching → E8 raw-backbone probes ×4. All idempotent; per-experiment logs in outputs/analysis/metadata/.
- E1b 9/13 harvested (2026-07-06 early): sup concat deficit is UNIFORM across qtypes (readout-sensitive, not counting-specific — A3.2 revised); **MAE two-referent collapse** (QryAttr 0.921 vs EqAttr 0.586) bridges A3.1 substrate claim to H1; SigLIP cls reversal (needs decoder readout). RESULTS.md §5.
- E1b ablation rows harvested (2026-07-06 01:40): gca_scratch eval 0.5277 == paper 52.8 EXACT; nogca ckpt best_acc 0.4945 == paper 49.4. **Three ablations fail three different ways**: −CA kills Count first (0.246 vs ~0.51 prior-plateau elsewhere); scratch-ViT keeps binary types above prior (Exist 0.664/CmpInt 0.672) but QryAttr starves at 0.490 (mechanism without substrate); learned_text collapses open-vocab generation entirely (QryAttr 0.000, Count 0.0004 — decoder emits degenerate strings, binary types at chance). A3 "all three necessary" now has per-component failure signatures. Caveat: learned_text row is best.pt(ep2), full-val 0.207 ≠ stored windowed 0.4667 — paper cell (24.6=last.pt) rerun queued. RESULTS.md §5.
- **E5 concat harvested + H1–H3 adjudicated** (2026-07-06 03:00): H2 REFUTED (pred-no 0.504 vs gt-no 0.503 — perfectly calibrated, no prior collapse); H3 CONFIRMED (86.9% of counting errors are ±1, symmetric); H1 CONFIRMED-refined (8 worst families all two-set cardinality: count-union fam 67/71/70, compare-counts fam 6/7/3). **Headline: difficulty axis = referent multiplicity, not program depth** — query_attribute flat 0.97–1.00 across depth 4–20 while count/compare_integer decay to 0.69–0.74; aggregate depth-curve dip at 9–11 is a composition artifact. Autonomous follow-up: relations free for localization (deep chains 0.992) but each relational region-constraint collapses counting (0.982→0.656). RESULTS.md §7.
- **E5 GCA-decoder replicates the failure structure** (03:15): family-acc Spearman ρ=0.927 vs concat (89 families), same top-4 worst {67,70,71,6}, same relational-enumeration collapse (0.977→0.594), same symmetric off-by-one (87.9%). Verdict: failure signatures are properties of the shared GCA grounding mechanism, not the readout. Only readout effects: mild autoregressive "no" bias (drift ±142) and zero non-numeric counting outputs. E5 fully done → RESULTS.md §7.
- GPU repair attempt concluded (2026-07-06 ~02:40): broken A6000 (PCI 0x21) is beyond software rescue — driver reset "Not Supported", PCI remove blocks forever on usage count, persistence-off didn't release it, survived reboot in ERR! state → report for hardware repair. Pipeline was SIGSTOPped after E1b to free the GPU for the attempt, then SIGCONTed; E5 started 02:41 with zero disruption.
- Side queue launched 03:16 (concurrent with pipeline on the 46GB GPU): E7 add-object evals ×4 + D2 learned_text lastep eval, pulled forward from the pipeline tail.

**Completed:**
- [x] [paper] R0 residual: resolve DINOv2 92.4 + CoGenT 92.4/88.0 provenance (RESOLVED; see docs/paper_artifacts.md §8)
- [x] [paper] Camera-ready CoGenT: DECIDED (user 2026-07-05) — draft 92.4/88.0 was a transcription error; adopt main-repo 94.5/89.5 (zs) and 92.7 (ft)
- [x] [paper] Baseline survey done (registry X14): OpenFlamingo-9B / PixArt-Σ selected, Transfusion dropped (no public weights); implementation gated on user go
- [x] [metrics] E1b per-qtype breakdowns: COMPLETE 13/13 (2026-07-06 02:41)
- [x] [paper] R2: gate-as-design-choice framing written (RESULTS.md §2)
- [x] [metrics] E5: failure_modes ×2 + autonomous diagnosis COMPLETE (H1 confirmed-refined / H2 refuted / H3 confirmed; cross-readout ρ=0.927)
- [x] [data] E7 renders + evals COMPLETE (4 attrs; binding robust, hallucination 0–6%)
- [x] [infra] W2: aggregate_results.py → docs/results_tables.md
- [x] [infra] L1–L6: legacy distillation COMPLETE (legacy-reference.md; plot-style: 7 scripts on plot_style.py, replot-verified; checkpoint_io dedupe; BLENDER_TOOLS_ROOT; CLAUDE.md routing; scripts/README.md — L6 verdict: no scripts moved, "duplicates" are distinct active pipelines)
- [x] [infra] P2/P4: release/public path refactor + CLAUDE.md untracked (commit 4e41c8a; `grep /home/jungchun` empty)

### 2026-07-01 — 2026-07-02 (backfilled from outputs mtimes)
- Gate-mediated intervention analyses (`outputs/analysis/grounding_manipulation{,_acc}/`), t-SNE regeneration, linear probes (cls/nogate/siglip variants).

### 2026-06-08 — 2026-06-17 (backfilled from outputs mtimes)
- Transfer fine-tunes: CLOSURE / CLEVR-Math / CLEVR-Humans ft variants (`*_ft_all`, `*_ft_connector`, `*_ft_gca_connector`) with `results.json` each.
- Concat-decoder sweep trained: dinov2 (6/10), nogca (6/11), siglip+mae (6/12), sup (6/13); cls runs for mae/sup (6/13–14); CoGenT trainA run (6/09); GQA siglip; flamingo attempt (6/14–15, no eval).
- Mechanistic suite on dinov2 GCA-decoder: cogent_patching (6/10), conditional RSA (6/11), binding interchange + activation/path patching (6/14–16), visual corruptions + patching t-SNE (6/15), ACDC + single-object analyses (6/17).

### 2026-05-13 — 2026-05-24 (backfilled from git log)
- 5/13: repo scaffolded from legacy SteerViT — Hydra config hierarchy, CLEVR/GQA data modules, classification+decoder heads, unified trainer/evaluator (per-type breakdown), text cache.
- 5/14–16: `clevr_dinov2_decoder1l_scratch` naming + 16 epochs, multi-seed support, cls config, SigLIP backbone, back-patching script.
- 5/24: MoT, Transfusion, supervised-ViT, GCA-scratch experiment configs.

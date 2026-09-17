# X27 colour-subspace replacement — implementation and engineering pilot

Author: Claude. Date: 2026-09-17. Reviewer: Codex.
Spec: `writing/ATTRIBUTE_GEOMETRY_INTERVENTION_SPEC_CODEX_2026-09-17.md` plus the four review
items of 2026-09-17 (role difference, cross-patch sampling of the random control, executable
manipulation rule, explicit H1 formula).
Status: stage 1 only — code, unit tests, 8-image engineering pilot. **No effect interpretation.**
The full run (stage 2) has not been launched.

## 1. Where the code lives

- Worktree `main/.claude/worktrees/patch-pca-cluster`, branch `worktree-patch-pca-cluster`;
  the pilot ran at commit 3511226 (manifest `git_head`), the code is committed on top of it.
- `scripts/analysis/patch_language_condition.py`, new block before the readout section:
  `colour_subspace_folds`, `replace_colour_coords`, `matched_rotation`, `_boot_family`,
  `_cr_obj_stats`, `run_colour_replace`, `summarise_colour_replace`, `plot_colour_replace`.
  CLI: `--colour-replace [--colour-replace-pilot N] [--colour-replace-doses 0,0.5,1]
  [--colour-replace-seeds 10]`; `--colour-replace-summarise <dir>` recomputes `summary.json`
  and the figure from `per_image.jsonl` without a model.
- `tests/test_colour_replace.py`: 12 tests (pytest-style; the project venv has no pytest, run
  with a plain function runner). All pass.
- Nothing existing was modified; no existing output directory was touched.

## 2. Site, donor, subspace (as implemented)

| Item | Implementation |
|---|---|
| Site | `patches = steervit.forward(images, question)[:, prefix:]`, the tensor `model.decoder(bos, patches)` receives. Edited in place before the decoder call; one decoder call per (image, question) over all variants. |
| Feature-space check | `trunk.norm(block-11 output) == patches` asserted on the first batch (atol 1e-4); recorded in the manifest (`True`). |
| Donor | `steervit.forward(images, None)[:, prefix:]`, same image, same patch positions, never edited. |
| Edited tokens | owner mask of the selected object only (`owner.npy` of n2); all other tokens are the recipient's exact values. |
| Colour subspace | n1 (one-object, no-question) object means at the decoder input; 5 folds by `pair_index % 5`; per fold, 7 colour class means (gray is excluded by the segmentation, `COLORS`), centred by their unweighted mean, SVD, singular values ≥ 1e-6 × max ⇒ rank 6 (the spec's "eight classes, rank ≤ 7" is 7 classes, rank 6 in these data). Saved in `subspace.npz`; per-fold fit / held-out pair IDs in the manifest. |
| Own-value directions for the manipulation check | `attribute_directions_crossfit` with the same 5 folds (block 11). |
| Edit | spec §6 exactly (norm preserved; outside-subspace direction preserved; colour coordinates of the normalised token = donor's at dose 1). Invalid tokens counted, never repaired. |
| Controls | sham (donor = recipient); matched rotation with independent tangent per token; matched rotation with one tangent source per object (review R2); matched rotation with tangent ⊥ colour span (spec §7.3). 10 seeds each, same per-token angle ⇒ same norm and same Euclidean edit length. |
| Doses | 0, 0.5, 1 (dose 0 kept as an identity row; rotations only at doses > 0). |
| Statistics | per-image quantities averaged over the two question directions, then family bootstrap (family = `pair_index`), 2000 replicates, seed 42; the referent dose-1 T0/T1 use 97.5 % intervals, all else 95 %. |
| H1 formula (R4) | T1 = mean_i [Δm_edit,i − mean_s Δm_rot,s,i] with Δm = margin − clean margin; support requires T1 upper bound < 0 **and** T0 = mean_i Δm_edit,i upper bound < 0. |
| Role difference (R1) | D_role,i = Δm_edit,i − mean_s Δm_rot,s,i; reported D_ref − D_nonref, paired within image. |
| Manipulation rule (R3) | ρ_i = (cos_edited − cos_clean)/(cos_donor − cos_clean) on images with |donor − clean| > 0.01; `manipulation_ok` = ρ lower 95 % bound > 0.5 and sign-consistent fraction ≥ 0.8; secondary analysis on the ρ ≥ 0.5 subset. |
| H3 | not run: n2 has only colour questions. |

## 3. Unit tests (all pass)

dose 0 = recipient · sham = recipient · norm preserved and outside-subspace direction preserved at doses 0.5 / 1 · colour coordinates of the normalised token equal the donor's at dose 1 · unedited tokens untouched · zero-norm tokens flagged invalid · rotations reproduce the per-token angle, norm and edit length for all three variants · targeted tangent ⊥ colour span and ⊥ u · object-shared rotation more coherent than independent · seeds reproducible · folds: rank ≤ 6, U orthonormal, fit ∩ held-out = ∅, held-out accuracy on separable synthetic data > 0.9 · family bootstrap and the summary's T0 / T1 / role-difference / ρ / decision fields recover the planted signs on synthetic rows.

## 4. Engineering pilot (DINOv2, first 8 images, all conditions)

Output `outputs/analysis/patch_language_condition/n2_colour_replace_pilot/` (`manifest.json`,
`per_image.jsonl` 2064 rows, `subspace.npz`, `summary.json`, `colour_replace.png`); log
`outputs/analysis/patch_language_condition/log_colour_replace_pilot.txt`. Wall time 3 s.

Checks (engineering only):

| Check | Result |
|---|---|
| feature-space assert | True |
| sham rows equal clean | max |Δmargin| 1.9e-6 (float32) |
| dose-0 edit rows equal clean | max |Δmargin| 1.9e-6 |
| invalid tokens | 0 in all four edit kinds |
| per-token edit length, edit vs all rotations | identical (14.95 referent, 18.04 non-referent, dose 1) |
| object-mean displacement, dose 1 (referent) | edit 14.5; independent rotation 6.0; object-shared rotation 14.9; targeted 6.0 |
| patch coherence (mean pairwise cosine of edit vectors) | edit 0.91; independent 0.03; shared 1.00 |
| object-mean distance to donor inside the colour subspace | edit 0.03; rotations 0.31 |
| held-out colour discrimination of the subspace (nearest projected class mean, 7 classes, chance 0.14) | 0.39–0.46 per fold (raw or unit-normalised means alike) |

The independent-tangent rotation cancels at the object level (displacement 0.4 × the edit),
exactly the concern raised in review item 2; the object-shared rotation matches the edit's
object-level displacement. Both are kept and reported side by side.

The per-object statistics in `summary.json` (ρ, T0, T1, role difference) are produced for the
8 images to exercise the code path only; they are not results.

## 5. Open before stage 2

1. Go for the full DINOv2 run (324 images) and the SigLIP run
   (`--out-dir outputs/analysis/patch_language_condition/siglip --checkpoint outputs/model/clevr_siglip_decoder1l_scratch_s42/best.pt`).
2. Registry section X27 (hypotheses, predictions, controls, decision table) to be committed
   before the first full run.
3. Held-out colour discrimination of about 0.4 on 7 classes: to be reported as is; whether it
   changes the reading of the manipulation check is Codex's call.

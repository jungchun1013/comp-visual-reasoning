# Paper → repo artifact provenance

Workshop paper: **"Language Elicits Compositional Reasoning in Pretrained VFMs"**
(accepted, ICML 2026 CompLearning workshop). Every paper number is traced here to the
repo artifact that produced it, with the regenerating command. Rows that cannot be
reproduced from any repo artifact are explicitly marked — never guessed.

All numbers below re-extracted first-hand on 2026-07-05 (session R0/R1). Log-format
note: `train_log.jsonl` rows carry per-epoch `val_acc`; plain-text logs print
per-epoch `Val acc: X` and a final `Done. Best val acc: X` line — the **paper uses
final-epoch accuracy** (matches everywhere it can be checked; see learned-text, where
best 0.4667 ≠ paper's 24.6 = final 0.2456).

## 1. Model-variant identification (paper term → repo name)

Verified against `configs/experiment/*.yaml` and `src/tasks/decoder.py:296-330`:

| Paper term | Repo experiment config | Decoder class | Notes |
|---|---|---|---|
| Main model ("one-layer concat self-attention decoder", App. A) | `clevr_<bb>_concat_decoder_scratch` → run dirs `clevr_<bb>_concat_decoder1l_scratch_s42` | `ConcatSelfAttnDecoder` (`task.decoder.type: concat_self_attention`, num_layers 1) | **Performance tables use this variant** |
| GCA-decoder variant (mechanistic-analysis model) | `clevr_<bb>_decoder1l_scratch` | `VQADecoder` (cross-attention decoder, num_layers 1) | **All patching / intervention / RSA-figure checkpoints** are `clevr_dinov2_decoder1l_scratch_s42` |
| Classifier readout | `clevr_<bb>_cls_scratch` | classification head | Table 4 |
| −CA ablation | `clevr_dinov2_concat_decoder_nogca_scratch` → dir `..._concat_decoder1l_nogca_scratch_s42` | ConcatSelfAttn, no GCA | |
| Scratch-ViT ablation | `clevr_dinov2_gca_scratch` (dir `clevr_dinov2_gca_scratch_s42`) | ViT trained from scratch + GCA | |
| Learned-text ablation | `clevr_dinov2_learned_text_decoder1l` | learned text embeddings replace RoBERTa | |
| (unreported) ungated CA | `clevr_<bb>_nogate_scratch` | GCA without tanh gate | design-choice ablation, kept out of paper claims |
| Backbones | `dinov2` / `siglip` / `sup` (augreg_in21k) / `mae` | | |

## 2. Backbone table (paper Table 1, decoder rows) — concat runs, seed 42

| Paper cell | Paper | Repo artifact | Repo value | Match |
|---|---|---|---|---|
| DINOv2 | 92.4 | `outputs/model/clevr_dinov2_concat_decoder1l_scratch_s42/best.pt` stored `val_acc` **0.92375** (ep15) = tail of top-level log `clevr_dinov2_concat_decoder_scratch_s42.log` (`Val acc: 0.9237`) | 0.9237 | ✓ (see §8.1 for the dir-contamination caveat) |
| SigLIP | 92.6 | `outputs/model/clevr_siglip_concat_decoder1l_scratch_s42.log` | final 0.9256 | ✓ |
| Sup-ViT | 86.6 | `outputs/model/clevr_sup_concat_decoder1l_scratch_s42.log` | final 0.8655 | ✓ |
| MAE | 74.8 | `outputs/model/clevr_mae_concat_decoder1l_scratch_s42.log` | final 0.7476 | ✓ |

Per-category columns (QryAttr / EqAttr / Exist / Count / CmpInt): **no main-repo
artifact** — `outputs/analysis/generalization/` holds only dinov2 image_only/text_only +
siglip decoder1l JSONs. (A legacy GCA-decoder breakdown exists at
`SteerViT-legacy/exp_vqa/outputs/phase1_clevr/odd_scratch_decoder_1l/eval_breakdown.json`
— QryAttr 0.9915, EqAttr 0.9211, Exist 0.9607, Count 0.8577, CmpInt 0.7900 — possibly
the draft's source for those cells, but it is a different variant/run.) Regenerate
single-provenance cells with E1b:
`PYTHONPATH=src $INTERP scripts/eval_generalization.py --checkpoint outputs/model/<name>/best.pt --data-root $CLEVR_ROOT --skip-closure --skip-cogent --skip-humans`

## 3. Ablation table — seed 42

| Paper cell | Paper | Repo artifact | Repo value | Match |
|---|---|---|---|---|
| Classifier readout | 90.1 | `clevr_dinov2_cls_scratch_s42/train_log.jsonl` (ep15) | 0.9014 | ✓ |
| Scratch ViT | 52.8 | `clevr_dinov2_gca_scratch_s42/train_log.jsonl` (ep15) | 0.5277 | ✓ |
| −CA | 49.4 | `clevr_dinov2_concat_decoder1l_nogca_scratch_s42.log` | 0.4945 | ✓ |
| Learned text | 24.6 | `clevr_dinov2_learned_text_decoder1l_s42_train.log` | final-epoch `Val acc: 0.2456` (best was 0.4667, early epoch — run degrades) | ✓ |

## 4. Classifier-readout backbone table (paper Table 4) — cls runs, seed 42

| Paper cell | Paper | Repo artifact | Repo value | Match |
|---|---|---|---|---|
| DINOv2 | 90.1 | `clevr_dinov2_cls_scratch_s42/train_log.jsonl` | 0.9014 | ✓ |
| SigLIP | 84.8 | `clevr_siglip_cls_scratch_s42/stdout_v2.log` | 0.8476 | ✓ |
| Sup-ViT | 86.6 | `clevr_sup_cls_scratch_s42.log` | 0.8663 | ✓ |
| MAE | 77.0 | `clevr_mae_cls_scratch_s42.log` | 0.7701 | ✓ |

## 5. CoGenT

| Paper cell | Paper | Repo artifact | Repo value | Match |
|---|---|---|---|---|
| ValA (train A) | 92.4 | **LEGACY**: `SteerViT-legacy/exp_vqa/outputs/phase1_clevr/cogent_odd_scratch_decoder_1l/train_log.jsonl` **epoch 11** val_acc 0.9243 (run continued to ep15 = 0.9449). Main-repo rerun: `cogent_dinov2_decoder1l_scratch_s42` final 0.9408 | 0.9243 (legacy ep11) | ✓ legacy, mid-training epoch (see §8.2) |
| ValB zero-shot | 88.0 | **no persisted artifact anywhere** (legacy `--eval_only` valB config exists — `cogent_decoder_odd_gca_1layer_evalB.yaml` — but stdout was never saved). Main-repo equivalents: `sample_efficiency.json` "before" valB 0.89479 | none | ✗ **cannot reproduce** (see §8.2) |
| ValB after ft | 92.7 | `cogent_sample_efficiency/50k_8ep.log` (50k, 8 epochs): valB 92.7 (valA 92.4 after ft) | 92.7 | ✓ |
| Sample-efficiency curve (figure) | — | `cogent_sample_efficiency/sample_efficiency.json` + `.png` (1k–50k × 4 epochs) | | ✓ |

Note: post-ft `50k_8ep.log` also prints valA 92.4 — numerically equal to the paper's
Table 1 DINOv2 cell and its CoGenT ValA cell; possible source-of-copy confusion in
the paper draft. Legacy-repo hunt (below) is checking whether an older run actually
scored 92.4/88.0.

## 6. Transfer table (paper Table 6) — zero-shot vs fine-tune-all, dinov2 decoder1l

All from `results.json` inside the ft run dirs (verified 2026-07-05):

| Dataset | Paper zs / ft | Repo artifact | Repo values |
|---|---|---|---|
| CLEVR-Humans | 54.5 / 73.8 | `clevr_dinov2_decoder1l_scratch_s42_humans_ft_all/results.json` | 0.54499 / 0.73785 ✓ |
| CLEVR-Math | 10.5 / 94.1 | `clevr_dinov2_decoder1l_scratch_s42_clevrmath_ft_all/results.json` | 0.10459 / 0.94079 ✓ |
| CLOSURE | 57.9 / 70.3 | `clevr_dinov2_decoder1l_scratch_s42_closure_ft_all/results.json` | 0.57873 / 0.70286 ✓ |

Partial-freeze variants (`*_ft_connector`, `*_ft_gca_connector`) also exist for
Humans/Math (`clevrmath_ft_connector` died during zero-shot — no results).

## 7. Mechanistic figures (all on `clevr_dinov2_decoder1l_scratch_s42`, GCA-decoder)

| Paper content | Repo artifact |
|---|---|
| Headwise patching heatmaps (binding heads L5H0 color, L7H9 material, L7H11 size, L7H3 shape; SA11) | `outputs/analysis/activation_patching/clevr_dinov2_decoder1l_scratch/headwise_by_type_stats.json` + `headwise_fine_attribute{,_query}_denoising.png` |
| Visual-corruption (perturbation C) patching | same dir, `visual_{color,material,shape,size}_stats.json` |
| Path patching / circuit | `outputs/analysis/path_patching/` + `path_patching_methodology.md` |
| Back patching | `outputs/analysis/back_patch/clevr_dinov2_decoder1l_scratch/` |
| Binding interchange | `outputs/analysis/binding_interchange/` |
| Gate-mediated intervention | `outputs/analysis/grounding_manipulation{,_acc}/clevr_dinov2_decoder1l_scratch/` |
| Conditional RSA (3 attr_query categories) | `outputs/analysis/conditional_rsa/concat_decoder_1l/` (main model!) + `clevr_siglip_decoder1l_scratch/`, `clevr_dinov2_nogate_scratch/` |
| Linear probes | `outputs/analysis/linear_probe/concat_decoder_1l/probe_results.json` (main model!) + cls / nogate / siglip / single_object / multi_object |
| t-SNE stage figures | `outputs/analysis/tsne/clevr_dinov2_decoder1l_scratch/` (+ per-backbone, single_object*, `legacy_dinov2_decoder1l`) |
| Gate values | `outputs/analysis/gate_values.png` |
| Alpha sweep | `outputs/analysis/cogent_zeroshot/zeroshot_alpha_sweep.json` |

**Plan impact**: probe + conditional RSA on the paper main model already exist under
the (non-standard) name `concat_decoder_1l` — E4's gap is only the *GCA-decoder*
(`clevr_dinov2_decoder1l_scratch`) probe/RSA, i.e. matching the mechanistic checkpoints.

## 8. Unresolved / flagged to user

1. **DINOv2 92.4 RESOLVED — it IS the main-repo concat run.** All four concat
   checkpoints store epoch-15 `val_acc` matching Table 1 exactly: DINOv2 0.92375,
   SigLIP 0.92556, Sup 0.86551, MAE 0.74763 → 92.4/92.6/86.6/74.8. **Table 1 is
   single-provenance** (main-repo concat, s42). The earlier confusion came from
   **directory contamination**: `clevr_dinov2_concat_decoder1l_scratch_s42/stdout.log`
   (final Val acc 0.9437) belongs to a *different, unidentified* run — the checkpoint
   in that dir and the top-level `clevr_dinov2_concat_decoder_scratch_s42.log` agree on
   0.9237. (Same dir also contains a stray `clevr_siglip_nogate_scratch_s42_ckpt/`
   subdir.) The legacy `odd_scratch_decoder_1l` run (0.9248) is numerically close but
   NOT the source. Camera-ready action: none needed for the number; optionally clean up
   the contaminated dir listing in a README note (never delete the files).
2. **CoGenT ValA 92.4 = legacy `cogent_odd_scratch_decoder_1l` at epoch 11** (not the
   final epoch — ep15 reached 0.9449); **ValB 88.0 has no persisted artifact anywhere**
   (the `--eval_only` valB stdout was never saved). The main-repo rerun gives 94.5/89.5
   (`sample_efficiency.json` "before"). → camera-ready should adopt the reproducible
   main-repo numbers (they are *better*) or rerun the legacy eval.
3. **Seed coverage**: paper claims 3 seeds (42/43/44). Repo has **no s44 anywhere**;
   MAE has only s42. Existing s43 runs are anomalous: `clevr_dinov2_decoder1l_scratch_s43`
   train_log stale at ep7 (0.8295), `clevr_siglip_decoder1l_scratch_s43` stale at ep3
   (0.5494), `clevr_sup_decoder1l_scratch_s43` stopped ~ep6 (0.6975). The tabled
   **concat** variant has s42 only. → user decision (E1c): run missing seeds or note
   single-seed in camera-ready.
4. **Per-category columns of Tables 1–5**: DINOv2 row recovered from legacy
   `eval_breakdown.json` (§2); all other rows have no artifact — regenerate via E1b.
5. **Baselines with no recoverable numbers**: `clevr_flamingo_dinov2_early_s42`
   (last.pt only, no eval), `clevr_llava_dinov2_lora_s42` (empty dir),
   `clevr_transfusion_scratch_s42` (config only, never trained),
   `clevr_dinov2_mean_scratch_s42` (crashed mid-eval; last.pt exists → E1a evaluates).
   MoT baseline is fine: `clevr_mot_scratch_s42/train_log.jsonl` ep15 = 0.7483.

## 9. Full accuracy matrix (R1, final-epoch val acc, seed 42 unless noted)

| Variant | DINOv2 | SigLIP | Sup-ViT | MAE |
|---|---|---|---|---|
| concat_decoder1l (paper main) | 0.9237 | 0.9256 | 0.8655 | 0.7476 |
| decoder1l (GCA-decoder) | 0.9095 | 0.9297 | 0.9376 | 0.7420 |
| cls | 0.9014 | 0.8476 | 0.8663 | 0.7701 |
| nogate (unreported) | 0.9683 (s43 0.9632) | 0.9457 (s43 0.9527) | 0.9439 | 0.9136 |

Others: gca_scratch 0.5277 · learned_text 0.2456 (final; best 0.4667) ·
concat nogca 0.4945 · MoT 0.7483.
s43 decoder1l runs incomplete (see §8.3).

**Byproduct rows (policy 2026-07-05: not project artifacts — never eval/analyze/claim
on them)**: nogate (row above, kept only for the RESULTS.md §2 gate note), film 0.8672,
mean (epoch-0 ckpt only, unrecoverable), 20ep, sup/mae decoder1l, flamingo/llava/transfusion
(no numbers). siglip decoder1l 0.9297 is sanctioned solely as the E3 second-backbone
mechanistic model.

## 10. Regenerating commands

`INTERP=/nfs/turbo/coe-chaijy/jungchun/vault/a-concept/comp-visual-reasoning/SteerViT-legacy/.venv-aspen/bin/python`, run from `main/`.

- Train any table cell: `PYTHONPATH=src $INTERP scripts/train.py +experiment=<config-name>` (+ `training.seed=<s> wandb.name=<dir>_s<s>` for extra seeds)
- Overall accuracy of a checkpoint: `PYTHONPATH=src $INTERP scripts/evaluate.py +experiment=<config-name> checkpoint=outputs/model/<dir>/best.pt`
- Per-category breakdown: `scripts/eval_generalization.py --checkpoint … --skip-closure --skip-cogent --skip-humans`
- Transfer (Table 6): `scripts/eval_{closure,clevr_math,legacy_humans}.py` (zero-shot + ft results already in the `*_ft_*/results.json` files above)
- Patching: `scripts/analysis/activation_patching.py --checkpoint … --groups fine_attribute,fine_attribute_query --directions denoising` (+ `--visual-pairs` for perturbation C)
- RSA / probe: `scripts/analysis/conditional_rsa.py --categories attr_query_direct,attr_query_same,attr_query_spatial` / `scripts/analysis/linear_probe.py`

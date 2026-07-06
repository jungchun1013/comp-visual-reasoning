# JOURNAL

## TODO
> [!NOTE] Persistent until done or removed. Every item requires a bracketed tag.
> [!NOTE] Tags: `[model]`, `[data]`, `[metrics]`, `[infra]`, `[plot]`, `[main flow]`, `[paper]`, `[ablation]`, `[debug]`

- [ ] [main flow] Confirm EXPERIMENT.md objective with user (initialized 2026-07-05 from the user's v2 outline)
- [x] [paper] R0 residual: resolve DINOv2 92.4 + CoGenT 92.4/88.0 provenance (RESOLVED; see docs/paper_artifacts.md §8)
- [ ] [paper] Camera-ready: CoGenT ValB 88.0 unreproducible — adopt main-repo 94.5/89.5 or rerun legacy eval — user decision
- [ ] [paper] E1c seed decision — paper claims seeds 42/43/44; repo has no s44, concat variant s42 only → user decides: run seeds or note single-seed
- [ ] [paper] R4: transfusion baseline has no checkpoint — retrain or drop from baseline table? (user decision)
- [ ] [metrics] E1b per-qtype breakdowns: running in background at low priority (user 2026-07-05: E1 可以慢慢跑不佔主線). PAPER MODELS ONLY, 13 ckpts, order = Table 1 concat → Table 4 cls → ablations → mechanistic decoder1l → MoT. Idempotent resume: `bash <scratchpad>/run_e1_evals.sh`.
- [ ] [main flow] **POLICY (user 2026-07-05)**: non-paper run dirs (nogate/film/mean/20ep/sup+mae decoder1l/flamingo/llava/transfusion) are training-exploration byproducts — do not eval, analyze, or build claims on them. Exception: existing gate writeup (RESULTS.md §2) stays; siglip_decoder1l sanctioned for E3.
- [ ] [model] dinov2_mean rerun if wanted: only an epoch-0 last.pt exists (crashed run; ep0 acc 0.218 — meaningless). Retrain or drop the cell.
- [ ] [paper] R2: write gate-as-design-choice framing (evidence: alpha sweep, grounding_manipulation, gate_values.png)
- [ ] [ablation] E3: activation patching on SigLIP decoder1l; E4: probe+RSA on dinov2 decoder1l (GCA-decoder; concat main model already has both under `concat_decoder_1l`)
- [ ] [metrics] E5: failure_modes.py + agent-autonomous diagnosis (yes/no worst — why)
- [ ] [data] E7: add-object hallucination (Blender smoke test first); E8: raw-backbone substrate probing
- [ ] [plot] E9: A/B/C × {CA,SA} localization contrast table+figure; E10: 2-stage-label replots (new dirs)
- [ ] [infra] W2: scripts/analysis/aggregate_results.py → docs/results_tables.md
- [ ] [infra] L1–L6: legacy distillation (docs/legacy-reference.md, plot-style consolidation, checkpoint_io dedupe, BLENDER_TOOLS_ROOT, routing, scripts/ cleanup)
- [ ] [infra] P1–P6: publish (commit WIP, release/public path refactor, README, LICENSE, smoke tests)

## Today's Progress
> [!NOTE] Append entries as work happens. Write so a stranger understands three months later.

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

## Log
> [!NOTE] Day Rotation inserts archived entries here. Newest on top.

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

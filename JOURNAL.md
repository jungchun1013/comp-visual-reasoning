# JOURNAL

## TODO
> [!NOTE] Persistent until done or removed. Every item requires a bracketed tag.
> [!NOTE] Tags: `[model]`, `[data]`, `[metrics]`, `[infra]`, `[plot]`, `[main flow]`, `[paper]`, `[ablation]`, `[debug]`

- [ ] [main flow] Confirm EXPERIMENT.md objective with user (initialized 2026-07-05 from the user's v2 outline)
- [x] [paper] R0 residual: resolve DINOv2 92.4 + CoGenT 92.4/88.0 provenance (RESOLVED; see docs/paper_artifacts.md §8)
- [x] [paper] Camera-ready CoGenT: DECIDED (user 2026-07-05) — draft 92.4/88.0 was a transcription error; adopt main-repo 94.5/89.5 (zs) and 92.7 (ft). Apply when editing the camera-ready text.
- [ ] [paper] Baseline survey: pretrained cross-attn from image-generation research as baseline (Transfusion weights? SD/PixArt/OpenFlamingo?) — survey running, then user picks
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
- [x] [infra] L1–L6: legacy distillation COMPLETE (legacy-reference.md; plot-style: 7 scripts on plot_style.py, replot-verified; checkpoint_io dedupe; BLENDER_TOOLS_ROOT; CLAUDE.md routing; scripts/README.md — L6 verdict: no scripts moved, "duplicates" are distinct active pipelines)
- [ ] [infra] P remaining: release/public branch path refactor (`src/model/model.py` hardcoded root top priority, `${oc.env:CLEVR_ROOT}` in configs), CLAUDE.md rm --cached on release branch, `grep -rn "/home/jungchun"` empty check. README/LICENSE/tests done on master.

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
- E1b ablation rows harvested (2026-07-06 01:40): gca_scratch eval 0.5277 == paper 52.8 EXACT; nogca ckpt best_acc 0.4945 == paper 49.4. **Three ablations fail three different ways**: −CA kills Count first (0.246 vs ~0.51 prior-plateau elsewhere); scratch-ViT keeps binary types above prior (Exist 0.664/CmpInt 0.672) but QryAttr starves at 0.490 (mechanism without substrate); learned_text collapses open-vocab generation entirely (QryAttr 0.000, Count 0.0004 — decoder emits degenerate strings, binary types at chance). A3 "all three necessary" now has per-component failure signatures. Caveat: learned_text row is best.pt(ep2), full-val 0.207 ≠ stored windowed 0.4667 — paper cell (24.6=last.pt) still needs the D2 last-epoch rerun. RESULTS.md §5.

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

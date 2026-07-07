# Presentation 素材清單

> 2026-07-07 整理。三部分：可直接取用的圖檔位置、八個實驗結果表的 LaTeX、每頁
> slide 可用的一句話主張。所有路徑相對於 `main/`；所有數字與
> `docs/results_overview_paper_style.md`、`RESULTS.md` 一致。
> 標籤現況：`_v3labels` 目錄與 `abc_contrast.png` 為 2026-07-07 命名
> （Binding → Retrieval，答案層級 = Answer classification）；其餘圖沿用產出當時的
> script 內部訊號名，與階段命名無衝突。

## 1. 圖檔位置

### E4 — 階段幾何（Binding → Retrieval）

| 檔案 | 內容 |
|---|---|
| `outputs/analysis/conditional_rsa/clevr_dinov2_decoder1l_scratch_v3labels/attr_query_direct/rsa_conditional.png` | conditional RSA 逐層曲線（direct）：Binding \| All、Retrieval \| Binding、Answer classification \| Binding，虛線為 unsteered control。**主圖候選** |
| `.../attr_query_same/rsa_conditional.png`、`.../attr_query_spatial/rsa_conditional.png` | 同上，relational categories（可見 anchor→target 交接） |
| `outputs/analysis/linear_probe/clevr_dinov2_decoder1l_scratch/probe_{direct,same,spatial}.png` | 逐層 linear probe 曲線（answer_decode / answer_match 等 probe 訊號） |

### E3 / A4 — patching 電路

| 檔案 | 內容 |
|---|---|
| `outputs/analysis/activation_patching/clevr_dinov2_decoder1l_scratch/headwise_fine_attribute_denoising.png` | DINOv2 headwise 熱圖（described-attr，A 擾動）：中段特化 GCA binding heads |
| `.../headwise_fine_attribute_query_denoising.png` | DINOv2 headwise 熱圖（queried-attr，B 擾動） |
| `outputs/analysis/activation_patching/clevr_siglip_decoder1l_scratch/headwise_fine_attribute_denoising.png`、`..._query_denoising.png` | SigLIP 對照組——與 DINOv2 並排即為 E3 複製證據 |

### E9 — A/B/C 定位梯度

| 檔案 | 內容 |
|---|---|
| `outputs/analysis/abc_localization/clevr_dinov2_decoder1l_scratch/abc_contrast.png` | 每屬性一組 A/B/C CA-share bar、0.5 無偏好線。**單圖即可承載 A4.1** |

### E8 — raw substrate

| 檔案 | 內容 |
|---|---|
| `outputs/analysis/raw_backbone_probe/vit_base_patch14_dinov2.lvd142m/raw_backbone_probe.png` | DINOv2 逐 block per-object 可解碼度曲線 |
| `.../vit_base_patch16_siglip_224/`、`.../vit_base_patch16_224.augreg_in21k/`、`.../vit_base_patch16_224.mae/` 同名檔 | 其餘三個 backbone，四張並排 = A1.2 |

### E10 — manipulation 因果測試（GCA-decoder）

| 檔案 | 內容 |
|---|---|
| `outputs/analysis/grounding_manipulation/clevr_dinov2_decoder1l_scratch_v3labels/manipulation_{grounding,answer,random}.png` | 各 manipulation 前後 RSA bar |
| `.../retrieval_{grounding,answer,random}.png` | 各 manipulation 前後 1-NN retrieval acc |
| `.../tsne_grounding_manipulation.png` | 操縱前後 t-SNE（圖例已用現行命名） |

### T2I — PixArt-Σ

| 檔案 | 內容 |
|---|---|
| `outputs/analysis/t2i_pixart/t2i_probe.png` | t=100（pre-registered）逐 block probe + CA 質量 |
| `outputs/analysis/t2i_pixart_t261/t2i_probe.png`、`outputs/analysis/t2i_pixart_t400/t2i_probe.png` | post-hoc sweep 對照 |

### E7 — add-object

無圖檔（數字型結果，用 §2 的 LaTeX 表）；刺激樣例影像在
`outputs/analysis/add_object/<attr>/images/`（base 與 added 成對，可挑一對當
示意圖）。

## 2. LaTeX 表格

均採 booktabs（`\usepackage{booktabs}`）。

### E1 — 準確率矩陣與 ablation

```latex
\begin{table}
\centering
\caption{Frozen-backbone CLEVR accuracy (concat readout, seed 42) and ablations.}
\begin{tabular}{lcccccc}
\toprule
Run & Overall & QryAttr & EqAttr & Exist & Count & CmpInt \\
\midrule
DINOv2      & 0.924 & 0.991 & 0.925 & 0.960 & 0.853 & 0.785 \\
SigLIP      & 0.926 & 0.990 & 0.921 & 0.964 & 0.863 & 0.786 \\
Supervised  & 0.866 & 0.940 & 0.839 & 0.914 & 0.792 & 0.742 \\
MAE         & 0.748 & 0.921 & 0.586 & 0.777 & 0.603 & 0.718 \\
\midrule
$-$CA (nogca)        & 0.459 & 0.516 & 0.522 & 0.570 & 0.246 & 0.501 \\
Scratch-ViT          & 0.528 & 0.490 & 0.517 & 0.664 & 0.459 & 0.672 \\
Learned-text (ep15)  & 0.197 & 0.000 & --    & --    & 0.003 & --    \\
\bottomrule
\end{tabular}
\end{table}
```

### E4 — 階段幾何層級地標（attr\_query\_direct）

```latex
\begin{table}
\centering
\caption{Stage geometry on the GCA-decoder model: half-rise and peak layers.}
\begin{tabular}{lcc}
\toprule
Signal & Half-rise & Peak \\
\midrule
Probe answer\_decode & L1 (0.66) & L11 0.92 (plateau L8--11) \\
RSA Binding $\mid$ All & L7 & L11 0.76 \\
Probe answer\_match & L5 & L11 0.77 \\
RSA Retrieval $\mid$ Binding & L9 & L11 0.57 \\
\bottomrule
\end{tabular}
\end{table}
```

### E3 — 跨 backbone 電路複製

```latex
\begin{table}
\centering
\caption{Top recovered head per signal ($\Delta$ logit, denoising), DINOv2 vs SigLIP.}
\begin{tabular}{lll}
\toprule
Signal & DINOv2 & SigLIP \\
\midrule
Color (described)    & GCA L3H1 $+0.77$  & GCA L5H11 $+1.08$ \\
Material (described) & GCA L9H10 $+0.81$ & GCA L5H12 $+1.25$ \\
Size (described)     & GCA L5H13 $+0.93$ & GCA L5H9 $+0.93$ \\
Shape (described)    & GCA L7H9 $+0.57$  & GCA L7H9 $+0.34$ \\
Query routing (shared) & GCA L7H2/H3, L9H15 & GCA L7H1/H14/H15 \\
Query-side SA        & L10--L11 (SA11 dominant) & L10--L11 (L11H10/H2, L10H6) \\
Described-side SA    & concentrated at L11 & mid-layer (L3--L7: L3H5, L7H6) \\
\bottomrule
\end{tabular}
\end{table}
```

### E9 — A/B/C 定位梯度

```latex
\begin{table}
\centering
\caption{Perturbation localization: CA share of per-head effect mass.}
\begin{tabular}{lccl}
\toprule
Perturbation & CA share & CA heads in top-10 & Strongest head \\
\midrule
A: described attr (text) & 0.53--0.55 & 4--5 & mid-layer CA (L3--L9) \\
B: queried attr (text)   & 0.43--0.49 & 2--5 & CA L7H3 ($\Delta$ $+1.5$--$2.0$) \\
C: queried attr (image)  & 0.20--0.43 & 0--3 & late SA (L11H0 / L11H11) \\
\bottomrule
\end{tabular}
\end{table}
```

### E5 — 失效模式（三表）

```latex
\begin{table}
\centering
\caption{Pre-registered failure-mode hypotheses.}
\begin{tabular}{lll}
\toprule
Hypothesis & Verdict & Key numbers \\
\midrule
H1: two-referent chains & Confirmed, refined & worst 8 families all two-set cardinality \\
 & & (count-over-union 0.52--0.64, compare-counts 0.61--0.74) \\
H2: yes/no prior collapse & Refuted & pred-no 0.504 vs gt-no 0.503; yes/no acc 0.9075 \\
H3: counting off-by-one & Confirmed & 86.9\% of 1{,}315 counting errors are $\pm 1$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}
\centering
\caption{Difficulty axis: referent multiplicity, not program depth.}
\begin{tabular}{lc}
\toprule
Condition & Accuracy \\
\midrule
query\_attribute, depth 4--20 & 0.97--1.00 (flat) \\
query\_attribute, deep chains (depth $\geq$18) & 0.992 \\
count, shallow $\to$ deep & 0.99 $\to$ 0.69 \\
single-set count, 0/1/2/3 spatial relations & 0.982 / 0.789 / 0.767 / 0.656 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}
\centering
\caption{Cross-model replication of per-family accuracy (Spearman $\rho$, 89 families).}
\begin{tabular}{lc}
\toprule
Model pair & $\rho$ \\
\midrule
GCA-decoder vs concat main       & 0.927 \\
Legacy SteerViT vs concat main   & 0.927 \\
Legacy SteerViT vs GCA-decoder   & 0.953 \\
\bottomrule
\end{tabular}
\end{table}
```

### E8 — raw substrate per-object 可解碼度

```latex
\begin{table}
\centering
\caption{Per-object attribute decodability of the raw backbone (peak over blocks).}
\begin{tabular}{lcccc}
\toprule
Backbone & Color & Material & Shape & Size \\
\midrule
DINOv2     & 0.966 (B7) & 0.987 (B10) & 0.986 (B11) & 0.997 (B8) \\
SigLIP     & 0.932 (B2) & 0.953 (B8)  & 0.951 (B8)  & 0.983 (B3) \\
Supervised & 0.938 (B2) & 0.934 (B2)  & 0.924 (B7)  & 0.983 (B6) \\
MAE        & 0.920 (B4) & 0.914 (B6)  & 0.914 (B9)  & 0.979 (B9) \\
\bottomrule
\end{tabular}
\end{table}
```

### E7 — add-object hallucination

```latex
\begin{table}
\centering
\caption{Add-object hallucination on identical stimulus pairs (100 pairs per attribute).}
\begin{tabular}{llcccc}
\toprule
Model & Metric & Color & Material & Shape & Size \\
\midrule
Concat main & hallucination\_rate & 0.02 & 0.00 & 0.01 & 0.06 \\
 & acc\_base $\to$ acc\_added & 0.98$\to$0.97 & 1.00$\to$1.00 & 1.00$\to$0.98 & 0.90$\to$0.94 \\
\midrule
$-$CA (nogca) & hallucination\_rate & 0.24 & 0.55 & 0.46 & 0.59 \\
 & acc\_base $\to$ acc\_added & 0.34$\to$0.28 & 0.55$\to$0.45 & 0.46$\to$0.39 & 0.46$\to$0.41 \\
\midrule
Legacy SteerViT & hallucination\_rate & 0.02 & 0.02 & 0.00 & 0.07 \\
 & acc\_base $\to$ acc\_added & 0.98$\to$0.98 & 0.99$\to$0.98 & 0.99$\to$1.00 & 0.93$\to$0.93 \\
\midrule
Flamingo (4/16 ep.) & acc\_base & 0.13 & 0.54 & 0.32 & 0.44 \\
\bottomrule
\end{tabular}
\end{table}
```

### T2I — PixArt-Σ zero-shot（negative）

```latex
\begin{table}
\centering
\caption{PixArt-$\Sigma$ zero-shot probing under question prompts.
Referent-local probe, best block (majority baseline in parentheses);
last column: peak CA mass on the referent window (chance $9/1024 \approx 0.0088$).}
\begin{tabular}{lccccc}
\toprule
$t$ & Color & Material & Shape & Size & CA mass peak \\
\midrule
100 & 0.169 (0.152) & 0.554 (0.566) & 0.421 (0.361) & 0.556 (0.527) & 0.0095--0.0096 \\
261 & 0.168 (0.159) & 0.689 (0.571) & 0.542 (0.400) & 0.556 (0.556) & 0.0097--0.0099 \\
400 & 0.230 (0.159) & 0.594 (0.571) & 0.500 (0.400) & 0.632 (0.556) & 0.0112--0.0117 \\
\bottomrule
\end{tabular}
\end{table}
```

## 3. 每頁一句話（slide takeaways）

- **Setting**：凍結 ViT + 凍結 text encoder，只訓練 GCA 與輕量 decoder，即達 CLEVR 0.92+。
- **E1**：三個元件各有 mechanism-level 崩塌 signature——移除 CA 先殺 counting、scratch-ViT 讓 retrieval 無基底可讀、learned-text 讓 open-vocabulary 生成崩解。
- **E4**：Binding 的 RSA 升起窗口與 patching 定位的 binding heads 同層（CA L3–L9），Retrieval 僅在 L9–L11 分離——相關性幾何與因果定位互鎖。
- **E3**：中段特化 GCA binding heads + 晚期 query-side SA 的 motif 在 SigLIP 完整複製；唯一偏離是 described-side SA 的整合位置。
- **E9**：文字端擾動修復以 CA 為主、影像端以晚期 SA 為主，成立為 gradient（A 0.53–0.55 > B 0.43–0.49 > C 0.20–0.43）。
- **E5**：難度軸是每步必須 bind 的 referent cardinality，不是 program depth；worst-family 結構跨三個獨立訓練模型複製（ρ 0.927–0.953）。
- **E8**：raw backbone 已 per-object 編碼屬性（0.91–1.00，四個 backbone）；缺的是 selection，不是資訊。
- **E7**：同一組 bait 刺激，訓練後擄獲 ≤6%、移除 CA 後 24–59%——grounding 的因果貢獻就是 Binding/selection。
- **T2I**：question prompt 下 PixArt-Σ 量不到 zero-shot binding（雙判準皆負）；機制需任務訓練誘發，此即本文貢獻的定位。

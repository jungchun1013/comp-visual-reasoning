# 結果總覽（paper 風格）

本文以論文導論的方式整理本專案目前的結果。先陳述整體動機、研究問題與假設框架，
再逐一呈現每一項實驗；每項實驗都附上代號，並依相同的段落順序敘述：動機、研究問題、
假設、設計、結果、詮釋。所有數字皆取自 `RESULTS.md`、`docs/paper_v2_outline.md`
與 `JOURNAL.md`，不作任何推估。術語固定為：Grounding 指整套語言條件化機制，其兩個
階段依序為 Binding 與 Retrieval；Retrieval 階段的兩個子訊號分別為 Retrieval
(object) 與 Retrieval (answer)。

---

## 整體動機與研究問題

近年的 vision foundation model（VFM）在大規模預訓練後，其凍結特徵已被廣泛用作下游
任務的表徵基底。本專案關切的問題是：**當一個 frozen VFM 面對多物件場景與自然語言查詢
時，它究竟缺少什麼，而語言條件化又補上了什麼。** 我們以 CLEVR VQA 為受控環境，在完全
凍結的 backbone 與 text encoder 之上，只訓練一個 gated cross-attention（GCA）模組與
一個輕量 decoder，藉此把「語言如何介入視覺表徵」隔離成唯一可訓練、可分析的變因。

具體研究問題為：frozen VFM 是否已經以 compositional 的形式編碼物件屬性；若已編碼，
語言條件化的因果貢獻落在哪個環節；這個環節是否具有可定位、可跨 backbone 複製的內部
電路；以及此機制在什麼情況下失效。

## 假設框架（A1–A5）

本專案的主張登錄於 `docs/paper_v2_outline.md`，五組假設構成後續每項實驗的檢驗對象：

- **A1（substrate）**：frozen VFM 已在無語言條件化下線性編碼物件屬性，且此編碼在
  多物件場景中維持在 per-object 層級；VFM 真正缺少的不是屬性資訊，而是在多個物件
  競爭時對被描述物件的 fixation。
- **A2（grounding = Binding → Retrieval）**：透過 gated cross-attention 注入查詢，
  可把凍結基底轉為 compositional reasoner；此條件化分兩階段展開——描述屬性在網路
  中段被 bound，被查詢屬性在網路末段被讀出。
- **A3（performance 與 architecture）**：機制可跨語意型與判別型 backbone 運作；
  pixel-reconstruction 預訓練（MAE）是一致的落後者，但其弱點須界定為 binding-usable
  結構較弱，而非屬性資訊缺失。
- **A4（mechanistic）**：文字端擾動的修復以 cross-attention 為主、影像端擾動以晚期
  self-attention 為主，此定位為一個 gradient 而非絕對；binding heads 的介入會因果
  地串連到 Retrieval，且電路 motif 可跨 backbone 複製。
- **A5（failure modes）**：失效集中在需要列舉並合併多個 referent 集合的問題，且此
  失效結構為 mechanism-level，可跨獨立訓練的模型複製。

`docs/paper_v2_outline.md` 另記有兩項經使用者核可的修訂：E7 的 headline metric 由
bait_share_of_errors 改為 hallucination_rate（前者在零錯誤時未定義、在二元屬性近
random 時退化為 1.0）；T2I 的 t=261/400 timestep sweep 記為 post-hoc robustness
check，非 pre-registered。

---

## Setting — CLEVR VQA 上的凍結基底 + gated cross-attention

**動機。** 若要把「語言介入視覺表徵」隔離為唯一變因，backbone 與 text encoder 都必須
凍結，讓所有可訓練參數集中在條件化與讀出。

**研究問題。** 在此極度受限的可訓練預算下，凍結基底能否被語言查詢驅動成 compositional
reasoner？

**設計。** 架構為 frozen pretrained ViT backbone + frozen text encoder + 可訓練的
gated cross-attention（GCA）+ 輕量 decoder。共比較四個 backbone（DINOv2、SigLIP、
supervised ViT、MAE）。讀出方式區分兩種：concat readout（把查詢併入 self-attention
讀出）與 GCA-decoder（讀出端再加一層 cross-attention）。論文的主模型為 concat
readout、seed 42；機制分析多在 DINOv2 GCA-decoder 上進行。

**結果。** concat readout 的 final-epoch val acc（seed 42，取自 checkpoint 內存的
`val_acc`）為 DINOv2 0.9237、SigLIP 0.9256、supervised 0.8655、MAE 0.7476；
GCA-decoder 為 0.9095 / 0.9297 / 0.9376 / 0.7420。

**詮釋。** 僅訓練條件化與讀出即可讓凍結基底達到 0.92 以上的 CLEVR 準確率，說明基底
本身已足以支撐 compositional 推理，缺的是介入機制而非表徵。concat 與 GCA-decoder 的
差異在後續 A3 的分析中成為關鍵：supervised ViT 在 concat 下明顯落後，在 GCA-decoder
下卻反超，指向讀出交互作用而非基底缺陷。

---

## E1 — 準確率矩陣與 per-question-type 拆解

**動機。** 要支撐 A2.1（cross-attention 是把基底轉為 reasoner 的關鍵）與 A3
（backbone 排序、三項 ablation 的必要性），需要一張單一 provenance 的準確率矩陣，
並拆到 question type 以定位每個元件負責什麼。

**研究問題。** 移除各個元件後準確率如何崩塌，且崩塌是否在特定 question type 上留下
可辨識的 signature？

**設計。** 在四個 backbone 上量測整體與 per-qtype 準確率（QryAttr、EqAttr、Exist、
Count、CmpInt），並加入三項 ablation：移除 cross-attention（−CA / nogca）、以 scratch
ViT 取代預訓練 backbone、以 learned text 取代預訓練 text encoder。所有 cell 取自
single-provenance 的 concat/cls seed 42 評估（`outputs/analysis/generalization/`）。

**結果。** backbone 排序為 SigLIP 0.926 ≈ DINOv2 0.924 > supervised 0.866 > MAE
0.748。per-qtype 上，DINOv2 QryAttr 0.991、Count 0.853、CmpInt 0.785；MAE 的 EqAttr
崩至 0.586（QryAttr 仍有 0.921）。三項 ablation 各以不同方式失效：−CA 整體 0.459，
其中 Count 掉到 0.246（遠低於其他型別約 0.50 的 answer-prior plateau）；scratch-ViT
整體 0.528，二元型別回升（Exist 0.664、CmpInt 0.672）但 QryAttr 僅 0.490；learned-text
整體 0.197，QryAttr 恰為 0.000、Count 0.003，僅二元型別落在 chance 約 0.48。

**詮釋。** 三項 ablation 對應 A3 的「三元件皆必要」主張，且各自帶有 mechanism-level
signature：移除 cross-attention 先殺掉 counting（需要在視覺 token 上迭代的能力）；
移除視覺預訓練讓 binding 存活但 retrieval 因缺乏結構化基底而挨餓；移除語言預訓練讓
open-vocabulary 生成整個崩解。supervised ViT 的 concat 缺口是 **uniform** 的（每個
型別都掉約 5 點，含 QryAttr 0.940 vs 0.991），而它在 GCA-decoder 下達 0.9376，故其
弱點應描述為 readout-sensitive，而非「不會 counting」。MAE 的缺口則集中在 two-referent
的 EqAttr，構成 A3 與 A5 之間的橋樑。

---

## E4 — linear probe 加 conditional RSA（GCA-decoder 模型）

**動機。** A2 主張條件化分 Binding → Retrieval 兩階段，A4 主張此順序有對應的因果
定位。若要把這兩者鎖在一起，需在**同一個**跑過 activation patching 的模型上，量出
相關性層面的階段幾何。

**研究問題。** Binding 與 Retrieval 的層級幾何是否與 patching 定位出的 head 層級一致，
且兩階段是否嚴格有序？

**設計。** 在 DINOv2 GCA-decoder 模型上，對每一層做 linear probe 與 conditional RSA
（GCA 層 1,3,5,7,9,11 加 12 個 ViT 層與 decoder probe），量測 answer_decode、
answer_match、Binding | All、Retrieval | Binding 各訊號的 half-rise 與 peak 層。

**結果。** probe answer_decode 在 L1 即 half-rise（0.66）、L8–11 達 0.92 plateau；
RSA Binding | All 在 L7 half-rise、L11 peak 0.76；RSA Retrieval | Binding 到 L9 才
分離、L11 peak 0.57。在每個 category，Binding half-rise 都比 Retrieval half-rise 早
2 層。relational category（same / spatial）另顯示 anchor binding 在 L8 達峰後於 L12
崩落（0.42→0.11），而 target binding 持續爬升至 L11。

**詮釋。** A2↔A4 lock 成立：相關性層面的階段幾何重現了因果定位——Binding 在 patching
所定位 binding heads（CA L3–L9，如 color L5H0、material L7H9、size L7H11、shape L7H3）
所在的中段窗口升起，Retrieval 僅在晚期（L9–L11，對應 patching 的 SA block 11）分離。
anchor→target 的交接顯示 relational chaining 以序列式 re-binding 實作，與 E5 中
「relational localization 便宜、enumeration 昂貴」的發現一致。依使用者決定，中間鏈層級
稱 Retrieval (object)、答案層級稱 Retrieval (answer)。

---

## E3 — 在 SigLIP 上複製 activation patching

**動機。** A4.3 主張電路 motif 可跨 backbone 複製。若機制只在 DINOv2 上成立，就無法
排除它是單一 backbone 的偶然結構。

**研究問題。** 在 SigLIP GCA-decoder 上做同樣的 patching，是否重現中段特化 GCA binding
heads 加晚期 SA retrieval 集中的三項結構 signature？

**設計。** 以與 DINOv2 reference run 完全相同的 pipeline 與設定（n=50/category，
denoising），在 SigLIP GCA-decoder 上跑 activation patching
（`outputs/analysis/activation_patching/clevr_siglip_decoder1l_scratch/`）。

**結果。** SigLIP 重現三項 signature：(1) 稀疏、屬性特化的 GCA binding heads 全落在
L3–L7（color L5H11 +1.08、material L5H12 +1.25、size L5H9 +0.93、shape L7H9 +0.34，
top head 為 runner-up 的 3–7 倍）；(2) 一小組共享的 query-routing GCA heads 位於 L7；
(3) query 側 SA 效應集中於最後兩層（L11H10/H2、L10H6）。shape head 在兩個 backbone
皆落在 L7H9（coincidence-grade 巧合）。唯一可報告的偏離：SigLIP 的 described-attribute
SA 效應分佈於中段（L3–L7）而非像 DINOv2 堆在 L11。

**詮釋。** 電路 motif 複製成立，head identity 為 backbone-specific（如預期）。Binding
階段的圖像（中段特化 GCA heads）完全不變；偏離僅限於 SA 於何處 re-integrate，故
retrieval 側整合在 SigLIP 上較為分散，而 Binding 定位穩健。此結果把 A4 的機制主張
從單一 backbone 推廣為語言條件化的一般性質。

---

## E9 — A/B/C 擾動 × {CA heads, SA heads} 定位

**動機。** A4.1 主張文字端擾動的修復走 cross-attention、影像端走晚期 self-attention。
`docs/paper_v2_outline.md` 明訂此定位須寫成 gradient 而非絕對，因為資料不支持「A only
affects CA, C only affects SA」的措辭。

**研究問題。** 三種擾動——A（described-attr，文字）、B（queried-attr，文字）、C
（queried-attr，影像）——在 CA 與 SA head 上的效應質量分佈，是否呈現單調的 A>B>C
梯度？

**設計。** 從既有 patching 統計（denoising，DINOv2 GCA-decoder，n=50/category）聚合
每個 head 的效應質量，計算各擾動的 CA-share，並產出對比圖
（`outputs/analysis/abc_localization/.../abc_contrast.png`，每屬性一組 CA-share bar，
含 0.5 無偏好線）。

**結果。** CA 佔 per-head 效應質量的比例呈乾淨梯度：A 0.53–0.55 > B 0.43–0.49 >
C 0.20–0.43。top-10 最強 head 中 CA head 數量為 A 4–5 個、B 2–5 個、C 0–3 個。
C 的最強 head 對每個屬性都是晚期 self-attention（L11H0 / L11H11）；A 的最強 head 為
中段 cross-attention（L3–L9，含已知 binding heads L7H9、L7H3）。B 在 CA L7H3 上有強峰
（what-color / what-material 的 Δ +1.5–2.0）。少數例外：C-size 有一個大的 CA outlier
（L9H14，Δ +3.36），C-shape 是最純的 SA 情形（CA share 0.199，top-10 內 0 個 CA head）。

**詮釋。** 定位主張以清楚的 gradient 成立：文字端擾動的修復以 cross-attention
（Binding）為主，影像端擾動以晚期 self-attention（Retrieval 側整合）為主，而查詢屬性
的文字擾動 B 居中——與 B 同時觸及兩階段一致。絕對措辭不成立（C-size 即有一個大的 CA
效應），故正式敘述採 gradient 而非 only。

---

## E5 — 失效模式

**動機。** A5 主張失效集中在需列舉並合併多個 referent 集合的問題。既有草稿曾把最差型別
歸因於 yes/no 的 answer prior，需以資料裁決此說並找出真正的難度軸。

**研究問題。** 難度是由 program depth 決定，還是由每個 query step 必須 bind 的 referent
cardinality 決定？此失效結構是 readout-level 還是 mechanism-level？

**設計。** 在 concat 主模型上 dump per-question 記錄（n=37,498，stride 4，整體 0.9240
≈ full-val 0.9237），裁決三項 pre-registered 假設 H1–H3，並依 program depth 與問句
中的空間關係數分層；再在 GCA-decoder 與 legacy SteerViT 上重跑以測跨模型複製
（`outputs/analysis/failure_modes/`）。

**結果。** H2（yes/no answer-prior collapse）**被否證**：pred-no 0.504 vs gt-no 0.503，
confusion 對稱，yes/no acc 0.9075，無多數類偏誤。H3（counting off-by-one）**被確認**：
1,315 個 counting 錯誤中 86.9% 為 ±1。H1（two-referent chains）**被確認並細化**：8 個
最差 family 全是 two-set cardinality 問題（count-over-union acc 0.52–0.64、
compare-counts 0.61–0.74）。難度軸為 referent multiplicity 而非 program depth：
query_attribute 在 depth 4 到 20 之間平坦維持 0.97–1.00，而 count 隨 depth 由 0.99
降至 0.69。同一組空間關係詞在 query_attribute 中零成本（depth ≥18 的深鏈 acc 0.992），
在 single-set count 中卻讓準確率隨關係數崩落（0 rel 0.982 → 3 rel 0.656）。跨模型
複製：GCA-decoder 對 concat 的 per-family 準確率 Spearman ρ = 0.927；legacy SteerViT
對 concat ρ = 0.927、對 GCA-decoder ρ = 0.953（89 個 family）。

**詮釋。** 難度由每個 query step 必須 bind 的 cardinality 設定：單一物件在任何 program
depth 都是 ceiling 準確率，變動大小的集合須全數列舉時成本逐步累加，數個集合須列舉並
合併時最差。program depth 與 relational vocabulary 本身零成本，瓶頸是 query 機制
one-referent-at-a-time 的 binding，而非對長程或關係性程式的語言理解。跨三個獨立訓練、
橫跨兩個 codebase 世代的模型（Spearman ρ = 0.927 / 0.953）皆重現同一 worst-family
結構，確立此失效為 mechanism-level。

---

## E8 — raw-backbone substrate probing

**動機。** A1.2 主張屬性編碼在多物件場景中維持在 per-object 層級。E7 已顯示訓練後的
fixation 穩健，但要完成 A1→A2 的論證，必須先確認**未經語言條件化的原始基底**是否
就已 per-object 地編碼屬性。

**研究問題。** 在完全無語言條件化的 raw backbone 前向傳播下，多物件場景中每個物件的
屬性是否線性可解碼？

**設計。** 以 fresh zero-gated GCA（等同純 pretrained ViT 前向）處理 300 個多物件場景
（1,917 個物件），在每個物件的 pixel_coords 做 3×3 patch pooling，對每個 block 做
5-fold logistic regression（`outputs/analysis/raw_backbone_probe/`）。

**結果。** per-object 屬性可解碼度（取 block 峰值）在四個 backbone 上皆為 0.91–1.00：
DINOv2 color 0.966 / material 0.987 / shape 0.986 / size 0.997；SigLIP 0.932–0.983；
supervised 0.924–0.983；MAE 0.914–0.979。

**詮釋。** A1.2 確認：原始基底在多物件場景中即以 per-object 形式編碼屬性，早於任何
語言條件化或任務訓練；raw backbone 所缺的不是資訊而是 selection。A3 的 MAE nuance：
substrate 可解碼度排序雖與下游排序一致（DINOv2 > SigLIP ≈ supervised > MAE），但
substrate 差距很小（MAE color 0.920 vs DINOv2 0.966），而下游 VQA 差距很大（0.748
vs 0.924）。因此 per-object peak 可解碼度本身無法解釋 MAE 的失效，「較弱 substrate」
必須界定為較弱的 binding-usable 結構，而非較弱的屬性資訊——與 E5 中 MAE 的 EqAttr
崩塌一致。

---

## E7 — add-object hallucination

**動機。** A1.3 把 VFM 的瓶頸定義為「多物件競爭時對被描述物件的 fixation」。需要一個
因果測試，判定訓練後的 grounding 機制是否解決此 fixation，以及移除 cross-attention
是否恰好以 fixation 失效。

**研究問題。** 加入一個匹配大部分描述屬性、但在被查詢屬性上帶有 bait 值的干擾物件後，
答案不變的前提下，模型的預測是否被 bait 擄獲？

**設計。** 每個屬性 100 對影像，干擾物件為「翻轉一個 described-attr + 在 queried attr
上放 bait 值」，答案不變性以 program execution 驗證，base 重繪控制 render-domain shift
（`outputs/analysis/add_object/`）。同一組 pair 另跑在 −CA（nogca）模型與 legacy
SteerViT 上。headline metric 為 hallucination_rate。

**結果。** 訓練後的 concat 主模型穩健：hallucination_rate 為 color 0.02 / material
0.00 / shape 0.01 / size 0.06，acc_base→acc_added 移動 ≤2 點（color 0.98→0.97、
material 1.00→1.00、shape 1.00→0.98、size 0.90→0.94）；少數錯誤呈 bait-shaped。相對地，
−CA 模型在同一組刺激上 hallucination_rate 為 0.24–0.59，bait 擄獲率 24–59% vs GCA 的
0–6%，構成約 10× 的差距。E5-on-nogca 提供觀察面對照：6,516 個 query_attribute 錯誤中
98.6% 是場景內另一物件的屬性值，僅 1 個為 out-of-scene；限縮到 color 時 2,276 個錯誤
100.0% 落在場景內（vs 52.3% chance baseline）——編碼完好，selection 損壞。legacy
SteerViT 重現訓練後穩健（hallucination color 0.02 / material 0.02 / shape 0.00 /
size 0.07）。**Flamingo：** 首次量測無效（harness bug——`generate_answer` 小寫化解碼，
adapter 卻以大寫 `"Answer:"` 切分，使每筆預測都是 prompt echo）；修正後
（`*_fixed.json`）acc_base 為 color 0.13 / material 0.54 / shape 0.32 / size 0.44，
四屬性皆 chance level 且預測近乎退化（color 98% "yellow"、size 100% "small"），故 E7
的 hallucination / bait 指標對其無法詮釋。

**詮釋。** 論文用以動機化 A1 的多物件 fixation 問題，被訓練後的 grounding 機制對直接
查詢所解決（擄獲 ≤6%）；殘餘的多物件瓶頸是 set enumeration / combination（見 E5）。
fixation triangle 由此閉合：raw substrate（E8）資訊在、無 selector；−CA trained
（§8b）selection 損壞、編碼完好；+CA trained（主模型與 legacy SteerViT）selection
成立。grounding 的因果貢獻正是 Binding / selection 這一步——per-object 屬性由基底
提供、不需任何任務訓練，語言條件化決定讀出看向哪個物件。Flamingo 的 chance-level 結果使其 E7 leg 須待
retrain（frozen-LLM recipe，見目前狀態）。

---

## T2I — PixArt-Σ 的 zero-shot 分析

**動機。** baseline 的 reframe（見 `docs/paper_v2_outline.md`）要問：Binding 這類結構
是否在大規模預訓練的 cross-attention 中 zero-shot 就存在？若 text-to-image 模型
不經任何微調即具備此結構，機制便是語言條件化的一般性質而非 CLEVR 訓練產物。

**研究問題。** PixArt-Σ 的 cross-attention 在 question prompt 下，是否 zero-shot 就對
被指涉物件展現 binding？

**設計。** 以 DIFT 式的 small-t 特徵抽取，套用 pre-registered 的雙重判準：(a)
referent-local probe 在 mid/late block 須 ≫ majority baseline；(b) referent 3×3 window
上 column-normalized 的 CA 質量須 ≫ chance（9/1024 ≈ 0.0088）。pre-registered
timestep 為 t=100（n=300/cat）；t=261/400（n=150/cat）為 post-hoc robustness check
（`outputs/analysis/t2i_pixart*/`）。

**結果。** referent-local probe 的超越 majority baseline 者在 block 與屬性間零散分佈、
無一致的 mid/late 結構（最佳如 t=261 material 0.689 vs 0.571、t=400 color 0.230 vs
0.159）。CA referent-window 質量峰值始終落在 1.0–1.33× chance 帶內（t=100 0.0095–0.0096、
t=261 0.0097–0.0099、t=400 0.0112–0.0117）；唯一結構化殘餘是 t=400 三個 category 的
CA 峰值同落於 block B6，但仍遠低於可用的 binding 訊號。

**詮釋。** 在每個測試 timestep，pre-registered 判準下皆為 **NEGATIVE**。依 pre-registered
的 caveat（question ≠ caption，對 T2I text encoder 構成 domain mismatch），此負結果
無法區分「機制不存在」與「prompt 分佈不匹配」。有界結論：PixArt-Σ 的 cross-attention
在 question prompt 下量不到 zero-shot binding——如此量測的 grounding 需要任務訓練。
主論文的貢獻因而定位為：展示 VQA 訓練如何從 pretrained substrate 中引出此機制。

---

## 目前狀態

已完成並可入稿的部分涵蓋核心結果：準確率矩陣與 ablation（E1）、階段幾何與 A2↔A4 lock
（E4）、跨 backbone 電路複製（E3）、A/B/C 定位梯度（E9）、失效模式與跨三模型複製
（E5）、raw substrate 的 per-object 可解碼度（E8）、add-object hallucination 與
fixation triangle（E7 的訓練後、−CA、legacy SteerViT 三角），以及 T2I 的 negative
verdict（T2I）。

進行中：Flamingo 的 frozen-LLM retrain（`clevr_flamingo_dinov2_frozenllm_s42`，
2026-07-06 launched，只訓練 GCA + connector 共 106M 參數、LLM 凍結）正在跑，結果待收——
先取 final-epoch val acc，再以 `*_fixed` 協定重跑 E7；舊的 4/16-epoch checkpoint 維持
qualitative-only reference。

待使用者決策：E1c 的 seed 議題（論文宣稱 seed 42/43/44，repo 目前只有 s42）；
learned-text 的 paper cell 為 protocol-dependent（training-log 0.2456 vs 獨立協定
0.197/0.207），camera-ready 採 footnote 或改號待定；R4 的 transfusion baseline 重訓或
移除；以及 gated 於 pre-registration 的其餘 baseline（OpenFlamingo zero-shot 機制分析、
T5-vs-RoBERTa capacity axis）是否啟動。T2I 的 declarative-caption 變體為可選的
domain-mismatch 追蹤，僅在使用者要求時進行。

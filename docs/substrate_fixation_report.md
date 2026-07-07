# 實驗報告：基底與凝視（Substrate & Fixation）——claims A1–A2 的整合證據

> 論文中文草稿用節。撰寫：Opus agent，2026-07-06（user 指示：實驗報告由 Opus 撰寫）。
> 事實來源：`RESULTS.md` §8、§8b、§8c、§9、§11、§13、§14；`docs/paper_v2_outline.md`
> §A1–§A2、§T2I；`docs/t2i_experiment_design.md`。所有數字均可依文中所引 artifact
> 路徑回溯，未經 artifact 佐證者不入本節。
>
> 術語遵循命名規範：**Grounding** 指整個語言 conditioning 機制，其階段依序為
> **Binding**（將描述屬性綁定至被指涉物體）與 **Retrieval**（讀出被詢問屬性），
> Retrieval 階段的兩個子訊號為 **Retrieval (object)** 與 **Retrieval (answer)**。
> 全節不使用「object grounding」或「Object match」作為階段名稱。

## 研究問題

本文的核心主張分為兩層。其一（claim A1）：預訓練的視覺基礎模型（VFM）已在其表徵中
編碼出結構化、可組合的資訊，逐物體（per-object）地保存了每個物體的屬性；預訓練基底
**欠缺的並非屬性資訊，而是在多物體場景中對「被描述的那個物體」的凝視與選取**。其二
（claim A2）：語言 conditioning（經 gated cross-attention，GCA，注入問題）正是把這個
靜態基底轉化為組合式推理器的機制，其運作可分解為 Binding → Retrieval 兩階段。

把這兩層主張放在一起，即得到一個可由實驗裁決的因果問題：**當 grounding 把 CLEVR 上的
VQA 準確率從無語言 conditioning 的水準提升到 92% 時，它真正貢獻的那一步是什麼？** 一種
可能是它補上了基底所缺的屬性資訊；另一種可能是資訊本來就在基底裡，grounding 只是補上了
「選對物體」這一步選取器（selector）。本節以三項獨立量測裁決此問題（`RESULTS.md` 將此
三部分論證命名為 fixation triangle）：**證據一（E8）**量測未經任何語言 conditioning 的
原始基底是否已逐物體編碼屬性；**證據二（E5/E7，−CA 消融）**量測移除 cross-attention 的
訓練模型在何處失敗；**證據三（E7，訓練模型）**量測完整 grounding 的訓練模型在同一組
刺激下是否穩健。三項量測合併即回答上述因果問題。

本節另收兩項延伸證據，用以界定主張的邊界：一項 T2I 模型上的 zero-shot 檢驗，回應「此機制
是否只是 CLEVR 訓練的 artifact」的質疑；以及一則本地 Flamingo 基線的現況說明。

## 證據一：原始基底已逐物體編碼屬性（E8，raw-backbone probing）

**方法。** 我們取四個預訓練視覺骨幹——DINOv2、SigLIP、監督式 ViT（sup-ViT）、MAE——在
**完全不施加任何語言 conditioning、不作任何 task 訓練**的條件下量測其基底。具體作法是把
GCA 的 tanh 閘門歸零（zero-gated GCA），使前向傳遞退化為純粹的預訓練 ViT forward；問題
文字不進入 trunk。資料為 300 個多物體 CLEVR 場景、共 1,917 個物體。對每個物體，依其
`pixel_coords` 在特徵網格上取 3×3 patch 窗作 mean-pool，於每一個 block 以 5-fold logistic
regression 預測該物體的四項屬性。報告各骨幹在所有 block 上的**峰值**逐物體可解碼率。
Artifact：`outputs/analysis/raw_backbone_probe/<backbone>/`。

**結果。** 逐物體屬性可解碼率（block 峰值）如下：

| backbone | color | material | shape | size |
|---|---|---|---|---|
| DINOv2 | 0.966 (B7) | 0.987 (B10) | 0.986 (B11) | 0.997 (B8) |
| SigLIP | 0.932 (B2) | 0.953 (B8) | 0.951 (B8) | 0.983 (B3) |
| sup-ViT | 0.938 (B2) | 0.934 (B2) | 0.924 (B7) | 0.983 (B6) |
| MAE | 0.920 (B4) | 0.914 (B6) | 0.914 (B9) | 0.979 (B9) |

四個骨幹、四項屬性全數落在 **0.91–1.00** 區間。此結果確立 claim A1.2：在多物體場景中，
逐物體的屬性資訊**在語言 conditioning 之前、在任何 task 訓練之前**就已存在於原始基底裡。
換言之，基底缺的不是資訊，而是「用哪個物體」的選取——而選取正是證據二與證據三所檢驗的。

## 證據二：移除 cross-attention 後，失敗是「選錯物體」而非「編碼遺失」（E7 + E5，−CA 消融）

**方法。** 受測模型為 `clevr_dinov2_concat_decoder1l_nogca_scratch_s42`（−CA：trunk 內無
語言 conditioning，問題僅由 concat readout 進入）。我們在此模型上跑兩項互補分析。

其一是 **add-object 幻覺測試（E7）**。對每個被詢問屬性，構造 100 對刺激：在原場景中加入
一個「誘餌（bait）」物體，它匹配描述中除被詢問屬性外的所有屬性，但在被詢問屬性上帶一個
誘餌值；由於它不符合完整描述，經 program 執行驗證，正確答案**不變**。一個真正凝視被描述
物體的模型不受影響；一個「特徵袋（bag-of-features）」式的綁定器則會被誘餌捕獲。base
場景重新渲染以控制 render-domain 位移。Artifact：`outputs/analysis/add_object/<attr>/…nogca…json`。

其二是 **失敗模式分類（E5）**，作為觀察面的對照：把該模型在 CLEVR 上所有 query_attribute
錯誤逐一比對場景 ground truth，看被誤讀出的屬性值是否恰為場景中「另一個物體」的值。
Artifact：`outputs/analysis/failure_modes/clevr_dinov2_concat_decoder1l_nogca_scratch_s42/`。

**結果（E7）。** −CA 模型的誘餌捕獲率（hallucination_rate）：

| queried attr | acc_base | acc_added | hallucination_rate | bait_share_of_errors |
|---|---|---|---|---|
| color | 0.34 | 0.28 | 0.24 | 0.33 |
| material | 0.55 | 0.45 | 0.55 | 1.00 |
| shape | 0.46 | 0.39 | 0.46 | 0.75 |
| size | 0.46 | 0.41 | 0.59 | 1.00 |

一個新加入的誘餌物體，即捕獲 −CA 模型 **24–59%** 的答案。

**結果（E5）。** 在全部 6,516 個 query_attribute 錯誤中，**98.6% 被誤讀出的值恰是場景內
另一個物體的屬性值**；出場景（憑空捏造）的幻覺僅 1/6516。限縮到 color（8 個值，故「在
場景內」並非平凡事件）：**2,276 個錯誤顏色中 100.0% 都出現在場景裡**，而隨機挑一個錯誤
顏色落在場景內的 chance 基線僅 52.3%。也就是說，−CA 模型讀出的是一個**真實存在的物體**——
只是不是被描述的那個。屬性編碼完好，失效的是選取。此結論與證據一一致：屬性資訊存在，選取步驟缺失。

## 證據三：完整 grounding 使選取穩健（E7，主模型與 legacy SteerViT）

**方法。** 對完整 grounding 的訓練模型 `clevr_dinov2_concat_decoder1l_scratch_s42`（+CA），
跑與證據二**完全相同**的 E7 刺激（每屬性 100 對，答案不變性經 program 執行驗證）。為檢驗
此穩健性是否只是單一模型或單一 codebase 的偶然，我們另在一個獨立訓練、跨 codebase 世代的
祖先模型 `odd_scratch_decoder_1l/best.pt`（legacy SteerViT）上重跑同一組刺激。
Artifact：`outputs/analysis/add_object/<attr>/…concat_decoder1l_scratch_s42.json`；
legacy 結果記於 `RESULTS.md` §8c。

**結果（主模型）。**

| queried attr | acc_base | acc_added | hallucination_rate | bait_share_of_errors |
|---|---|---|---|---|
| color | 0.98 | 0.97 | 0.02 | 0.67 |
| material | 1.00 | 1.00 | 0.00 | —（0 錯誤）|
| shape | 1.00 | 0.98 | 0.01 | 0.50 |
| size | 0.90 | 0.94 | 0.06 | 1.00 |

同一組在 −CA 模型上捕獲 24–59% 答案的誘餌刺激，在完整模型上僅捕獲 **≤6%** 的答案；準確率移動
≤2 pts（size +4 屬 n=100 雜訊）。在同一組刺激上的因果對比，是 **約 10 倍的幻覺率落差**，
且此落差可歸因於單一被加入的物體。少數確實發生的錯誤仍是誘餌形狀的（bait_share 0.5–1.0），
故失敗模式存在，只是稀有。size 一貫是最弱屬性（acc_base 最低 0.90、hallucination 最高
0.06），與它在證據一中是最不可分屬性的觀察一致。

在觀察面上，E5 對主模型的錯誤結構與 −CA 模型同構——主模型稀有錯誤中 131/133 同樣是場景內
其他物體的值；失敗**模式**共享，GCA 改變的是**發生率**：query_attribute 錯誤率由 −CA 的
48.6% 降至 1.0%，約 50 倍。

**結果（legacy SteerViT）。** 跨世代複製：hallucination color 0.02 / material 0.02 /
shape 0.00 / size 0.07；acc_base→added 全平（0.98→0.98、0.99→0.98、0.99→1.00、
0.93→0.93）；有錯處仍為誘餌形狀，size 又是最弱。E5-on-SteerViT（整體 0.9247、qryattr
0.992，記於 `outputs/analysis/failure_modes/odd_scratch_decoder_1l/`）的家族準確率與 concat
主模型 Spearman ρ = 0.927、與 GCA-decoder ρ = 0.953。訓練後凝視穩健的輪廓，因此在一個
獨立訓練的模型世代上重現。

**度量選用說明。** 依 A1.3 的預先登記，`bait_share_of_errors`（誘餌佔錯誤的比例）本用作
「隔離幻覺、排除一般分佈位移」的頭條指標。但實作上此指標在錯誤稀有時不穩定（主模型
size 僅少數錯誤即得 bait_share 1.00，material 0 錯誤而無定義），無法承載「捕獲率」的量。
因此本報告以 `hallucination_rate`（誘餌捕獲率）作為此三部分論證的主軸讀數（0–7% vs 24–59% 的
因果對比），並以 `bait_share_of_errors` 作為「錯誤確為誘餌形狀」的**形狀確認**輔助指標。
（此度量選用與 A1.3 的字面登記略有出入，見§限制。）

## 三項證據的綜合：grounding 的因果貢獻是 Binding／選取這一步

三項量測對同一因果問題給出一致的答案：

| 量測 | 結果 | 讀法 |
|---|---|---|
| 原始基底（E8，證據一）| 逐物體可解碼率 0.92–1.00，四骨幹 | 資訊存在，無選取器 |
| −CA 訓練（E5/E7 nogca，證據二）| 誘餌捕獲 24–59%；100% 的錯誤顏色為場景內他物之值 | 選取失效，編碼完好 |
| +CA 訓練（E7，證據三）| 誘餌僅捕獲 0–7%（主模型與 legacy SteerViT 皆然）| 選取運作 |

三項量測合併，把因果貢獻定位在**單一步**：基底本身已提供逐物體屬性（證據一）；移除語言
conditioning 後屬性資訊仍在，但讀出選錯物體（證據二）；具備完整 grounding 時，選取穩健
（證據三）。屬性資訊的有無在三項量測間不變，唯一隨 GCA 的有無而改變的是「讀出看哪個
物體」。因此
**grounding 的因果貢獻正是 Binding／選取這一步**——語言 conditioning 挑選讀出所面對的
物體，而非補上任何缺失的屬性資訊。這同時完成了 A1→A2 的論證：A1 動機所訴諸的「多物體
凝視」瓶頸，對 direct query 而言，正是被 A2 的 grounding 機制所解（多物體場景中殘餘的
瓶頸轉移到集合枚舉／組合，屬另議）。此結論與機制模型上的階段幾何一致（§9）：Binding
訊號在中層 GCA 綁定窗（CA L3–L9）上升，Retrieval 訊號僅在晚層（L9–L11）分離，二者
half-rise 在每個類別皆嚴格相差約 2 層，與 patching 定位的因果電路吻合。

### MAE 的微妙之處：弱的是「可供 Binding 使用的結構」，非屬性資訊

四骨幹的基底可解碼率排序（DINOv2 > SigLIP ≈ sup-ViT > MAE）與下游 VQA 準確率排序相符，
容易讓人把 MAE 的弱歸因於基底缺屬性資訊。但兩處落差的**尺度**不相稱：基底層面 MAE 與
DINOv2 的差距很小（color 0.920 vs 0.966），下游 VQA 的差距卻很大（0.748 vs 0.924）。
峰值逐物體可解碼率因此**無法**單獨解釋 MAE 的失敗。這與失敗模式分析一致——MAE 的缺陷集中
在雙指涉物綁定（EqAttr），而非屬性編碼本身。故「MAE 基底較弱」必須嚴格表述為**可供
Binding 使用的結構較弱，而非屬性資訊較少**；MAE 保有屬性資訊，只是該資訊較不易被
grounding 的選取步所利用。

## 延伸證據：T2I 模型上的 zero-shot 檢驗為負向（PixArt-Σ）

一個對 A2 的自然質疑是：Binding→Retrieval 的 grounding 或許只是我們在 CLEVR 上做 VQA
訓練所誘發的 artifact，而非語言 conditioning 的通用性質。`docs/t2i_experiment_design.md`
預先登記了一項 zero-shot 檢驗以回應之：取 PixArt-Σ（DiT 骨幹 + 凍結 T5-XXL，僅為文字生圖
目標訓練、從未見過 VQA 監督），以 DIFT-style 單步流程抽取其 28 個 block 的特徵與
cross-attention，檢驗兩項可觀測特徵是否 zero-shot 出現。判讀採**預先登記的雙條件標準**：
(a) referent-local probe 在中／晚層顯著高於 majority baseline，且 (b) column-normalized
的 cross-attention 質量在 referent 3×3 窗上顯著高於 chance（9/1024 ≈ 0.0088）。
Artifact：`outputs/analysis/t2i_pixart/`（t=100，n=300/類）與
`outputs/analysis/t2i_pixart_t{261,400}/`（n=150/類）。

referent-local probe 最佳 block 準確率（括號為 majority baseline）：

| t | color | material | shape | size |
|---|---|---|---|---|
| 100 | 0.169 (0.152) | 0.554 (0.566) | 0.421 (0.361) | 0.556 (0.527) |
| 261 | 0.168 (0.159) | 0.689 (0.571) | 0.542 (0.400) | 0.556 (0.556) |
| 400 | 0.230 (0.159) | 0.594 (0.571) | 0.500 (0.400) | 0.632 (0.556) |

cross-attention referent 窗質量（block 峰值）：t=100 為 0.0095–0.0096、t=261 為
0.0097–0.0099、t=400 為 0.0112–0.0117（三類同在 B6 達峰），僅約 chance 的 1.27–1.33 倍。

**判讀：在所有受測 timestep 上，依預先登記的雙條件標準皆為負向。** probe 對 majority
的超出零散分佈於各 block 與各屬性，無一致的中／晚層結構（t=261/400 時 n=84–133）；CA
定位始終停留在 1.0–1.3× chance 帶內。t=400 三類同在 B6 達峰是唯一有結構的殘跡，但遠低於
任何可用的 binding 訊號。依**預先登記的 domain-mismatch 警語**——CLEVR 問句非 caption，
對 T2I 的文字編碼器構成分佈不匹配——此負向結果**無法區分「機制不存在」與「prompt 分佈
不匹配」**兩種可能。故受此警語約束的有界結論為：**zero-shot binding 未在 PixArt-Σ 的
cross-attention 中、於任何受測 timestep 的問題 prompt 下出現；本文所量測的 grounding
機制需要 task 訓練方能出現。** 這反而正面定位了本文的貢獻——揭示 VQA 訓練如何從預訓練
基底中誘發並強化此機制，而非其自發存在。

（說明：預先登記的設計僅登記單一 timestep t=100 與每類 300 題；實際執行擴展為 t=100/261/400
的 timestep sweep，且 t=261/400 為每類 150 題。此為對預登記設計的擴充，見§限制。）

## 現況說明：本地 Flamingo 基線尚不構成資料點

`docs/paper_v2_outline.md` 的 baseline 規劃中，I2T 的 zero-shot 機制分析（OpenFlamingo
系）本可作為另一條「機制是否為訓練 artifact」的證據。本地訓練的 Flamingo 基線目前
**尚不足以貢獻一個資料點**。經修正一處 harness 量測 bug 後（`generate_answer` 將解碼
小寫化、adapter 卻以大寫 `"Answer:"` 切分，導致原始 JSON 全部記成 prompt echo；無效檔案
依 never-overwrite 政策保留，修正結果為 `*_fixed.json`），修正後的 E7 base 準確率為
color 0.13 / material 0.54 / shape 0.32 / size 0.44——在每個屬性（8/2/3/2 類）上皆為
chance 水準，且輸出近乎退化（color 98% 全答 "yellow"、size 100% 全答 "small"）。預測雖
落在被詢問屬性的答案空間內（LLM 讀得懂問題型別），視覺路徑則尚無貢獻。此 4/16-epoch
checkpoint 確認其僅具定性地位；Flamingo 的 E7 量測等待重訓後方能納入。Artifact：
`add_object_eval_clevr_flamingo_dinov2_early_s42_fixed.json`。

## 限制

1. **單一 seed。** 三項量測均為 s42 單一 seed；size 屬性的小幅移動（如主模型 E7 size
   acc +4）落在 n=100 的雜訊範圍內，不宜過度解讀。

2. **正向結論限於 direct query。** 證據三的穩健性結論針對 direct 屬性查詢；多物體
   場景中殘餘的瓶頸（集合枚舉、雙集合基數比較）不在此三部分論證的裁決範圍，屬失敗模式
   分析（E5）另議。

3. **T2I 負向的不可辨識性。** 如上所述，依預先登記的 domain-mismatch 警語，PixArt-Σ 的
   負向結果無法區分「機制不存在」與「問句—caption 分佈不匹配」。此外，實際執行的
   timestep sweep（t=100/261/400、後兩者每類 150 題）**超出**了設計文件預先登記的範圍
   （單一 t=100、每類 300 題）；此擴充雖不改變負向判讀，但嚴格而言屬預登記之外的探索。
   512×512 resize 對 CLEVR 原生渲染尺寸造成非等比變形，referent 座標映射至 patch 網格
   帶量化誤差（已以 3×3 窗緩解）。

4. **度量選用與預登記的出入。** A1.3 預先登記以 `bait_share_of_errors` 為頭條指標，本
   報告因該指標在錯誤稀有時不穩定，改以 `hallucination_rate` 為主軸指標、`bait_share`
   為形狀確認。此為對預登記字面的偏離，特此聲明（見證據三「度量選用說明」）。

5. **Flamingo 量測缺位。** 本地 Flamingo 基線因 4/16-epoch 訓練不足、預測退化至 chance
   水準，尚不構成資料點；I2T zero-shot 機制證據待重訓後補。

（全節術語遵循命名規範：Grounding 指整個語言 conditioning 機制，其階段依序為 Binding
與 Retrieval，Retrieval 階段的子訊號為 Retrieval (object) 與 Retrieval (answer)；
全節不使用「object grounding」或「Object match」作為階段名稱。）

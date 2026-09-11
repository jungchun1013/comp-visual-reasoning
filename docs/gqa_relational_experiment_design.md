# 實驗設計:CLEVR 上的機制觀察在真實影像(GQA)上的重現

> 提案,2026-09-11(第二版:範圍從 relational 擴成 CLEVR 全部機制觀察,模型加入
> CLEVR 訓練的 SigLIP 做 zero-shot 對照)。狀態:未啟動,等使用者核可。
> 事實來源:`docs/experiment_registry.md` §X19–X22、`JOURNAL.md`、
> `outputs/model/gqa_siglip_decoder1l_scratch_s42.log`、`src/data/gqa.py`、
> `src/model/model.py`、`scripts/analysis/patch_language_condition.py`,以及
> 2026-09-11 對 `val_balanced_questions.json` 的 program 統計。

## 1. 研究問題

CLEVR 上的主張分兩段。Direct query:GCA 把問句寫進 visual stream,referent 在中層被
標出,non-referent 的被問 attribute 從自己的 patch 上被移除,decoder 的 cross-attention
只讀 referent 的 patch;被問 attribute 的 difference-in-means 方向可加性地決定答案。
Relational:先選 anchor,anchor 的 conditioning quantity(same-as 是被共享的 attribute,
spatial 是位置)經 frozen self-attention 送到其他 patch,target 被選出後得到與 direct
query 相同的 referent direction;attribute similarity 落在少數 attention head,position
difference 分散在多個 head 且依賴 positional embedding。

本節回答:**這些觀察在真實影像與真實問句上是否重現;若重現,是因為語言條件化的架構,還是
因為訓練資料。** 主張層級同 CLEVR 的跨 backbone 主張:只主張 stage 的順序、效應的有無與
head 的稀疏或分散,不主張層數與數值。

## 2. 模型(兩個模型,同一個 backbone)

| 模型 | checkpoint | 訓練資料 | 在 GQA 上的角色 |
|---|---|---|---|
| GQA 訓練的模型 | `gqa_siglip_decoder1l_scratch_s42` | GQA balanced train | 在分布內;全套觀察的主要重現 |
| CLEVR 訓練的模型 | `clevr_siglip_decoder1l_scratch_s42` | CLEVR | zero-shot;只答得出答案在 CLEVR 八種顏色內的題 |

兩個模型共用 SigLIP ViT-B/16 @256(grid 16×16,每個 patch 16 px)、GCA 於第 1/3/5/7/9/11
層、1 層 decoder、凍結的 RoBERTa-large 文字端。文字端凍結,所以 CLEVR 訓練的模型能編碼 GQA 的句子;
GCA 的投影只看過 CLEVR 句子,GQA 的名詞對它是分布外。答案為分類(`max_answers 1500`),
CLEVR 訓練的模型的有效答案只有 CLEVR 的 28 個,顏色八種(red、blue、green、yellow、gray、brown、
purple、cyan)。影像處理為直接 resize 到 256×256(`src/model/model.py:209`,無 crop),
scene graph 的 box 以 (x / W · 16, y / H · 16) 對到 patch。

CLEVR 訓練的模型的用途:若 GQA 訓練的模型上的觀察在它身上也出現,機制來自語言條件化架構在凍結 backbone 上
的作用,與訓練影像的領域無關;若它的 accuracy 低於 0.3(chance 0.125)則只報行為,
不做機制分析。CLEVR 訓練的模型不是主線,結果無論方向都報。

GQA 訓練的模型在 GQA balanced val 的表現(最後一次 eval,131,727 題):overall 0.636;
semantic/rel 0.548(61,363);semantic/attr 0.711(42,165);structural/query 0.509
(67,866)。checkpoint 以 val 選出(`best.pt`),行為數字註明此點。

## 3. 資料與篩選

**來源描述(論文用)**:GQA 建立在 Visual Genome 的圖片與 scene graph 之上,圖片來自
COCO 與 YFCC100M;問句由 functional program 自動生成,每題附 program,relate 步驟的
argument 直接帶 target 的 object id(例:`vegetable,to the right of,s (595636)`),
問句名詞與答案也對應到 object id。只有 train 與 val 有 scene graph,故用 **val balanced**
(132,062 題;`val_sceneGraphs.json`)。

**與 Song, Lepori & Pavlick 2026 的差別**:他們人工挑 100 張 COCO 圖並自行合成問句;
本節不手挑,全以規則篩選,每一層的數量寫成 funnel,並隨機抽 50 題人工檢查 scene graph
錯誤率。

### 3.1 母體(2026-09-11 統計,val balanced)

| 題型 | 定義(program) | n |
|---|---|---|
| direct colour / attribute | 無 relate、無 same 的 query;detailed ∈ {directOf, directWhich, material} | 6,481 |
| direct position | 同上,detailed = positionQuery(「哪一側」) | 6,460 |
| spatial,答 category | 恰一個 relate,left / right,無額外 filter,query | 3,619 |
| spatial,答 colour / size / material | 同上,detailed ∈ {directOf, directWhich, how, material} | ≈ 900 |
| spatial,有額外 filter | 同上但 select 後有 filter | 1,623 |
| same-as,open query | relate 的 argument 為 same color,query(「和 Y 同色的 X 是什麼」) | 245 |
| same-as,verify / compare | same / common 類 program,答 yes / no 或 attribute 名 | ≈ 2,200 |

### 3.2 篩選規則(依序套用,每步記數量)

共用:
1. anchor(direct 為 referent)的名稱在 scene graph 中唯一;relational 題 anchor 與
   target 的 category 不同。
2. box 幾何(patch 座標):role 之間 IoU = 0;每個 role ≥ 4 個 patch;任一 role ≤ 30%
   影像。
3. 三個 role 的答案值都在 GQA 訓練的模型的答案詞彙內;CLEVR 訓練的模型另篩答案在八種顏色內。
4. c1 答對(兩個模型各自)。

Direct(對應 CLEVR two-object 的 refer-target / refer-distractor pair):
5. 同一張圖內另有一題 direct query 指向不同物件、問同一 attribute、答案不同;兩題互為
   c1 / c2。找不到配對的題以 scene graph 自動改寫問句(換 referent 名稱,答案取自
   scene graph 的 attribute)補上,記為合成 c2 並分開報。

Spatial:
6. 2D 一致性:box 中心算出的 left / right 與 scene graph 標的 relation 一致;不一致者
   剔除並報比例。沿 relation 軸中心距 ≥ 2 個 patch。
7. c2(換方向詞):anchor 另一側恰有一個與 target 同 category、答案不同的物件,通過
   第 2 條;此物件即 third object。c3(換 anchor):另有一物件對 target 唯一成立同一
   relation、名稱唯一、category 與 anchor / target 皆不同。c2 與 c3 至少一個可構造。
8. 行為分離集:target 落在方向詞相反的影像半邊(對應 X22-H7d),另記子集。

Same-as(open query 245 題):
9. anchor 的 colour 在問句中未出現(問句只說「same color as the Y」,天然滿足);
   場景中恰一個其他物件與 anchor 同色,即 target;另有一個不同色、與 target 同
   category 的物件當 third object(若無則 third object 取任一不同色物件,分開報)。
   c2 = 命名 target 為 anchor。

預估:spatial 篩後數百到一千題,direct 數千題,same-as 一百上下。same-as 若 < 60 題,
只報行為與 head ablation,transplant 曲線標明 n。

### 3.3 Role 與 patch 分組

anchor / referent、target、third object / distractor 各取 box 內 patch。background 分兩類
分開報:(a) 有標注但非 role 的其他物件;(b) 不在任何標注 box 內。CLEVR 的 background
對應 (b);transplant 的 background 組用 (b),regression 兩類都跑。

### 3.4 Forward pass

c0 無問句;c1 原題;c2 依題型(direct:指向 distractor 的配對題;spatial:換方向詞;
same-as:命名 target);c3(spatial):換 anchor。與 CLEVR `relational_*_v2` 定義相同。

## 4. 要重現的觀察與預測(跑前登記)

預測以 CLEVR SigLIP 的結果為準(同 backbone);「通過」指方向與順序一致。標 ◇ 者在
CLEVR 上已知 backbone-specific,列為探索,不設通過條件。

### 4.1 Direct query(GQA 訓練的模型;CLEVR 訓練的模型視 accuracy)

| id | CLEVR 觀察 | GQA 測量 | 預測 |
|---|---|---|---|
| D1 | X21-C referent probe:referent 對 non-referent 的 per-patch probe 從中層起 ≥ 0.96,c0 為 0.50 | 同法,object patch 上 GroupKFold by image | 有起始層,c0 0.50 |
| D2 | X21-A1/A2 selection contrast:refer-target 減 refer-distractor 在 no-question object direction 上的投影,SigLIP 第 5 層起;non-referent 被壓低 | V = c0 的 object patch mean 減 background (b) mean;同一投影 | 起始層存在;non-referent 投影低於 no-question |
| D3 | X21-E attribute directions:own-colour 投影 referent 上升、non-referent 下降(SigLIP 第 7 層起分裂) | colour direction 由 scene graph 標色的物件(獨立影像集)difference-in-means 得到 | 分裂存在 |
| D4 | X21-D1 decoder attention:referent ≫ background ≫ non-referent(SigLIP 113 : 1.8 : 0.1 ×1e-3) | 同法,依 role 分組 | referent 的 per-patch mass 至少 5× background |
| D5 | X21-D2 activation patching:object token 從中層起帶答案,SigLIP 無第 11 層 background 步驟 ◇ | c1 ← c2 逐層換 object / background (b) | object 組有效;background 組是否有效列為探索 |
| D6 | X21-B additive intervention:colour difference-in-means 加到 referent 上翻轉答案,random / background 對照為 0 | 同法,α ∈ {1, 2},逐層 | 翻轉率 ≥ 0.5 於某層;對照 ≤ 0.1 |
| D7 | X21-H head scan:無單一 head 必要;整層 GCA 第 7 / 9 層 ablation 拿掉 selection | 同法,258 個 ablation | 單 head 中位 Δ ≈ 0;至少一整層 GCA 有效 |
| D8 | X21-M2:attention 不是 high-norm artifact | 同法 | 排除 high-norm token 後比值不變 |
| D9 | X21-D3:SigLIP 的 background 在深層帶 question type ◇ | c0 的 background 換入 | 探索 |
| D10 | X21-A5:GCA write norm 無 role 選擇性,cos(write, V) ≤ 0.11 | 同法 | 一致 |

### 4.2 Relational(GQA 訓練的模型;spatial 的 colour 子集另用 CLEVR 訓練的模型跑)

| id | CLEVR 觀察 | GQA 測量 | 預測 |
|---|---|---|---|
| R0 | X22-H7d 行為:答的是 anchor-relative 關係 | 分離集 accuracy 對整體;c2 答案移到 third object 的比例 | 分離集不低於整體 0.1;c2 移動 ≥ 0.6 |
| R1 | X22-H1 / H2c 順序:anchor 先於 target 變 causally necessary | transplant 起始層 | anchor ≤ target |
| R2 | X22-H2a:anchor 未被說出的 conditioning quantity 可從 background decode,早於 target 必要 | spatial:anchor 中心 ridge;same-as:anchor 的 colour 7-way | 起始層 < target 必要層 |
| R3 | X22-H2b:candidate → anchor 的 SA attention c1 − c0 有峰 | `SAAttnCapture` | 峰存在 |
| R4 | X22-H8:same-as 規則選出的 head ablation 掉 ≥ 0.15(SigLIP 0.98 → 0.61);spatial null | 同規則(|Δ| ≥ 10× median,上限 8),隨機對照 | same-as 掉、spatial 不掉;反向亦照報 |
| R5 | X22-H5:anchor 用後未被 suppress;target 的最後 GCA 寫入才必要 ◇(SigLIP 全為 1.00) | `GCAWriteMasker` | 探索 |
| R6 | X22-H7b:翻轉 positional embedding 沿 relation 軸使答案跟著翻;正交軸不變 | `PosEmbedEditor` | 沿軸翻轉錯誤率 ≥ 0.5,正交 ≤ 0.1 |
| R7 | X22-H7c:absolute → relative 的 write field 切換 ◇(只在 DINOv2 s42) | regression 兩類 background | 探索 |
| R8 | X22-H4:target 得到 direct query 的 referent direction | D2 的 V 投影於 target,c1 − c0 | target 投影上升 |

### 4.3 不可移植的觀察(論文明說留在 CLEVR)

- X19-3/4/5、X21-A7/A8 template RSA:需要逐位置相同的 background template 或成對渲染,
  真實影像沒有。
- X19-1/2 additivity 需要同一物件有無 distractor 的成對圖;真實影像只能做同 category
  跨圖的弱版本,列選配。
- X20-C −CA:需要 GQA 的無 GCA 模型;隔壁專案 grounding check 有 SigLIP no-CA 的 GQA
  模型,若可取得只報行為。
- X21-G/J 換被問 attribute:GQA 的 material / size 題數夠,列第二輪。

## 5. 控制組

c0 無問句;transplant 與 patching 的 clean-self 對照每格 1.00;隨機 8 個 disjoint head;
norm-matched random vector;random background rows;ridge 與 probe 用 GroupKFold(5) by
image;R2 另以 c0 的 background 對照;CLEVR 訓練的模型的 chance 為 0.125。

## 6. 實作

全部放在 `scripts/analysis/patch_language_condition.py`,不新開分析腳本:
- `assign_roles` 新增 GQA 路徑(輸入 scene graph box、program 的 object id;幾何規則在
  patch 座標上做),direct 題的配對與合成 c2 也在此。
- 篩選為 CPU 步驟(`--gqa-filter`),輸出 `records.jsonl` 與 funnel 表。
- 既有 flags 沿用:預設 extract、`--attr-directions`、`--intervene`、`--readout`、
  `--head-scan`、`--attn-norm-control`、`--relational-v2` 系列、`--h7-posembed`、
  `--h8-head-ablation`、`--h5-gca-mask`。
- 輸出目錄 `outputs/analysis/patch_language_condition/gqa_{direct,spatial,same}/` 與
  `gqa_{direct,spatial}_clevrmodel/`;舊目錄不動。

## 7. 執行順序與成本(GPU 一次一件,先驗證 CVD=0)

| 步驟 | 內容 | 資源 |
|---|---|---|
| 0 | 三個題型的篩選、funnel、50 題抽查、CLEVR 訓練的模型的顏色子集計數 | CPU,一天 |
| 1 | GQA 訓練的模型:c0–c3 feature cache、行為(R0)、D1/D2/D3/D8/D10 從 cache 算 | GPU 約 1.5 小時 + CPU |
| 2 | GQA 訓練的模型:D5 patching、D6 intervention、R1 transplant | GPU 約 2 小時 |
| 3 | GQA 訓練的模型:D4 decoder attention、R3 SA capture、R2/R7/R8 regression | GPU 30 分鐘 + CPU |
| 4 | GQA 訓練的模型:D7 head scan、R4 head ablation、R5 mask、R6 pos-embed | GPU 約 1.5 小時 |
| 5 | CLEVR 訓練的模型:行為;若 ≥ 0.3,重跑步驟 1–4 的 direct 與 spatial 部分 | GPU 約 3 小時 |
| 6 | 圖與 JSON、JOURNAL、registry X23 | — |

步驟 0 出結果後再排 GPU;n 不足時放寬 3.2 第 7 條,記為新版本目錄。

## 8. 第二輪(視第一輪):RefCOCO+ 圖片、換被問 attribute

本機 `/nfs/turbo/coe-chaijy/jungchun/data/refcocop` 有 19,992 張 COCO train2014 圖與 box
及不含位置詞的 expression。以 GQA 模板措辭合成 spatial 與 direct 題,答案為 COCO
category,篩選同 3.2,mask 從 COCO instances 標注補;先報行為,accuracy ≥ 0.4 才做機制。
另以 GQA 的 material / size 題重跑 D3 / D6(對應 X21-G/J)。

## 9. 風險

- rel/query 子集 accuracy 約 0.5,clean 答對後 n 折半;以 funnel 報告並放寬規則。
- scene graph 漏標使唯一性不成立;抽查錯誤率 > 10% 則加人工複核(只複核不挑題)。
- 真實影像的 role 常很大或互相遮擋;IoU = 0 與 ≤ 30% 規則處理,另報 role 大小分層。
- direct 的 c2 配對題不一定存在;合成 c2 的問句措辭與 GQA 生成句不同,分開報。
- CLEVR 訓練的模型在分布外,可能連行為都不成立;此時結論是「機制需要在該領域訓練」,照報。
- 只有一個真實影像 backbone;跨 backbone 主張留在 CLEVR,必要時再訓 DINOv2 GQA
  (每 epoch 約 2 小時,18 epoch 約 36 小時)。

## 10. 記錄

registry 新增 X23(本表);JOURNAL 在步驟 0 與每個 GPU 步驟後追加;結果先給使用者,網站與
RESULTS 待使用者決定。

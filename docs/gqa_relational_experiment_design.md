# 實驗設計:relational selection 在真實影像上的驗證(GQA)

> 提案,2026-09-11。目的:證明 CLEVR 上找到的 relational selection 機制在真實影像與
> 真實問句上仍成立。狀態:未啟動,等使用者核可。
> 事實來源:`JOURNAL.md`(X22 各條)、`docs/experiment_registry.md` §X22、
> `outputs/model/gqa_siglip_decoder1l_scratch_s42.log`、`src/data/gqa.py`、
> `scripts/analysis/patch_language_condition.py`。

## 1. 研究問題

CLEVR 上的主張:一個 relational 問題是「以另一個被選物件的性質為條件的選取」。GCA
先標出 anchor,anchor 的 conditioning quantity(same-as 是被共享的 attribute,spatial
是位置)經 frozen self-attention 送到其他 patch,target 被選出後得到與 direct query
相同的 referent direction,答案在最後一個 GCA 層 copy 到 background patch 供 decoder
讀出。兩類問題差在條件的計算:attribute similarity 由少數 attention head 完成;position
difference 需要先把 anchor 座標 copy 到 background,再由較深的 GCA 算出 anchor-relative
field,分散在多個 head。

本節要回答的問題只有一個:**同一套 signature 在真實影像上是否重現。** 主張層級同
CLEVR 的跨 backbone 主張:只主張 stage 的順序與 head 的稀疏或分散,不主張層數或數值相同。

## 2. 要重現的三個 signature(CLEVR 參考值)

| signature | CLEVR DINOv2 s42 | CLEVR SigLIP s42 |
|---|---|---|
| S1 順序:anchor 的 patch 先於 target 的 patch 變成 causally necessary | same-as anchor 第 3 層起、target 第 3 層起(最低點第 7 層);spatial 無物件在第 9 層前必要,第 9 層 target 必要 | same-as anchor 第 3 層起且到第 11 層仍必要;target 第 3 到 10 層 |
| S2 conditioning quantity 在 target 必要之前可從 background decode | spatial:anchor 質心 ridge R² 0.57 於第 7 層,早於 target 必要的第 9 層 | anchor 的 shared shape 從 background decode 第 5 層起 |
| S3 head 的稀疏或分散 | same-as 規則選出的 8 個 head ablation 1.00 → 0.77;spatial 為 null(1.00,與隨機 head 相同) | same-as 0.98 → 0.61(head 在第 5 到 6 層);spatial null |

Spatial 另有 S4:GCA write field 從 absolute coordinate(第 1 層 R² 0.61)切換到
anchor-relative(第 9 層 R² 0.66)。S4 只在 DINOv2 清楚,SigLIP 上沒有看到,所以在
GQA(SigLIP)上列為探索項,不列入通過條件。

## 3. 模型

`gqa_siglip_decoder1l_scratch_s42`:凍結 SigLIP ViT-B/16 @256(grid 16×16,每個
patch 16 px),GCA,1 層 decoder,答案為 top-1500 答案詞彙上的分類(`GQAClsDataset`)。
影像處理是直接 resize 成 256×256(`src/model/model.py:209`,無 crop),所以 scene
graph 的 box 以 (x / W · 16, y / H · 16) 線性對到 patch 格。

GQA balanced val 上的表現(最後一次 eval,131,727 題):

| 子集 | accuracy | n |
|---|---|---|
| overall | 0.636 | 131,727 |
| semantic/rel | 0.548 | 61,363 |
| semantic/attr | 0.711 | 42,165 |
| structural/query | 0.509 | 67,866 |

checkpoint 是以 val 選的(`best.pt`),行為數字註明此點;機制分析不受影響。

## 4. 資料與篩選

**來源描述(論文用)**:GQA 建立在 Visual Genome 的圖片與 scene graph 之上,圖片來自
COCO 與 YFCC100M;GQA 以 functional program 自動生成問句,每題附 program 與問句名詞對應
的 object id。只有 train 與 val 有 scene graph,故本節用 **val balanced**
(`val_balanced_questions.json`,113 MB;`val_sceneGraphs.json`)。

**與 Song, Lepori & Pavlick 2026 的差別**:他們人工挑 100 張 COCO 圖並自行合成問句;
本節不手挑,全部以規則篩選,每一層的數量寫成 funnel 報告,並隨機抽 50 題人工檢查
scene graph 錯誤率。

### 4.1 Spatial 篩選規則(依序套用,每步記數量)

1. program 恰為 select(anchor) → relate(target, rel) → query(name 或 color),無額外
   filter、and / or;structural 為 query。
2. rel ∈ {to the left of, to the right of};第二輪視 n 再加 above / below。
3. anchor 與 target 的 category 不同;anchor 的名稱在 scene graph 中唯一。
4. box 幾何(resize 後的 patch 座標):anchor、target 兩兩 IoU = 0;每個 role ≥ 4 個
   patch;任一 role ≤ 30% 影像;沿 relation 軸的中心距離 ≥ 2 個 patch
   (CLEVR 用 `SPATIAL_MARGIN_FRAC = 2/24`,這裡固定 2 個 patch)。
5. 2D 一致性:以 box 中心算出的 left / right 與 scene graph 標的 relation 一致;不一致
   者剔除並報比例(對應 CLEVR 3D 相機軸與 2D 質心的限制)。
6. counterfactual 可構造:
   - c2(換方向詞):anchor 另一側存在恰一個與 target 同 category、答案值不同的物件,
     且該物件通過第 4 條幾何規則;此物件即 third object。
   - c3(換 anchor):存在另一個物件,對 target 也唯一成立同一 relation,名稱在 scene
     graph 中唯一且與 anchor、target category 皆不同。
   - c2 與 c3 至少一個可構造才保留;兩者皆可構造的題另記一個子集。
7. 三個 role 的答案值都在模型的 1,500 個答案詞彙內。
8. clean forward pass(c1)答對。

預估:第 1 到 2 條後約數千題;加上第 6 條後數百到一千;第 8 條後再折半。低於 200 題
時放寬第 6 條為只要求 c3。

### 4.2 Same-as 篩選

GQA 的 same 類 program(`same color / same material / same shape` 等)多為 verify 或
choose 題型,開放式 query 可能極少。第一步先數:val balanced 中 program 含 same 且
structural 為 query 的題數。若 ≥ 200 題,規則同 4.1 第 3 到 8 條,c2 改為「命名 target
為 anchor」(答案變為原 anchor 的值,同 CLEVR)。若 < 200 題,same-as 在 GQA 只報
behaviour(以 choose 題的兩個候選 logit 讀),機制分析留在 CLEVR,論文如實說明。

### 4.3 Role 與 patch 分組

- anchor / target / third object:各自 box 內的 patch。
- background 分兩類分開報:(a) scene graph 有標注但非 role 的其他物件的 patch;
  (b) 不在任何標注 box 內的 patch。CLEVR 的「background」對應 (b);真實影像的 (a)
  是新情況,S2 與 S4 的 regression 兩類都跑。
- 三個 role 之外的所有 patch 合稱 non-role,transplant 的 background 組用 (b)。

### 4.4 Forward pass

c0 無問句;c1 原題;c2 換方向詞(spatial)或命名 target(same-as);c3 換 anchor
(spatial)。與 CLEVR 的 `relational_{same,spatial}_v2` 定義相同。

## 5. 假設與預測(跑前登記)

| # | 假設 | 通過 | 反證 |
|---|---|---|---|
| R0 行為 | 模型在篩選集上答的是 anchor-relative 關係 | c1 accuracy 與 rel/query 子集相當(約 0.5);clean 答對題中 c2 的答案移到 third object 的比例 ≥ 0.6;c3 同理 | c2 答案不動(模型答絕對半邊或 prior) |
| R1 順序 = S1 | anchor 的 patch 先於 target 變 causally necessary | transplant 起始層 anchor ≤ target | target 先於 anchor,或兩者皆不必要 |
| R2 decodable = S2 | anchor 座標可從 background (b) 用 ridge 讀出,起始層 < target 必要層 | 起始層(c1 − c0,R² ≥ 0.3)小於 R1 的 target 起始層 | 起始層 ≥ target 起始層,或 R² 全程 < 0.3 |
| R3 head = S3 | spatial 依固定規則(|Δ| ≥ 10× median,上限 8 個)選出的 head ablation 與隨機 head 無差 | accuracy 差 < 0.05 | 選出的 head 使 accuracy 掉 ≥ 0.15 而隨機不掉(spatial 在真實影像上變稀疏,亦為結果,照報) |
| R4 探索 | write field 由 absolute 切換到 relative | 報 R² 曲線,不設通過條件 | — |

Same-as 若有足夠題數:R1、R3 同表,S3 的預測改為稀疏(ablation 掉 ≥ 0.15)。

## 6. 控制組(與 CLEVR 相同)

c0 無問句;transplant 的 clean-self 對照每格 1.00;隨機 8 個 disjoint head;
ridge regression 用 GroupKFold(5) by image;R2 另以 c0 的 background 做對照。

## 7. 實作

全部放在 `scripts/analysis/patch_language_condition.py`,加一個 GQA 的資料介面,不新開
分析腳本:
- `assign_roles` 新增 GQA 路徑:輸入 scene graph box 與 program 的 object id,輸出與
  CLEVR 相同的 role dict;幾何規則在 patch 座標上做。
- 篩選為獨立的 CPU 步驟(`--gqa-filter`),輸出 `records.jsonl` 與 funnel 表,不碰 GPU。
- 其餘(feature cache、transplant、write-position regression、head ablation)沿用既有
  flags,輸出目錄 `outputs/analysis/patch_language_condition/gqa_spatial/`
  (與 `gqa_same/`),舊目錄不動。

## 8. 執行順序與成本(GPU 一次一件,先驗證 CVD=0)

| 步驟 | 內容 | 資源 |
|---|---|---|
| 0 | 篩選 + funnel + same 題數統計 + 50 題人工抽查 | CPU,半天 |
| 1 | c0–c3 feature cache 與 R0 行為 | GPU,約 1 小時(千題級) |
| 2 | R1 transplant(anchor / target / third / background,每層) | GPU,約 1 小時 |
| 3 | R2、R4 regression | CPU |
| 4 | R3 head ablation(選 head 用 c1 − c0 的 SA attention 到 anchor) | GPU,約 30 分鐘 |
| 5 | 圖與 JSON,JOURNAL 條目,registry X23 | — |

步驟 0 出結果後再排 GPU;n < 200 時回到 4.1 放寬規則,記為新版本目錄。

## 9. 第二輪(視第一輪結果):RefCOCO+ 圖片

本機 `/nfs/turbo/coe-chaijy/jungchun/data/refcocop` 有 19,992 張 COCO train2014 圖與
box、RefCOCO+ expression(不含位置詞)。若第一輪 R0 到 R3 通過,以 GQA 模板措辭合成
「What is the [category] to the left of the [expression]?」,答案為 COCO category
(需對到 1,500 答案詞彙),篩選規則同 4.1 第 4 到 7 條,mask 從 COCO instances 標注補。
先報 behaviour,accuracy ≥ 0.4 才做 transplant。這一輪的功能是跨 dataset 的第二次驗證,
與 Song 的 COCO 測試對齊。

## 10. 風險

- rel/query 子集 accuracy 只有約 0.5,clean 答對後 n 可能不足:以 funnel 報告,並在
  4.1 第 6 條放寬。
- scene graph 漏標使「唯一」不成立:50 題抽查報錯誤率;錯誤率 > 10% 時加人工複核
  (只複核,不挑題)。
- 真實影像的 anchor 常很大(佔多 patch):S1 的 transplant 換掉整個 box 可能同時換掉
  部分 target 資訊;IoU = 0 規則處理重疊,另報 anchor 大小分層的結果。
- SigLIP 在 CLEVR 上第 7 層就完成 read-out,GQA 上可能更淺:R1 到 R3 都是逐層報,不預設層數。
- 只有一個真實影像 backbone:跨 backbone 主張留在 CLEVR;若審稿需要,再訓
  DINOv2 GQA 模型(每 epoch 約 2 小時,18 epoch 約 36 小時)。

## 11. 記錄

registry 新增 X23(本表的假設與預測);JOURNAL 在步驟 0 與每個 GPU 步驟後追加;結果先給
使用者,網站與 RESULTS 待使用者決定。

# Proposal:補強 selection → removal 機制主張的後續實驗(2026-09-13)

## Context

現有證據(本 session 逐項核過):CLEVR 三個 backbone 上,referent selection 在 block 5 啟動;
語言條件化對被問屬性的放大是**非選擇性**的(DINOv2 b8:refer-it +12.79、refer-other +11.97、
refer-neither +14.90),真正帶指涉資訊的是 block 9 起對 non-referent 的**移除**(+4.10 → −3.42,
低於無問題基線);SigLIP 在 b7 分開、幅度更大;MAE 幾乎沒有這套機制。GQA 上同樣形狀晚兩個
block(b7)出現,但 GQA 目前只有相關性證據。

三個結構性缺口:(一)屬性層的數字沒有 gain 控制、own-vs-other 是恆等式,目前不能用;
(二)selection 與 removal 只有**順序**,沒有**依賴**證明;(三)GQA 沒有因果證據、沒有屬性分解。
本 proposal 先修(一),再用現成 hook 補(二)與(三)。compositionality 的 factorial 明寫為
future work,不排 GPU。

模型:`clevr_dinov2_decoder1l_scratch_s42`(主)、`clevr_siglip_decoder1l_scratch_s42`(複製)、
`gqa_siglip_decoder1l_scratch_s42`(GQA 唯一模型)。所有程式改動都在
`scripts/analysis/patch_language_condition.py`,hook 來自 `src/analysis/patching_utils.py`。
結果一律寫入新目錄,舊目錄唯讀。

---

## A. 修補(純 CPU,`OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=`,可與訓練並行)

### A1. 屬性投影的 gain 控制
- **問題**:`attr_direction_analysis` (:659) 在 :670–676 直接對 `obj_mean` 做內積,沒有
  normalisation 參數;若 h → αh,所有投影同乘 α。整體物件分析有 `norm_std` 版
  (`part_a` :418,`partA_metrics_normstd.json` 由 :5349 迴圈寫出)且結論撐得住(b9
  +0.36/−0.36),屬性分析沒有。存的 `offset_norm` 屬於 offset 向量,不能事後補算。
- **改動**:`attr_direction_analysis(caches_n2, labels_n2, V, norm_std=False)`;`norm_std=True`
  時 `om = _unit(om)`(沿用 `token_table` :407 同一個 `_unit`)。`--attr-directions` 分派
  (:5327–5341)改成與 :5349 相同的 `(("", False), ("_normstd", True))` 迴圈,輸出
  `partA_attr_directions_normstd.json` / `attr_directions_normstd.png`。另外把每條件
  `obj_mean` 的 norm 逐 block 存進 JSON(`gain` 欄),讓 gain 本身可見。
- **輸入**:現有 cache `n1/`、`n2/`(DINOv2)、`siglip/n{1,2}`、`mae/n{1,2}`、`shape/n{1,2}`
  (`feats_c{0..3}.npz` + `labels.json`)。
- **判讀**:normalised 之後 refer-it / refer-other / refer-neither 在 b≤8 仍無差、b≥9
  non-referent 仍低於基線 → 「非選擇性放大 + 選擇性移除」成立且不是 gain 假象;
  若 normalised 後移除消失 → 移除只是 norm 變化,第二節結論撤回。

### A2. 換掉恆等式對照
- **問題**:二值屬性(material、size)的 `V_other = −V_own`,own-vs-other 必為 −1.000
  (shape 三值為 −0.42 至 −0.50,同理受限)。
- **改動**:保留 `*_other` 欄但在 JSON 加 `"deprecated": "algebraic identity for k-valued attribute"`;
  新增兩個統計量(都有 per-image bootstrap CI,沿用 `_boot`):
  1. `own_vs_c0`:own-value 投影相對無問題基線(`refvs0_*`、`nonrefvs0_*` 已存在,正式採用);
  2. `queried_vs_unqueried`:同一物件、同一條件下,被問屬性 own-value 的 Δ(c1−c0)減去
     未被問屬性 own-value 的 Δ。這才是「attribute-specific」的可檢驗定義。
- 與 A1 同一次跑完。

### A3. 指涉詞分層
- `docs/experiment_registry.md:508–511` 的 n=223/59/42 分層在程式裡不存在(`grep -c` = 0)。
- **改動**:若 `labels.json` 帶指涉詞屬性欄位(先查;若沒有,從 `labels_n2` 的問句模板反推
  shape/size/material 詞),在 A1 的函式裡加 `stratify_by_ref_word`,輸出三層各自的
  series 與 n。若無法重建 → 在 registry 該段加「撤回:未實作」註記,不改原文。

### A4. 三個登記缺失(各一小段程式)
- trained pooled probe 的 fold-local PCA(現為全體 PCA 後再 fold)。
- GQA 結果檔寫入登記統計量(`x23_results.json` 現在只有中間量)。
- spatial marker 與 marker 來源重疊的 2 張影像(ids 2400661、2407031):**建議 cross-fit**
  (marker 在不含該影像的子集估),不刪樣本。這三項是長期待決事項,列在此等你定。

A 合計:程式改動 < 100 行,CPU 數小時,無 GPU。

---

## B1. 依賴實驗:阻斷 selection,看 removal 是否消失(GPU,CLEVR)

### 假設
- **H-dep**:block 9 起對 non-referent 的屬性移除,依賴 block ≤5 建立的 selection。
- **預測(通過)**:阻斷 selection 後,b9–11 的 `nonrefvs0_*_own`(normalised)從負值回到
  ≈0(幅度縮減 ≥ 50%,CI 不含未阻斷值);同時 b5–8 的 selection contrast(`delta.ref_imgdir`
  :446)消失(sanity)。隨機對照不變。
- **反證(平行路)**:selection contrast 消失、但 b9–11 的移除仍在未阻斷值的 CI 內 →
  removal 不依賴早期 selection,故事改寫為兩條平行路徑。
- **中間結果**:部分縮減 → 報告 dose-response,不下二元結論。

### 介入(兩種,互為補充)
1. **GCA 寫入阻斷**(`GCAWriteMasker(trunk, layer, mask)` patching_utils.py:374):
   - 劑量序列:遮 GCA 層 {1}、{1,3}、{1,3,5}、{1,3,5,7};mask = 全部 patch。
   - 角色拆分(核心):只遮 {1,3,5} 在 **target 的 patch** vs 只遮在 **distractor 的 patch**
     vs 只遮背景(以物件身分定義 mask,不以指涉狀態定義,因為同一物件在 c1 是 referent、
     在 c2 是 non-referent,而量測的對象始終是 target)。回答「晚期移除 non-referent
     (c2 下的 target)靶的是早期對它自己的寫入,還是對另一個物件的寫入」。
   - 對照:遮 {9,11}(晚期,應直接消掉 removal,但不動 selection)。
2. **marker 方向投影掉**(`SubspaceProjector(trunk, layer, basis, mask)` :346):
   - basis = 該 block 的 marker 方向(`run_marker_test` :1074 同法:n2 c1−c2 target
     `raw_obj_mean` 均值,unit;**以 image 二折 cross-fit 估**,避免用同一批影像估又測)。
   - 施於 block 7 或 8(onset 與 removal 之間),mask = 全部 patch。
   - 對照:同 norm 隨機單位向量(5 個 seed 取均值);同一 basis 施於 block 10(時序對照)。

### 統計量與樣本
- 與 A1/A2 相同的 normalised 屬性 series、selection contrast、`offset_norm`;加行為層:
  accuracy、P(答案 = non-referent 的屬性值)、logit margin。
- n2 的 324 張影像 × c0–c3;per-image bootstrap(1000)CI。
- DINOv2 主;SigLIP 重跑相同格子(它在 b7 就分開,預測 onset 前移但依賴方向相同);
  MAE 只跑 baseline 與 {1,3,5} 一格,作為「沒有機制就沒有東西可阻斷」的負對照。

### 實作
- `extract_condition_sparse` (:310) 目前不接 hook;加 `hooks: list[ContextManager]` 參數,
  在 forward 前 `ExitStack` 進入。新增 CLI `--intervene gca-mask:1,3,5[:target|distractor|bg]`
  與 `--intervene project:7:marker|random[:seed]`,輸出到 `n2_int_<spec>/feats_c*.npz`,
  之後 A1/A2 的分析原樣重跑(只換 `--cache-dir`)。
- 成本:每格一次 324×4 條件抽取 ≈ 5 分鐘;DINOv2 約 14 格、SigLIP 14 格、MAE 2 格
  → 約 2.5 小時 GPU,分批。磁碟:每格 ≈ 2.7 GB(同 n2),30 格 ≈ 80 GB;若不可接受,
  抽取時只存 `obj_mean/raw_obj_mean/bg_mean/raw_norm`(去掉 `tok`),每格 < 10 MB。

---

## B2. GQA 第一個因果結果:marker 注入(GPU,小)

### 假設
- **H-inj**:GQA 上估出的 selection marker 是**因果有效**的角色訊號,而不只是與指涉相關的方向。
- **預測(通過)**:在 c1(問 T)把 marker 加到 D 的 patch → P(答案 = D 的屬性值)上升、
  accuracy 下降,CI 不含 0 且超過隨機對照;從 T 的 patch 減掉 marker → accuracy 下降。
- **反證**:效應與同 norm 隨機方向無差。

### 設計
- 記錄:`x23_gqa_direct` 的 184 題 query-attr;篩 D 的被問屬性值在答案詞表且 ≠ T 的值
  (篩後 n 回報)。
- marker:`gqa_h4_marker` :2723 同法(c1−c2 `raw_obj_mean[:,0]` 逐 block 均值),
  **二折 cross-fit by image**。
- hook:`ResidualAdder(trunk, layer, delta, mask, alpha)` :786,已存在,不寫新 hook。
  layer ∈ {7, 9}(GQA onset 與其後);alpha ∈ {1, 2}(沿用 `--marker-alphas` 預設)。
- 對照:同 norm 隨機方向 ×5 seed;加到背景 patch;alpha=0 自我對照。
- 統計:ΔP(D 值)、Δaccuracy、Δmargin,image bootstrap CI,paired 對隨機對照。
- 成本:184 題 × (2 layer × 2 alpha × 3 目標 + 對照) ≈ 數千次 forward,< 20 分鐘。
- 實作:`run_gqa_causal` :3280 加分支 `--gqa-inject`,輸出 `x23_gqa_direct_inject/`。

---

## B3. GQA 屬性分解(GPU 一次抽取 + CPU)

### 主測試:Song 式屬性方向,真實影像版
- **物件池**:GQA val scene graph 中帶 colour 屬性、覆蓋 ≥ 4 patch、**不在 184 題的影像**中
  的物件;目標 ≥ 30 個 / 每個 colour 值、≥ 8 個值 → 約 800 張影像、c0(無問題)一次抽取
  (只存 `obj_mean/raw_obj_mean`,不存 `tok`)。輸出 `gqa_attr_pool/`。
- 方向:`attribute_directions` 同法(自身值均值 − 池總均值,unit),colour 為主;material
  若值數 ≥ 3 且每值 ≥ 15 則一起做,否則只報 colour。
- 投影:184 題的 T 與 D,c0/c1/c2,`obj_mean` normalised(A1 版);統計量同 A2
  (`own_vs_c0`、`queried_vs_unqueried`),blocks 7–11。
- **預測**:若 CLEVR 的圖像在真實影像上重現,放大在 b≤8 非選擇性、b≥9 non-referent
  低於 c0。若只有放大沒有移除 → 寫為「真實影像上只重現 selection,不重現 removal」,
  這本身是可報告的結果。

### 次測試:同一物件、只換被問屬性(本 session 核過可行性)
- 184 題中有 36 題的 T 在題庫 `gqa_meta/question_pool_val_all_refword2.json` 另有
  plain select→query 的不同屬性問句;扣掉 18 題問影像左右位置後約 30 題,其中
  colour↔material 19 題。
- 設計:同一物件、同一影像,問 colour vs 問 material,比較它在 colour own-value 方向上的
  投影差(paired,image bootstrap)。這是把物件完全控制住的 attribute-specificity 檢驗。
- n 小,**只作次要證據**,報 CI 不報結論性語句。需要一次 30 題 × 2 問句的抽取(分鐘級)。

### 這一項對 Song 的區別
他們的自然影像測試只有類別、無屬性標註(自述限制);GQA scene graph 有。這是可檢查的
差異,不是宣稱。

---

## C. 範圍與穩健性(只排一項)

- **C1. GQA 模型 dev-split 重訓**:`configs/experiment/gqa_siglip_decoder1l_scratch.yaml`
  與 `configs/data/gqa.yaml` 都沒有 split 選項,split 寫死在資料程式裡;需先加
  `data.dev_fraction`(從 train 切出,固定 seed),新 yaml `gqa_siglip_decoder1l_scratch_dev`
  並設 `wandb.name`。約 36 小時 GPU,**等 learned-text 訓練結束後**才排;是否做由你定。
- **不做**:spatial 擴族群(35 題 / 27 張卡住三個測試,但主線不靠 GQA spatial;論文寫
  「不可判定」);compositional factorial(寫為 future work);CLEVR SigLIP 累積消融曲線
  (非必要)。

---

## 執行順序與 GPU 排程(一次一件 GPU;先 torch alloc 驗 CVD=0)

| 階段 | 內容 | 資源 | 前置 |
|---|---|---|---|
| 0 | 把本文假設表與預測寫入 `docs/experiment_registry.md` 新節,commit | — | 第一個 GPU 作業前 |
| A | A1–A4 | CPU,數小時 | 現在即可,與訓練並行 |
| B1 | 依賴實驗(DINOv2 → SigLIP → MAE 負對照) | GPU ≈ 2.5 h | A1 完成(分析函式要先有 norm 版) |
| B2 | GQA marker 注入 | GPU < 20 min | 無 |
| B3 | GQA 屬性池抽取 + 分析 | GPU ≈ 15 min + CPU | A1 |
| C1 | dev-split 重訓 | GPU ≈ 36 h | learned-text 訓練結束;你決定 |

## 登記規則(寫進 registry)
- 預測、反證、對照在跑前 commit;所有格子報告,含 n 與 image-bootstrap CI。
- 對照固定:alpha=0 / 未遮 自我對照、同 norm 隨機方向、同秩隨機子空間、時序對照(晚層)。
- marker 一律 cross-fit by image;不得用估 marker 的影像測 marker。
- 主張分兩級:A/B3 主測試是相關;B1/B2 是因果。GQA 因果僅限 B2 的範圍。
- 不改假設配合結果;改設計 → 新目錄、留舊目錄。

## 驗證
1. A1 的 `norm_std=False` 輸出與現有 `partA_attr_directions.json` 逐值一致(回歸)。
2. B1 的無介入格(`--intervene none`)重現 n2 既有 series;alpha=0 / 空 mask 自我對照每格一致。
3. 隨機方向、隨機子空間、晚層時序對照落在 baseline bootstrap 誤差內。
4. B2 cross-fit marker 在兩折上都重現 marker 投影 T − D > 0(與既有結果同向)。
5. 每個假設一張圖 + 一個 JSON,標題以 backbone 開頭;plot 走 `analysis.plot_style`。

## 產出
- 本文複製為 `docs/mechanism_followup_design.md`(與 `docs/gqa_relational_experiment_design.md`
  同格式),registry 新節,JOURNAL 各階段一條。
- 結果先給你看;網站與 RESULTS 由你決定。
- 另一件待你定:把今天對 Codex 文件的九點補充另署名寫進
  `DISCUSSION_FRAMING_CODEX_2026-09-12.md`(不動原文與簽名)。

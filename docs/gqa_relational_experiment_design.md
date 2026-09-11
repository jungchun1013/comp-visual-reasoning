# 實驗設計 X23:CLEVR 上的機制觀察是否在真實影像(GQA)上存活

> 提案第三版,2026-09-11。狀態:未啟動,等使用者核可。
> 第三版依使用者十點修改:分析母體分層、checkpoint 與 val 的關係、統計判準、null 的
> 解讀、spatial ground truth 的定義、synthetic c2 的地位、CLEVR 訓練模型的解讀、probe
> leakage 控制、human audit、primary 假設縮減為四條。
> 事實來源:`docs/experiment_registry.md` §X19–X22、`JOURNAL.md`、
> `outputs/model/gqa_siglip_decoder1l_scratch_s42.log`、`src/data/gqa.py`、
> `src/model/model.py`、`scripts/analysis/patch_language_condition.py`、2026-09-11 對
> `val_balanced_questions.json` 的 program 統計。

## 1. 研究問題與設計核心

CLEVR 上的發現依賴兩件真實影像沒有的東西:scene graph 讓每個 predicate 可以精確求值,
成對渲染讓 counterfactual 只差一個變數。原 workshop paper 已把這列為最大限制。X23 的
問題因此不是「把 CLEVR 實驗搬到 GQA」,而是:**在 controlled domain 找到的機制發現,
在自然場景的變異、標注的模糊、與問句的自然分布下,哪些存活、哪些不存活。**

設計核心是三個嚴謹度風險,不是實作細節:
- **sample selection**:分析母體不得以模型是否答對來定義(§3.1)。
- **confirmatory / exploratory 邊界**:四條 primary 假設在跑前寫死統計判準(§4),其餘
  一律標 secondary 或 exploratory(§5),不得事後升級。
- **operationalization 的差異**:CLEVR 的每個量在 GQA 上的定義都重寫一次(§4 各條的
  「GQA 定義」欄),文字上區分 observational 與 causal,不沿用 CLEVR 的機制語言描述
  GQA 上尚未做因果檢驗的量。

主張層級:只主張 stage 順序、效應方向與 head 定位的有無,不主張層數與數值。

## 2. 模型

| 模型 | checkpoint | 訓練資料 | 在 X23 的地位 |
|---|---|---|---|
| GQA 訓練的模型 | `gqa_siglip_decoder1l_scratch_s42` | GQA balanced train | primary:H1–H4 的 confirmatory test |
| CLEVR 訓練的模型 | `clevr_siglip_decoder1l_scratch_s42` | CLEVR | secondary:transfer test,只能答顏色在 CLEVR 八種內的題 |

共用 SigLIP ViT-B/16 @256(grid 16×16)、GCA 於第 1/3/5/7/9/11 層、1 層 decoder、凍結的
RoBERTa-large 文字端、分類式答案頭(`max_answers 1500`;CLEVR 訓練的模型有效答案 28
個)。影像直接 resize 到 256×256(`src/model/model.py:209`,無 crop),box 以
(x / W · 16, y / H · 16) 對到 patch。

**Checkpoint 與 val 的關係(聲明)**:GQA 訓練的模型的 `best.pt` 是 2026-06 依 balanced
val 的 aggregate accuracy 選出的;X23 的任何機制測量都未參與 checkpoint 選擇。X23 的題目
是 held-out items,但 **不是 checkpoint-independent**,論文用這個措辭,不稱 independent
replication。乾淨版本是重訓一次:GQA train 切出一小塊 development split 選 checkpoint,
balanced val 完全凍結給 X23。成本一次訓練(每 epoch 約 2 小時,18 epoch 約 36 小時,
佔唯一 GPU)。是否重訓由使用者決定;不重訓則以上聲明進論文。

**CLEVR 訓練的模型的解讀上限**:兩個模型不只差訓練影像,還差問句分布、答案詞彙、視覺
統計、關係頻率與優化軌跡。若機制在它身上重現,只能寫「the mechanism transfers across
substantial visual and linguistic distribution shift without GQA-specific training」,
不能寫「由架構而非資料導致」。

GQA 訓練的模型在 balanced val 的表現(最後一次 eval,131,727 題):overall 0.636;
semantic/rel 0.548(61,363);semantic/attr 0.711(42,165);structural/query 0.509(67,866)。

## 3. 資料、母體與篩選

**來源描述(論文用)**:GQA 建立在 Visual Genome 的圖片與 scene graph 之上,圖片來自
COCO 與 YFCC100M;問句由 functional program 生成,relate 步驟的 argument 帶 target 的
object id,問句名詞與答案對應到 object id。只有 train 與 val 有 scene graph,故用 val
balanced(132,062 題)。

母體統計(2026-09-11):direct colour / attribute query 6,481;direct position 6,460;
spatial left / right 單一 relate 無額外 filter:答 category 3,619、答 attribute ≈ 900;
有額外 filter 1,623;same-as open query(relate argument 為 same color)245;
same-as verify / compare ≈ 2,200。

### 3.1 兩層母體

- **S_eligible**:只由 scene graph、program 與幾何定義的樣本(§3.2 的規則,不含任何
  模型輸出)。所有 representational 分析(probe、attention、regression、projection)
  的 primary 母體。
- **S_correct** = S_eligible ∩ {c1 答對}。只有以 clean answer 為 causal endpoint 的
  intervention(transplant、patching、head ablation、pos-embed 翻轉、additive
  intervention)用它。
- 每個結果同時報 n_eligible、n_correct、coverage = n_correct / n_eligible。兩個模型的
  S_eligible 相同(CLEVR 訓練的模型另加答案在八種顏色內的規則,形成子集,分開報);
  S_correct 各自不同,論文明說兩者的 intervention 母體不同。

### 3.2 S_eligible 的規則(依序套用,每步記數量與排除原因)

共用:
1. anchor(direct 為 referent)的名稱在 scene graph 中唯一;relational 題 anchor 與
   target 的 category 不同。
2. box 幾何(patch 座標):role 之間 IoU = 0;每個 role ≥ 4 個 patch;任一 role ≤ 30%
   影像。
3. 三個 role 的答案值都在 GQA 訓練的模型的答案詞彙內。

Direct:
4. **natural pair**:同一張圖內另有一題 GQA 原生的 direct query,指向不同物件、問同一
   attribute、答案不同;兩題互為 c1 / c2。這是 primary 母體。
5. **synthetic c2**(secondary,不與 natural pool):找不到配對時以 scene graph 改寫
   referent 名稱、答案取自 scene graph。附 linguistic control:synthetic 句與 GQA 原句
   的 RoBERTa embedding 距離分布、c1 → natural c2 與 c1 → synthetic c2 分開報、
   human audit 檢查 grammaticality 與 semantic validity(§7)。

Spatial:
6. relation 只取 to the left of / to the right of。front / behind 明確排除:它們牽涉
   深度、大小、遮擋與透視,不是同一種 reference-frame 計算,不與 left / right 合併。
7. **2D image-plane criterion**:box 中心算出的 left / right 與 scene graph 標注一致;
   不一致者排除並報比例。論文措辭:「we restrict analysis to GQA relations whose
   scene-graph annotation agrees with a preregistered 2-D image-plane criterion」,
   不寫「GQA spatial relations are …」。
8. 沿 relation 軸中心距 ≥ 2 個 patch。理由:CLEVR 用 2/24 grid 的比例,16×16 grid
   下對應 1.3 個 patch,取整為 2 保證兩個 role 之間至少一個非 role patch。
   robustness:1 / 2 / 3 個 patch 三個版本,appendix 報 H2、H3 的效應是否穩定。
9. counterfactual:c2(換方向詞)要求 anchor 另一側恰有一個與 target 同 category、
   答案不同的物件並通過第 2 條,此物件為 third object;c3(換 anchor)要求另一物件對
   target 唯一成立同一 relation、名稱唯一、category 與兩者皆不同。至少一個可構造。
10. 行為分離集:target 落在方向詞相反的影像半邊,另記子集。

Same-as(open query 245 題):
11. 問句不含 anchor 的 colour(模板天然滿足);場景中恰一個其他物件與 anchor 同色
    (target);另有一個不同色、與 target 同 category 的物件(third object;若無,
    取任一不同色物件,分開報)。c2 = 命名 target 為 anchor。

### 3.3 Role 與 patch 分組

anchor / referent、target、third object / distractor 各取 box 內 patch。background 分
(a) 有標注但非 role 的物件、(b) 不在任何標注 box 內,分開報;transplant 的 background
組用 (b),regression 兩類都跑。

### 3.4 Forward pass

c0 無問句;c1 原題;c2 依題型;c3(spatial)換 anchor。

## 4. Primary 假設(四條;統計判準跑前凍結)

通用統計規則:所有 CI 為 bootstrap by image(1,000 次,95%);「起始層」以預先定義的
threshold 與 CI 下界決定,不以肉眼讀曲線;層的自由度以預先指定的層窗處理,全曲線另附。

### H1 — Referent selection generalizes
語言條件化在自然場景中把被指涉物件相對於非指涉物件重新組織。

- **GQA 定義**(observational,S_eligible):V = c0 下 referent 的 patch mean 減 background (b)
  的 patch mean(單位向量);selection contrast = referent 的 patch mean 在 V 上的投影
  (c1 − c0)減 non-referent 的同一量(c2 − c0)。
- **統計**:第 5 到 11 層(CLEVR SigLIP 的分裂層窗)的平均 contrast,bootstrap CI。
- **通過**:CI 下界 > 0。輔以 decoder attention 的 referent / background per-patch
  mass 比,CI 下界 > 1。
- **Causal 對應**(secondary,S_correct):activation patching object 組 c1 ← c2。

### H2 — Bound information becomes available downstream
anchor 的任務相關性質在 relational target 被選出之前就可從 anchor 之外 decode。

- **GQA 定義**(observational,S_eligible):spatial:從 background (b) 的 token 以 ridge
  回歸 anchor 的 box 中心,ΔR² = R²(c1) − R²(c0),GroupKFold(5) by image;same-as:
  從 background (b) 以 7-way probe 讀 anchor 的 colour,ΔAcc = Acc(c1) − Acc(c0)。
- **k_condition**:第一個 ΔR²(或 ΔAcc)的 CI 下界 > 0.1 的層。
- **k_target**(causal,S_correct):第一個 transplant target 組使 P(clean answer) 下降
  ≥ 0.2 且 CI 不含 0 的層。
- **統計**:ordering score = bootstrap 樣本中 k_condition < k_target 的比例。
- **通過**:ordering score ≥ 0.9。
- **leakage 控制**(§6):label-shuffled probe、category-matched split、
  category-only baseline;ΔR² 是主量,不報 R²(c1) 單獨值。

### H3 — Relation type determines the downstream computation
feature comparison 顯示稀疏的 head 層級因果定位;spatial 未顯示同樣的稀疏定位。

- **GQA 定義**(causal,S_correct):依固定規則(|c1 − c0| 的 SA attention 到 anchor
  ≥ 10× median,上限 8 個)選 head,zero;Δ = accuracy 下降;random = 同數量 disjoint
  隨機 head(3 個 seed)。
- **統計**:interaction = (Δ_selected − Δ_random)_same − (Δ_selected − Δ_random)_spatial,
  bootstrap CI。
- **通過**:interaction 的 CI 下界 > 0。
- **cumulative ablation curve**(primary 的一部分):同一 ranking 累積 ablate
  m = 1, 2, 4, 8, 16, 32 個 head,對照 matched-random curve;報兩個題型的曲線與
  「掉到 0.5 baseline 所需的 m」。
- **解讀限制**:spatial 的 null 只寫成「spatial does not exhibit the sparse head-level
  localization detected by this selection criterion」;不寫「distributed」,除非
  cumulative curve 顯示 spatial 需要多個 head 才逐步下降而 same-as 前幾個就陡降。
  null 不稱 evidence of absence,除非附 equivalence test(預先設 margin 0.05)。

### H4 — Relational selection returns to a shared referential state
被選出的 target 取得 direct query 的 referent representation。

- **GQA 定義**(observational,S_eligible):V 取自 direct 題(H1)的 referent 方向;
  relational 題中 target、anchor、third object 的 patch mean 在 V 上的投影 c1 − c0。
- **統計**:第 9 到 11 層平均,target 減 third object 的差,bootstrap CI。
- **通過**:CI 下界 > 0。
- **causal 對應**(secondary,S_correct):把 direct 題 c1 的 referent 狀態 transplant
  到 relational 題的 target patch,看答案是否維持。

若 H1–H4 皆成立,故事為:binding → {attribute retrieval, feature-based reasoning,
spatial reasoning} → shared target selection。任一不成立,該分支的主張撤回,不改判準。

## 5. Secondary 與 exploratory(照報,不進 confirmatory 判準)

| 原 id | 內容 | 地位 | 母體 |
|---|---|---|---|
| D1 | per-patch referent probe 起始層 | secondary | eligible |
| D3 | attribute-specific direction 的分裂 | secondary | eligible |
| D5 | activation patching(H1 的 causal 對應) | secondary | correct |
| D6 | additive colour intervention 與對照 | secondary | correct |
| D7 | 258 個單 head / 整層 ablation scan | secondary | correct |
| D8 | attention 的 high-norm 控制 | secondary | eligible |
| D9 | background 帶 question type(SigLIP-specific) | exploratory | correct |
| D10 | GCA write norm 無 role 選擇性 | exploratory | eligible |
| R0 | 行為:分離集、c2 移動比例 | secondary | eligible(行為本身) |
| R3 | candidate → anchor 的 SA attention 峰 | secondary | eligible |
| R5 | GCA write mask(anchor 未被 suppress) | exploratory | correct |
| R6 | positional embedding 翻轉 | secondary | correct |
| R7 | absolute → relative write field(DINOv2 s42-specific) | exploratory | eligible |
| synthetic c2 全部 | | secondary,不 pool | — |
| CLEVR 訓練的模型全部 | | secondary(transfer) | 其自身的 eligible / correct |
| threshold robustness(1/2/3 patch) | | appendix | — |

不可移植、留在 CLEVR 的觀察:template RSA(X21-A7/A8)、成對渲染的 additivity 與
KMeans(X19)、無 GCA 模型(X20-C)。

CLEVR 訓練的模型的分析何時跑:不用 aggregate accuracy 門檻;需要 clean answer 的
intervention 在其 S_correct 達到預先設定的最小 n(100)時跑,representational 分析在
S_eligible 上一律跑。

## 6. Probe 的 leakage 控制(H2、D1、D3 通用)

1. c1 對 c0 的差(ΔR²、ΔAcc)為主量。
2. label-shuffled probe(within image)給 chance 分布。
3. category-matched split:train / test 的 anchor category 不重疊,排除「banana 預測
   yellow」式的 category → attribute 捷徑。
4. category-only baseline:只用 anchor category(one-hot)預測同一 label,報其 accuracy;
   probe 必須顯著高於它。
5. image-grouped CV(GroupKFold(5) by image)。
6. same-as 的 colour probe 另限制到 colour entropy 高的 category(scene graph 中該
   category 的顏色分布 entropy ≥ 1.5 bit)做子集分析。

## 7. Human audit(blind)

- 分層抽樣:direct、spatial、same-as 各 ≥ 50 題,總計 150–200;synthetic c2 另抽 50。
- 審查者看不到模型答案與任何機制結果,只檢查:anchor / target assignment、box 品質、
  relation 有效性、唯一性、synthetic c2 的語法與語意。
- 至少一位審查者;兩位則報 agreement(Cohen's κ)。錯誤率 > 10% 的題型加全量複核
  (只複核,不挑題)。
- audit 在任何機制分析之前完成並凍結。

## 8. 控制組

c0 無問句;transplant 與 patching 的 clean-self 每格 1.00;disjoint 隨機 head(3 seed);
norm-matched random vector;random background rows;label shuffle;category-only
baseline;equivalence test 的 margin 0.05。

## 9. 實作

全部放在 `scripts/analysis/patch_language_condition.py`:`assign_roles` 加 GQA 路徑
(scene graph box、program object id、幾何規則、natural pair 配對、synthetic c2 生成與
標記);`--gqa-filter` 為 CPU 篩選,輸出 `records.jsonl`、funnel 表(含每步排除原因與
數量)、audit 抽樣清單;既有 flags 沿用;新增 `--h8-cumulative` 做 cumulative ablation
curve;bootstrap 與 CI 統一在後處理。輸出目錄
`outputs/analysis/patch_language_condition/gqa_{direct,spatial,same}/` 與
`gqa_{direct,spatial}_clevrmodel/`。任何規則放寬記為 `x23_v2` 目錄,不覆蓋。

## 10. 執行順序與成本(GPU 一次一件)

| 步驟 | 內容 | 資源 |
|---|---|---|
| 0 | 篩選、funnel、audit 抽樣與審查、threshold 三版本的母體 | CPU,一到兩天(含 audit) |
| 1 | GQA 訓練的模型:c0–c3 cache;H1、H4、H2 的 observational 部分;D1/D3/D8/D10/R3 | GPU 約 1.5 小時 + CPU |
| 2 | H2 的 k_target transplant;D5;D6 | GPU 約 2 小時 |
| 3 | H3 head ablation + cumulative curve(same、spatial);D7;R6;R5 | GPU 約 2.5 小時 |
| 4 | CLEVR 訓練的模型:同上,依最小 n 規則 | GPU 約 3 小時 |
| 5 | bootstrap、圖、JSON、JOURNAL、registry X23 | CPU |

步驟 0 完成並凍結 threshold 後才排 GPU。

## 11. 第二輪(視第一輪)

RefCOCO+ 圖片(本機 `data/refcocop`,19,992 張 COCO train2014 圖、box、不含位置詞的
expression)以 GQA 模板合成 spatial 與 direct 題,篩選同 §3,mask 從 COCO instances 補;
material / size 被問 attribute 的變體。

## 12. 風險

- rel/query 子集 accuracy 約 0.5,S_correct 可能不足:coverage 照報;intervention 的
  最小 n 為 100,未達則只報 observational。
- scene graph 漏標:audit 錯誤率與全量複核規則。
- natural pair 可能稀少:synthetic c2 為 secondary,不補進 primary。
- 真實影像的 role 大或遮擋:IoU = 0 與 ≤ 30% 規則,另報 role 大小分層。
- 單一真實影像 backbone:跨 backbone 主張留在 CLEVR。
- checkpoint 非 independent(§2 聲明)。

## 13. 學術倫理規則(寫進 registry,跑前提交)

- 所有 threshold、層窗、最小 n、equivalence margin 在看任何結果前凍結。
- 篩選規則的任何放寬成為 X23-v2,原結果保留不覆蓋。
- 同一 model / dataset 上重複探索後不稱 independent replication;用「held-out items,
  not checkpoint-independent」。
- observational probe 不稱 causal;null ablation 不稱 evidence of absence,除非附
  equivalence 或 power analysis。
- 公開報告 excluded-item 數量與排除原因(funnel)。
- 結果無論方向全部報告;secondary / exploratory 不事後升級為 primary。

## 14. 記錄

registry 新增 X23(本表);JOURNAL 在步驟 0 與每個 GPU 步驟後追加;結果先給使用者,
網站與 RESULTS 待使用者決定。

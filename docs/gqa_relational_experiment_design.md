# X23:CLEVR 上的機制發現能不能在真實影像(GQA)上存活

> 提案第四版,2026-09-11。狀態:未啟動,等使用者核可。
> 第四版只改寫法,內容與第三版相同(第三版依使用者十點修改)。
> 事實來源:`docs/experiment_registry.md` §X19–X22、`JOURNAL.md`、
> `outputs/model/gqa_siglip_decoder1l_scratch_s42.log`、`src/data/gqa.py`、
> `src/model/model.py`、`scripts/analysis/patch_language_condition.py`、2026-09-11 對
> `val_balanced_questions.json` 的 program 統計。

## 一頁摘要

我們在 CLEVR 上找到一套機制:語言經 GCA 寫進 visual stream,被問到的物件被標出,答案
從它身上讀走;relational 問題多一步,先標 anchor,再用 anchor 的某個性質去選 target。
這些發現依賴 CLEVR 的 scene graph 和成對渲染,真實影像沒有這兩樣。X23 問的是:**換成
自然場景、真實標注、自然問句之後,這套機制哪些還在。**

做法:用 GQA val balanced 的題目,以規則篩出可分析的樣本(不手挑),在 GQA 訓練的模型上
檢驗四條主假設,每條在跑前寫死統計判準。CLEVR 訓練的模型做 transfer 測試,屬 secondary。

四條主假設:
- **H1** 被指涉的物件在自然場景中一樣被語言選出來。
- **H2** anchor 的性質在 target 被選出之前就可以從 anchor 之外讀到。
- **H3** 關係的種類決定後面的計算:attribute 比較有稀疏的 head 定位,spatial 沒有。
- **H4** 被選出的 target 得到和 direct query 一樣的 referent 表徵。

四條都成立,故事就是:binding → {attribute retrieval, feature-based reasoning, spatial
reasoning} → shared target selection。任一條不成立,撤回那一支的主張,不改判準。

## 1. 為什麼這樣設計

嚴謹度的風險不在模型,在三件事:

- **樣本怎麼選。** 如果只分析模型答對的題,問的就變成「在模型成功的例子裡機制是否可見」。
  所以母體不能用模型輸出定義(§4)。
- **哪些是 confirmatory、哪些是 exploratory。** 只有四條主假設是 confirmatory,判準跑前
  凍結;其餘一律 secondary 或 exploratory,不事後升級(§5、§6)。
- **CLEVR 的量在 GQA 上是不是同一個量。** 每個量在 GQA 上重新定義,文字上分清
  observational 和 causal,不用 CLEVR 的機制語言描述 GQA 上還沒做因果檢驗的東西。

主張層級和 CLEVR 的跨 backbone 主張一樣:只主張順序、方向、定位的有無,不主張層數與數值。

## 2. 名詞,定義一次

- **role**:anchor(relational 題裡被命名的物件;direct 題叫 referent)、target(答案物件)、
  third object(counterfactual 下會變成答案的物件;direct 題叫 distractor)。
- **background**:(a) 有標注但不是 role 的物件的 patch;(b) 不在任何標注 box 內的 patch。
  兩類分開報;CLEVR 的 background 對應 (b)。
- **forward pass**:c0 沒有問句;c1 原題;c2 counterfactual(direct:指向 distractor 的
  配對題;spatial:換方向詞;same-as:把 target 當 anchor 命名);c3(spatial)換 anchor。
- **S_eligible**:只由 scene graph、program、幾何規則定義的樣本,與模型無關。
- **S_correct**:S_eligible 裡 c1 答對的樣本。
- **coverage** = n_correct / n_eligible。
- **observational**:probe、attention、regression、projection,在 S_eligible 上做。
- **causal**:transplant、patching、head ablation、positional embedding 翻轉、additive
  intervention,以 clean answer 為 endpoint,在 S_correct 上做。
- **CI**:全部是 bootstrap by image,1,000 次,95%。

## 3. 模型

| 模型 | checkpoint | 訓練資料 | 地位 |
|---|---|---|---|
| GQA 訓練的模型 | `gqa_siglip_decoder1l_scratch_s42` | GQA balanced train | primary |
| CLEVR 訓練的模型 | `clevr_siglip_decoder1l_scratch_s42` | CLEVR | secondary(transfer 測試) |

兩者共用 SigLIP ViT-B/16 @256(grid 16×16)、GCA 在第 1/3/5/7/9/11 層、1 層 decoder、
凍結的 RoBERTa-large 文字端、分類式答案。影像直接 resize 到 256×256,沒有 crop,所以
scene graph 的 box 線性對到 patch。CLEVR 訓練的模型只有 28 個答案,顏色八種,所以它只
能答顏色在那八種之內的題。

GQA 訓練的模型在 val balanced 的 accuracy:overall 0.636;有 relate 的題 0.548;開放式
答案的題 0.509。

**兩個必須寫在論文裡的限制。**
第一,這個 checkpoint 是 2026-06 依 val balanced 的整體 accuracy 選的。X23 沒有任何機制
測量參與過選 checkpoint,但 X23 的題目仍是「held-out items, not checkpoint-independent」,
不能稱 independent replication。乾淨的做法是重訓一次,從 train 切一小塊選 checkpoint,
val 完全凍結給 X23;成本約 36 小時 GPU。要不要重訓由使用者決定。
第二,CLEVR 訓練的模型和 GQA 訓練的模型差的不只是影像領域,還有問句分布、答案詞彙、
關係頻率、優化軌跡。若機制在它身上重現,只能寫「transfers across substantial visual and
linguistic distribution shift without GQA-specific training」,不能寫「來自架構而非資料」。

## 4. 資料與母體

**來源。** GQA 建立在 Visual Genome 的圖片與 scene graph 之上,圖片來自 COCO 與
YFCC100M。問句由 program 生成,relate 步驟直接帶 target 的 object id,所以 role 不用人標。
只有 train 和 val 有 scene graph,所以用 val balanced(132,062 題)。

**母體有多大(2026-09-11 統計)。**

| 題型 | n |
|---|---|
| direct colour / attribute query | 6,481 |
| spatial(left / right,單一 relate,無額外 filter),答 category | 3,619 |
| spatial 同上,答 colour / size / material | ≈ 900 |
| same-as open query(「和 Y 同色的 X 是什麼」) | 245 |

**篩選原則。** 全部用規則,不手挑;每一步記數量與排除原因,論文報 funnel。規則細則在
附錄 A,這裡只講四個關鍵決定:
- 母體 S_eligible 不含「模型答對」這一條。答對只用來定義 S_correct。
- direct 題的 c2 只用 GQA 原生的配對題(同圖、指向另一物件、同 attribute、答案不同)。
  自動改寫的 synthetic c2 是 secondary,不和 natural 混在一起。
- spatial 只用 left / right。front / behind 牽涉深度、大小、遮擋,是另一種計算,排除。
  另外要求 scene graph 的標注和 box 中心算出的 2D 關係一致,論文寫成「restricted to
  relations whose annotation agrees with a preregistered 2-D image-plane criterion」。
- 沿關係軸的距離門檻是 2 個 patch(CLEVR 比例換算取整),1 / 2 / 3 三個版本都跑,附錄報
  效應是否穩定。

**與 Song, Lepori & Pavlick 2026 的差別。** 他們人工挑 100 張 COCO 圖並自行合成問句;
我們不手挑、報 funnel、做 blind audit(§7)。

## 5. 四條主假設

每條三段:主張、怎麼量、怎麼算通過。層的自由度用預先指定的層窗處理,全曲線另附。

### H1 — Referent selection generalizes

**主張。** 語言條件化在自然場景中把被指涉物件相對於非指涉物件重新組織。

**怎麼量(observational,S_eligible)。** 取 c0 下 referent 的 patch mean 減 background (b)
的 patch mean,單位化,叫 V。selection contrast = referent 在 V 上的投影變化(c1 − c0)
減 non-referent 的同一量(c2 − c0)。取第 5 到 11 層的平均(CLEVR SigLIP 的分裂層窗)。

**通過。** contrast 的 CI 下界 > 0。輔助指標:decoder attention 的 referent 對 background
每 patch 比值,CI 下界 > 1。causal 對應(secondary):activation patching。

### H2 — Bound information becomes available downstream

**主張。** anchor 的任務相關性質在 target 被選出之前就可以從 anchor 之外讀到。

**怎麼量。** spatial:從 background (b) 的 token 回歸 anchor 的 box 中心,主量是
ΔR² = R²(c1) − R²(c0)。same-as:從 background (b) 讀 anchor 的 colour,主量是
ΔAcc = Acc(c1) − Acc(c0)。k_condition = 第一個 ΔR²(或 ΔAcc)CI 下界超過 0.1 的層。
k_target(causal,S_correct)= 第一個把 target 的 patch 換成 c2 版本後 P(clean answer)
掉 ≥ 0.2 且 CI 不含 0 的層。

**通過。** ordering score = bootstrap 樣本中 k_condition < k_target 的比例,≥ 0.9。
probe 的 leakage 控制見附錄 B,尤其要排除「banana 本來就預測 yellow」。

### H3 — Relation type determines the downstream computation

**主張。** attribute 比較有稀疏的 head 層級因果定位;spatial 沒有同樣的稀疏定位。

**怎麼量(causal,S_correct)。** 用固定規則選 head(c1 − c0 的 SA attention 到 anchor
≥ 10× median,上限 8 個),zero 掉,量 accuracy 下降 Δ;對照是同數量的隨機 head
(3 個 seed)。另做 cumulative ablation:同一 ranking 累積 zero 掉 1、2、4、8、16、32 個
head,對照 matched-random 曲線。

**通過。** interaction = (Δ_selected − Δ_random)_same − (Δ_selected − Δ_random)_spatial,
CI 下界 > 0。**解讀限制:** spatial 的 null 只能寫「does not exhibit the sparse head-level
localization detected by this criterion」。要寫「distributed」,得 cumulative curve 顯示
spatial 需要很多 head 才逐步下降、same-as 前幾個就陡降。null 不稱 evidence of absence,
除非附 equivalence test(margin 0.05)。

### H4 — Relational selection returns to a shared referential state

**主張。** 被選出的 target 取得 direct query 的 referent 表徵。

**怎麼量(observational,S_eligible)。** V 取自 direct 題(H1)。relational 題裡 target、
anchor、third object 在 V 上的投影變化(c1 − c0),取第 9 到 11 層平均。

**通過。** target 減 third object 的差,CI 下界 > 0。causal 對應(secondary):把 direct
題的 referent 狀態 transplant 到 relational 題的 target patch,看答案是否維持。

## 6. 其他分析(secondary / exploratory,照報,不進判準)

| 內容 | 對應 CLEVR | 地位 | 母體 |
|---|---|---|---|
| per-patch referent probe 起始層 | X21-C | secondary | eligible |
| attribute-specific direction 的分裂 | X21-E | secondary | eligible |
| activation patching | X21-D2 | secondary | correct |
| additive colour intervention | X21-B | secondary | correct |
| 258 個單 head / 整層 ablation scan | X21-H | secondary | correct |
| attention 的 high-norm 控制 | X21-M2 | secondary | eligible |
| 行為:分離集、c2 移動比例 | X22-H7d | secondary | eligible |
| candidate → anchor 的 SA attention | X22-H2b | secondary | eligible |
| positional embedding 翻轉 | X22-H7b | secondary | correct |
| background 帶 question type | X21-D3 | exploratory | correct |
| GCA write norm 無 role 選擇性 | X21-A5 | exploratory | eligible |
| GCA write mask | X22-H5 | exploratory | correct |
| absolute → relative write field | X22-H7c | exploratory | eligible |
| synthetic c2 的所有分析 | — | secondary,不 pool | — |
| CLEVR 訓練的模型的所有分析 | — | secondary | 自身的 eligible / correct |
| 距離門檻 1 / 2 / 3 patch | — | appendix | — |

留在 CLEVR、不移植:template RSA、成對渲染的 additivity 與 KMeans、無 GCA 模型。

CLEVR 訓練的模型何時做 causal 分析:不用 accuracy 門檻,S_correct 達到最小 n = 100 就做;
observational 分析在 S_eligible 上一律做。

## 7. 人工審查(blind)

direct、spatial、same-as 各抽 ≥ 50 題,總計 150 到 200;synthetic c2 另抽 50。審查者看不到
模型答案與任何機制結果,只檢查 role 指派、box 品質、關係有效性、唯一性、synthetic 句的
語法與語意。至少一位;兩位則報 Cohen's κ。某題型錯誤率 > 10% 就全量複核(只複核,不
挑題)。審查在任何機制分析之前完成並凍結。

## 8. 執行順序與成本(GPU 一次一件)

| 步驟 | 內容 | 資源 |
|---|---|---|
| 0 | 篩選、funnel、audit、三個門檻版本的母體 | CPU,一到兩天 |
| 1 | GQA 訓練的模型:c0–c3 cache;H1、H4、H2 的 observational 部分;secondary 的 observational 項 | GPU 約 1.5 小時 + CPU |
| 2 | H2 的 k_target transplant;patching;additive intervention | GPU 約 2 小時 |
| 3 | H3 head ablation 與 cumulative curve;head scan;pos-embed;write mask | GPU 約 2.5 小時 |
| 4 | CLEVR 訓練的模型:同上,依最小 n 規則 | GPU 約 3 小時 |
| 5 | bootstrap、圖、JSON、JOURNAL、registry X23 | CPU |

步驟 0 完成、門檻凍結後才排 GPU。第二輪(視結果):RefCOCO+ 圖片合成題目;material /
size 的變體。

## 9. 規則(寫進 registry,跑前提交)

- 所有門檻、層窗、最小 n、equivalence margin 在看任何結果前凍結。
- 篩選規則的任何放寬是 X23-v2,原結果保留不覆蓋。
- 不稱 independent replication;用「held-out items, not checkpoint-independent」。
- observational 不稱 causal;null 不稱 evidence of absence。
- 報 excluded-item 數量與原因;結果無論方向全報;secondary 不升級為 primary。

## 10. 風險

- 有 relate 的題 accuracy 約 0.5,S_correct 可能不足:coverage 照報,不足就只報 observational。
- scene graph 漏標:audit 錯誤率與全量複核。
- natural pair 稀少:synthetic c2 留在 secondary,不補進 primary。
- 真實影像的物件大或遮擋:IoU = 0 與 ≤ 30% 影像的規則,另報物件大小分層。
- 只有一個真實影像 backbone:跨 backbone 主張留在 CLEVR。

## 11. 實作與記錄

全部放在 `scripts/analysis/patch_language_condition.py`:`assign_roles` 加 GQA 路徑,
`--gqa-filter` 做 CPU 篩選並輸出 records、funnel、audit 清單,新增 `--h8-cumulative`,
其餘 flags 沿用。輸出到 `outputs/analysis/patch_language_condition/gqa_{direct,spatial,same}/`
與 `gqa_{direct,spatial}_clevrmodel/`。registry 新增 X23;JOURNAL 在步驟 0 與每個 GPU
步驟後追加;結果先給使用者,網站與 RESULTS 待使用者決定。

---

## 附錄 A:S_eligible 的篩選規則(依序套用)

共用:
1. anchor(direct 為 referent)的名稱在 scene graph 中唯一;relational 題 anchor 與
   target 的 category 不同。
2. box 幾何(patch 座標):role 之間 IoU = 0;每個 role ≥ 4 個 patch;任一 role ≤ 30% 影像。
3. 三個 role 的答案值都在 GQA 訓練的模型的答案詞彙內。CLEVR 訓練的模型另加「答案在
   八種顏色內」,形成子集。

Direct:
4. natural pair(primary):同圖另有一題 GQA 原生 direct query,指向不同物件、問同一
   attribute、答案不同,互為 c1 / c2。
5. synthetic c2(secondary):以 scene graph 改寫 referent 名稱、答案取自 scene graph。
   附 linguistic control:synthetic 句與原句的 RoBERTa embedding 距離分布;c1 → natural
   與 c1 → synthetic 分開報;human audit 檢查語法與語意。

Spatial:
6. relation 只取 to the left of / to the right of。
7. 2D 一致性:box 中心算出的 left / right 與 scene graph 標注一致,不一致者排除並報比例。
8. 沿關係軸中心距 ≥ 2 個 patch;1 / 2 / 3 三版。
9. c2:anchor 另一側恰有一個與 target 同 category、答案不同、通過第 2 條的物件(即
   third object)。c3:另有一物件對 target 唯一成立同一 relation、名稱唯一、category 與
   anchor / target 皆不同。至少一個可構造。
10. 分離集:target 落在方向詞相反的影像半邊,另記子集。

Same-as(245 題):
11. 問句不含 anchor 的 colour(模板天然滿足);場景中恰一個其他物件與 anchor 同色
    (target);另有一個不同色、與 target 同 category 的物件(third object;若無則取任一
    不同色物件,分開報)。c2 = 把 target 當 anchor 命名。

## 附錄 B:probe 的 leakage 控制(H2 與 secondary 的 probe 通用)

1. 主量是 c1 對 c0 的差(ΔR²、ΔAcc),不單獨報 R²(c1)。
2. label-shuffled probe(within image)給 chance 分布。
3. category-matched split:train / test 的 anchor category 不重疊。
4. category-only baseline:只用 anchor 的 category 預測同一 label,probe 必須顯著高於它。
5. image-grouped CV:GroupKFold(5) by image。
6. same-as 的 colour probe 另限制到 colour entropy 高的 category(scene graph 中該 category
   的顏色分布 entropy ≥ 1.5 bit)做子集分析。

## 附錄 C:控制組一覽

c0 無問句;transplant 與 patching 的 clean-self 每格 1.00;disjoint 隨機 head(3 seed);
matched-random cumulative curve;norm-matched random vector;random background rows;
label shuffle;category-only baseline;equivalence margin 0.05。

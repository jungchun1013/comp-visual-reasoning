# X23:CLEVR 上的機制發現能不能在真實影像(GQA)上存活

> 提案第五版,2026-09-11。狀態:未啟動,等使用者核可。
> 第五版把每個設計決定的理由寫出來;判準、控制、規則與第三版相同,沒有放鬆。
> 事實來源:`docs/experiment_registry.md` §X19–X22、`JOURNAL.md`、
> `outputs/model/gqa_siglip_decoder1l_scratch_s42.log`、`src/data/gqa.py`、
> `src/model/model.py`、`scripts/analysis/patch_language_condition.py`、2026-09-11 對
> `val_balanced_questions.json` 的 program 統計。

## 1. 我們要回答什麼,為什麼這個問題不能直接「搬」

CLEVR 上的發現可以濃縮成一句話:語言經 GCA 寫進 visual stream 之後,被問到的物件被
標出來,它的被問 attribute 從它身上讀走;relational 問題多一步,先標 anchor,再拿 anchor
的某個性質(same-as 是它的 attribute,spatial 是它的位置)去選 target,target 選出來之後
走的是和 direct query 一樣的讀出路徑。

這些發現之所以能做得乾淨,靠的是 CLEVR 給的兩樣東西。第一是 scene graph:每個物件的
位置、attribute、彼此關係都精確已知,所以「anchor 左邊唯一的 cube 是哪個」這種 predicate
可以確定求值,role 指派沒有模糊空間。第二是成對渲染:同一場景可以只換一個物件的一個
attribute 再渲染一次,counterfactual 只差一個變數。真實影像兩樣都沒有:標注有漏有錯,
物件互相遮擋,同一張圖沒有「只差一個變數」的孿生版本。原 workshop paper 自己把這點列
為最大限制。

所以 X23 的問題不是「CLEVR 的十九個效應在 GQA 上有幾個也出現」。那樣的清單式重現有
兩個毛病:讀者不知道哪個效應不出現會推翻故事;而且每個效應都用 CLEVR 的操作定義去量,
量出來的東西在真實影像上未必是同一個量。X23 的問題是:**把機制的主張縮到四條、每條在
真實影像上重新定義它的量、跑前寫死什麼算成立,然後看在自然場景的變異、標注的模糊與
問句的自然分布下,哪幾條存活。**

主張的層級跟 CLEVR 的跨 backbone 主張一樣:只主張 stage 的順序、效應的方向、head 定位
的有無;不主張層數與數值,因為 SigLIP 在 CLEVR 上就已經比 DINOv2 淺兩層,真實影像上
再移動是預期內的事。

## 2. 三個會讓結果不可信的地方,以及對應的設計

在寫任何假設之前,先講三個風險。它們不是實作細節,是這份設計的骨架。

**第一,樣本怎麼選。** 最順手的做法是只分析模型答對的題。這會把問題悄悄換掉:我們問
的是「GQA 的處理過程有沒有重現 CLEVR 的機制」,分析母體卻變成「模型碰巧成功的 GQA
例子」,在裡面看到機制不奇怪,看不到也說不了什麼。更糟的是兩個模型答對的題不同,
CLEVR 訓練的模型 zero-shot 答對的那批會是一個更奇特的子集,兩個模型根本沒在分析同一組
影像。所以母體分成兩層:S_eligible 只由 scene graph、program 與幾何規則定義,和任何模型
輸出無關;S_correct 是 S_eligible 裡 c1 答對的那些。probe、attention、regression、
projection 這類 observational 分析在 S_eligible 上做,因為它們不需要模型答對也有意義;
只有以「clean answer 是否被改變」當 endpoint 的 intervention 才用 S_correct,因為答錯的
題沒有 clean answer 可以被翻轉。每個結果都報 n_eligible、n_correct 與
coverage = n_correct / n_eligible,讀者可以自己判斷 causal 結果代表了母體的幾成。

**第二,哪些結論是事先預測的,哪些是看了資料才說的。** CLEVR 累積了幾十個觀察,如果
全部搬來當假設,結果一定有些成立有些不成立,故事就變成事後挑。所以 confirmatory 的部分
只有四條主假設(§5),每條的統計量、層窗、門檻在跑前凍結;其餘所有分析(§6)一律標
secondary 或 exploratory,照報但不進判準,也不在看了結果之後升級。

**第三,CLEVR 的量搬到 GQA 上是不是同一個量。** 例如 CLEVR 的 background 是真的空白
地板;GQA 的「非 role patch」裡有其他物件、有牆有天空。CLEVR 的 anchor colour 從
background 讀得出來就是被搬運過去;GQA 裡 anchor 是 banana 的話,pretrained 特徵可能本來
就從 category 猜得到 yellow。所以每個量在 GQA 上重新定義,並且文字上分清楚
observational(相關)和 causal(介入),不用 CLEVR 的機制語言去描述 GQA 上還沒做介入
的東西。

## 3. 名詞,全文只定義這一次

- **role**:anchor 是 relational 題裡被命名的物件(direct 題叫 referent);target 是答案
  物件;third object 是 counterfactual 下會變成答案的物件(direct 題叫 distractor)。
- **background**:分兩類,(a) 有標注但不是 role 的物件的 patch,(b) 不在任何標注 box
  內的 patch。分開報,因為 (a) 在 CLEVR 不存在,把它們混進 (b) 會讓 GQA 的 background
  和 CLEVR 的 background 不是同一個東西。
- **forward pass**:c0 沒有問句;c1 原題;c2 是 counterfactual(direct:同圖指向
  distractor 的配對題;spatial:換方向詞;same-as:把 target 當 anchor 命名);c3 只有
  spatial 有,換 anchor。
- **S_eligible、S_correct、coverage**:如 §2。
- **observational / causal**:如 §2。
- **CI**:全部是 bootstrap by image,1,000 次,95%。by image 而不是 by question,因為同
  一張圖的多題共用影像特徵,不獨立。

## 4. 模型,以及兩個必須寫進論文的限制

主要模型是 GQA 訓練的 `gqa_siglip_decoder1l_scratch_s42`:凍結 SigLIP ViT-B/16 @256
(grid 16×16)、GCA 在第 1/3/5/7/9/11 層、1 層 decoder、凍結的 RoBERTa-large 文字端、
分類式答案(1,500 個)。影像直接 resize 到 256×256 沒有 crop,所以 scene graph 的 box
線性對到 patch 格。它在 val balanced 的 accuracy 是 overall 0.636、有 relate 的題 0.548、
開放式答案的題 0.509。

對照模型是 CLEVR 訓練的 `clevr_siglip_decoder1l_scratch_s42`,同一個 backbone、同一個
架構,只是訓練資料是 CLEVR。它的文字端凍結,所以能編碼 GQA 的句子;但 GCA 的投影只
看過 CLEVR 句子,答案只有 28 個,顏色八種。它只能答顏色在那八種內的題。

**限制一:val 同時是選 checkpoint 的集合,又是 X23 的測試集。** `best.pt` 是 2026-06 依
val balanced 的整體 accuracy 選出來的。這不是 label leakage,X23 沒有任何機制測量參與過
選 checkpoint;但它不是 fully held-out 的 confirmatory evaluation,而且如果之後反覆看
val 上的機制結果再調篩選規則,就會一步步變成 researcher overfitting。乾淨的做法是重訓
一次:GQA train 切出一小塊 development split 選 checkpoint,val balanced 完全凍結給 X23。
成本是一次訓練,每 epoch 約 2 小時、18 epoch 約 36 小時,佔唯一的 GPU。要不要付這個
成本由使用者決定;不重訓的話,論文用「held-out items, but not checkpoint-independent」
這個措辭,不稱 independent replication,並登記「checkpoint selection predates X23 and
used only aggregate validation accuracy」。

**限制二:CLEVR 訓練的模型能證明的比直覺少。** 直覺上會想說:同一個架構、同一個
backbone,只換訓練影像,如果機制在 GQA 上也出現,那機制就來自架構而不是資料。這個
推論不成立,因為兩個模型差的不只是訓練影像:問句分布、答案詞彙、視覺統計、關係出現
的頻率、優化的軌跡全都不同,任何一項都可能是差異的來源。所以 CLEVR 訓練的模型是
secondary 的 transfer 測試;若機制在它身上重現,只能寫「the mechanism transfers across
substantial visual and linguistic distribution shift without GQA-specific training」。
另外它什麼時候做 causal 分析,不用「accuracy ≥ 0.3」這種門檻,因為 0.3 沒有統計或功能
上的根據,是一個事後風格的 cutoff;改成 S_correct 達到預先設定的最小 n = 100 就做,
observational 分析在 S_eligible 上一律做。

## 5. 資料:為什麼用 GQA,母體有多大,篩選的幾個關鍵決定

GQA 建立在 Visual Genome 的圖片與 scene graph 之上,圖片來自 COCO 與 YFCC100M。問句由
functional program 生成,relate 步驟的 argument 直接帶 target 的 object id,問句裡的名詞
與答案也對到 object id,所以 role 不用人標。只有 train 和 val 有 scene graph,所以用
val balanced(132,062 題)。選 GQA 而不是 COCO,是因為機制分析需要 relation 與 role,
COCO 只有 mask;圖片本身兩者同源。

母體(2026-09-11 統計):direct colour / attribute query 6,481 題;spatial 題裡 left /
right、單一 relate、無額外 filter 的,答 category 3,619 題、答 colour / size / material
約 900 題;same-as 的 open query(「和 Y 同色的 X 是什麼」)245 題。

篩選全部用規則,不手挑,每一步記數量與排除原因,論文報 funnel。這是和 Song, Lepori &
Pavlick 2026 的差別:他們人工挑 100 張 COCO 圖並自行合成問句,挑選過程無法被檢驗。
規則的完整清單在附錄 A;有四個決定需要說理由。

**Direct 題的 counterfactual 只用 GQA 原生的配對題。** direct 題的 c2 是「同一張圖、
指向另一個物件、問同一 attribute、答案不同」的另一題。GQA 不保證每題都有這樣的配對,
順手的補法是拿 scene graph 自動改寫 referent 名稱。但這樣做的 intervention 同時改了兩件
事:指涉的物件,和句子的自然度。GQA 生成句有固定模板,我們改寫的句子未必在模板分布內,
模型的反應差異就分不清是因為 referent 換了還是因為句子怪了。所以 synthetic c2 是
secondary,不和 natural pair 合併;附 linguistic control(synthetic 句與原句的 RoBERTa
embedding 距離分布、c1 → natural 與 c1 → synthetic 分開報、human audit 檢查語法與語意)。
natural pair 不夠就只報 natural pair 的 n,不補。

**Spatial 只用 left / right,並且要求標注和 2D 幾何一致。** GQA 的 relation 是標注者
判斷的,不是從座標算的。「left of」可能混了透視與常識;「in front of」、「behind」更是
牽涉深度、物件大小、遮擋,跟 left / right 不是同一種 reference-frame 計算,合併起來會把
兩種機制的訊號攪在一起。所以只留 left / right,front / behind 明確排除;而且要求 scene
graph 的標注和 box 中心算出的 2D 關係一致,不一致的排除並報比例。論文上這意味著我們
分析的不是「GQA 的 spatial relation」,而是「GQA 中標注與預先登記的 2-D image-plane
criterion 一致的那個子集」,要這樣寫,不能寫成前者。

**沿關係軸的距離門檻是 2 個 patch,並做 robustness。** 這個門檻是任意的,得說理由並
檢查結果對它敏不敏感。理由:CLEVR 用 2/24 grid 的比例,換到 16×16 grid 是 1.3 個 patch,
取整為 2 保證兩個 role 之間至少隔一個非 role 的 patch,transplant 才不會同時碰到兩個
物件。robustness:1、2、3 個 patch 三個版本都跑,附錄報 H2、H3 的效應是否穩定。

**Same-as 只用 open query 的 245 題。** GQA 的 same 類多是 verify(「兩個是不是同色」),
沒有一個被選出的 target 可以分析。open query 的模板「和 Y 同色的 X 是什麼」天然不說
anchor 的顏色,正好滿足 H2 需要的「conditioning quantity 在問句裡未出現」。245 題篩完
可能只剩幾十題,所以 same-as 的 n 會很小,每張圖都標 n,不因為 n 小而放寬規則。

## 6. 四條主假設

每條四段:它是哪個主張、在 GQA 上怎麼量、什麼算通過、我們怎麼防止自己看到想看的。
層的自由度用預先指定的層窗處理;全曲線另附,但判準只看層窗。

### H1 — Referent selection generalizes

**主張。** 語言條件化在自然場景中把被指涉物件相對於非指涉物件重新組織。這是整套機制
的第一步,不成立的話後面三條都沒有基礎。

**怎麼量(observational,S_eligible)。** 取 c0 下 referent 的 patch mean 減 background (b)
的 patch mean,單位化,叫 V;這是「這個物件在沒有問句時的方向」。selection contrast 是
referent 在 V 上的投影從 c0 到 c1 的變化,減去 non-referent 在同一 V 上從 c0 到 c2 的
變化。CLEVR 上這個 contrast 在 SigLIP 從第 5 層開始為正並持續到第 11 層,所以層窗取
第 5 到 11 層的平均。

**通過。** contrast 的 CI 下界 > 0。輔助指標是 decoder attention 的 referent 對
background 每 patch 的比值,CI 下界 > 1;CLEVR 上這個比值是 60 倍,真實影像上不預設倍數,
只要求大於 1。

**防呆。** 用層窗平均而不是「某一層顯著」,避免十二層裡挑最好的一層。V 從 c0 算,
不含任何問句資訊,所以 contrast 不會因為 V 的定義本身偏向 referent。causal 對應
(activation patching 把 object 組換成 c2 的 token)列 secondary,在 S_correct 上做。

### H2 — Bound information becomes available downstream

**主張。** anchor 的任務相關性質,在 target 被選出之前,就已經可以從 anchor 之外的 patch
讀到。這是 relational 機制的核心:CLEVR 上 spatial 的 anchor 座標在第 7 層就能從
background 讀出,target 到第 9 層才變成必要。

**怎麼量。** spatial:從 background (b) 的 token 用 ridge regression 回歸 anchor 的 box
中心。same-as:從 background (b) 用 7-way probe 讀 anchor 的 colour。兩者的主量都是
問句帶來的增量,ΔR² = R²(c1) − R²(c0) 和 ΔAcc = Acc(c1) − Acc(c0),不是 c1 的絕對值。
k_condition 定義為第一個 ΔR²(或 ΔAcc)的 CI 下界超過 0.1 的層。k_target 是 causal 量,
在 S_correct 上做:把 target 的 patch 換成 c2 版本,第一個使 P(clean answer) 下降 ≥ 0.2
且 CI 不含 0 的層。

**通過。** ordering score = bootstrap 樣本中 k_condition < k_target 的比例,≥ 0.9。用
比例而不是「看曲線判斷早於」,是因為兩個 k 都有抽樣誤差,單看點估計的先後可以是
噪音。

**防呆。** 這條最容易被 leakage 騙。真實影像裡 background 本來就帶場景資訊:anchor 是
banana,pretrained 特徵可能從 category 就猜得到 yellow;anchor 是 stove,位置可能從廚房
的 layout 猜得到。這些都不是「anchor 的性質被搬運出去」。所以主量是 c1 減 c0 的增量,
沒有問句時就能讀到的部分被扣掉;另外做 label-shuffled probe 給 chance 分布、
category-matched split(train / test 的 anchor category 不重疊)、category-only baseline
(只用 category 預測同一 label,probe 必須顯著高於它)、image-grouped CV;same-as 另做
colour entropy 高的 category 子集。完整清單在附錄 B。

### H3 — Relation type determines the downstream computation

**主張。** attribute 的比較顯示稀疏的 head 層級因果定位;spatial 沒有同樣的稀疏定位。
CLEVR 上這是兩條 branch 最清楚的差異:same-as 關掉規則選出的 8 個 head 掉到 0.61
(SigLIP),spatial 關掉同樣選出的 head 完全不動。

**怎麼量(causal,S_correct)。** 用固定規則選 head:c1 − c0 的 SA attention 到 anchor
的變化,|Δ| ≥ 10× median 的 cell,上限 8 個。zero 掉,量 accuracy 下降 Δ_selected。
對照是同數量的 disjoint 隨機 head,3 個 seed,Δ_random。同時做 cumulative ablation:
同一個 ranking 累積 zero 掉 1、2、4、8、16、32 個 head,對照 matched-random 曲線。

**通過。** 判準是 interaction:(Δ_selected − Δ_random)_same 減
(Δ_selected − Δ_random)_spatial,CI 下界 > 0。不用「same-as 顯著、spatial 不顯著」,因為
A 顯著加 B 不顯著不等於 A 和 B 有顯著差異,這是常見但錯的推論,而它正好是我們核心故事
的關鍵一步。

**防呆與解讀限制。** spatial 的 null 有很多來源:選 head 的統計量可能不適合 spatial、
其他 head 可能補上、MLP 可能參與、可能是很多弱 head、positional embedding 的路徑可能
繞過 SA head、或介入強度不夠。所以 null 只能寫成「spatial does not exhibit the sparse
head-level localization detected by this selection criterion」;要寫「distributed」,
得 cumulative curve 顯示 spatial 要關掉很多 head 才逐步下降、same-as 前幾個就陡降。null
也不稱 evidence of absence,除非附 equivalence test,margin 預設 0.05。

### H4 — Relational selection returns to a shared referential state

**主張。** relational 題被選出的 target,取得和 direct query 的 referent 一樣的表徵。
這條讓兩條 branch 收斂回同一個讀出路徑,是「binding → 分支 → shared target selection」
故事的最後一段。

**怎麼量(observational,S_eligible)。** V 取自 direct 題,也就是 H1 的方向。relational
題裡 target、anchor、third object 的 patch mean 在 V 上的投影從 c0 到 c1 的變化,取第 9
到 11 層平均(CLEVR 上 target 的投影在這幾層上升)。

**通過。** target 減 third object 的差,CI 下界 > 0。比的是 target 對 third object,而不是
target 對零,因為問句本身會讓所有物件的投影都動,要看的是 target 有沒有被額外選出。

**防呆。** V 從 direct 題算,和 relational 題的 role 無關,避免用 relational 題自己的資料
定義方向再用同一批資料測。causal 對應(把 direct 題 c1 的 referent 狀態 transplant 到
relational 題的 target patch,看答案是否維持)列 secondary。

**四條的關係。** 都成立,故事是 binding → {attribute retrieval, feature-based reasoning,
spatial reasoning} → shared target selection。H1 不成立,整節改成「機制未在真實影像上
重現」,不再談後三條。H2 或 H4 不成立,撤回 relational 的那一段。H3 不成立,撤回
「關係種類決定計算形式」,兩條 branch 只保留順序主張。判準不因結果改。

## 7. 其他分析:為什麼降級,不是不做

以下全部照跑照報,但不進 confirmatory 判準,也不在看了結果後升級。降級的原因分三種:
它是主假設的 causal 對應或輔助量(secondary);它在 CLEVR 上就是 backbone-specific,
在 GQA 上沒有預測(exploratory);它的樣本是 synthetic 或 transfer,母體不同。

| 內容 | 對應 CLEVR | 地位 | 母體 |
|---|---|---|---|
| per-patch referent probe 起始層 | X21-C | secondary | eligible |
| attribute-specific direction 的分裂 | X21-E | secondary | eligible |
| activation patching(H1 的 causal 對應) | X21-D2 | secondary | correct |
| additive colour intervention 與對照 | X21-B | secondary | correct |
| 258 個單 head / 整層 ablation scan | X21-H | secondary | correct |
| attention 的 high-norm 控制 | X21-M2 | secondary | eligible |
| 行為:分離集 accuracy、c2 答案移動比例 | X22-H7d | secondary | eligible |
| candidate → anchor 的 SA attention | X22-H2b | secondary | eligible |
| positional embedding 翻轉 | X22-H7b | secondary | correct |
| background 帶 question type(SigLIP-specific) | X21-D3 | exploratory | correct |
| GCA write norm 無 role 選擇性 | X21-A5 | exploratory | eligible |
| GCA write mask | X22-H5 | exploratory | correct |
| absolute → relative write field(DINOv2 s42-specific) | X22-H7c | exploratory | eligible |
| synthetic c2 的所有分析 | — | secondary,不 pool | — |
| CLEVR 訓練的模型的所有分析 | — | secondary(transfer) | 自身的 eligible / correct |
| 距離門檻 1 / 2 / 3 patch | — | appendix | — |

留在 CLEVR、不移植的:template RSA 需要逐位置相同的 background template;成對渲染的
additivity 與 KMeans 需要同一物件有無 distractor 的孿生圖;無 GCA 模型需要 GQA 版本。
論文明說這三類為什麼只在 CLEVR 有。

## 8. 人工審查:為什麼要 blind、為什麼 50 題不夠

scene graph 有漏標,「唯一」可能不成立;box 可能框錯;synthetic 句可能不合語法。這些
只能靠人看。但如果研究者一邊看模型結果一邊判斷 scene graph 合不合理,就會傾向留下
支持故事的題、剔掉不支持的,這是 confirmation bias 最直接的入口。所以:審查者看不到模型
答案與任何機制結果;只檢查 role 指派、box 品質、關係有效性、唯一性、synthetic 句的語法
與語意。抽樣分層,direct、spatial、same-as 各 ≥ 50 題,總計 150 到 200,synthetic c2
另抽 50;一題型 50 題才能把錯誤率的 CI 壓到可用。至少一位審查者,兩位則報 Cohen's κ。
某題型錯誤率 > 10% 就全量複核,複核只能剔除不合規則的題,不能挑題。審查在任何機制
分析之前完成並凍結。

## 9. 執行順序與成本

GPU 一次一件,先驗證 CVD=0。步驟 0 完成、所有門檻凍結之後才排 GPU。

| 步驟 | 內容 | 資源 |
|---|---|---|
| 0 | 篩選、funnel、audit、三個門檻版本的母體;所有門檻凍結並提交 registry | CPU,一到兩天 |
| 1 | GQA 訓練的模型:c0–c3 cache;H1、H4、H2 的 observational 部分;secondary 的 observational 項 | GPU 約 1.5 小時 + CPU |
| 2 | H2 的 k_target transplant;patching;additive intervention | GPU 約 2 小時 |
| 3 | H3 head ablation 與 cumulative curve;head scan;pos-embed;write mask | GPU 約 2.5 小時 |
| 4 | CLEVR 訓練的模型:同上,依最小 n 規則 | GPU 約 3 小時 |
| 5 | bootstrap、圖、JSON、JOURNAL、registry X23 | CPU |

第二輪視第一輪結果:RefCOCO+ 圖片(本機 `data/refcocop`,19,992 張 COCO train2014 圖、
box、不含位置詞的 expression)以 GQA 模板合成題目,做跨 dataset 的第二次驗證;material /
size 被問 attribute 的變體對應 CLEVR 的 X21-G/J。

## 10. 規則,以及每條規則防的是什麼

- **所有門檻、層窗、最小 n、equivalence margin 在看任何結果前凍結。** 防的是看了資料
  再調門檻。
- **篩選規則的任何放寬是 X23-v2,原結果保留不覆蓋。** 防的是「放寬之後結果變好,
  原版消失」。
- **不稱 independent replication,用「held-out items, not checkpoint-independent」。**
  防的是把同一 model 同一 dataset 上的重複探索說成獨立驗證。
- **observational 不稱 causal。** probe 讀得到不等於被使用;只有介入改變答案才是。
- **null 不稱 evidence of absence,除非附 equivalence 或 power analysis。** 沒偵測到
  不等於不存在。
- **報 excluded-item 數量與排除原因。** 讀者要能重建 funnel。
- **結果無論方向全報;secondary 不升級為 primary。** 防的是事後挑。

## 11. 風險

有 relate 的題 accuracy 約 0.5,S_correct 可能不足,coverage 照報,不足就只報
observational。scene graph 漏標,由 audit 的錯誤率與全量複核處理。natural pair 可能
稀少,synthetic c2 留在 secondary,不補進 primary。真實影像的物件大或遮擋,由 IoU = 0
與 ≤ 30% 影像的規則處理,另報物件大小分層。只有一個真實影像 backbone,跨 backbone
主張留在 CLEVR。checkpoint 非 independent,§4 的聲明進論文。

## 12. 實作與記錄

全部放在 `scripts/analysis/patch_language_condition.py`,不新開分析腳本:`assign_roles`
加 GQA 路徑(scene graph box、program object id、幾何規則、natural pair 配對、synthetic
c2 生成與標記);`--gqa-filter` 做 CPU 篩選,輸出 records、funnel(含每步排除原因與
數量)、audit 抽樣清單;新增 `--h8-cumulative`;其餘 flags 沿用。輸出到
`outputs/analysis/patch_language_condition/gqa_{direct,spatial,same}/` 與
`gqa_{direct,spatial}_clevrmodel/`。registry 新增 X23;JOURNAL 在步驟 0 與每個 GPU 步驟後
追加;結果先給使用者,網站與 RESULTS 待使用者決定。

---

## 附錄 A:S_eligible 的篩選規則(依序套用,每步記數量與排除原因)

共用:
1. anchor(direct 為 referent)的名稱在 scene graph 中唯一;relational 題 anchor 與
   target 的 category 不同(否則「the cup left of the cup」的 role 讀不清)。
2. box 幾何(patch 座標):role 之間 IoU = 0;每個 role ≥ 4 個 patch;任一 role ≤ 30%
   影像(太大會讓 background 太少)。
3. 三個 role 的答案值都在 GQA 訓練的模型的答案詞彙內。CLEVR 訓練的模型另加「答案在
   八種顏色內」,形成子集,分開報。

Direct:
4. natural pair(primary):同圖另有一題 GQA 原生 direct query,指向不同物件、問同一
   attribute、答案不同,互為 c1 / c2。
5. synthetic c2(secondary,不 pool):以 scene graph 改寫 referent 名稱、答案取自 scene
   graph;附 RoBERTa embedding 距離分布、分開報、audit。

Spatial:
6. relation 只取 to the left of / to the right of。
7. box 中心算出的 left / right 與 scene graph 標注一致,不一致者排除並報比例。
8. 沿關係軸中心距 ≥ 2 個 patch;1 / 2 / 3 三版。
9. c2:anchor 另一側恰有一個與 target 同 category、答案不同、通過第 2 條的物件(即
   third object)。c3:另有一物件對 target 唯一成立同一 relation、名稱唯一、category 與
   anchor / target 皆不同。至少一個可構造。
10. 分離集:target 落在方向詞相反的影像半邊,另記子集。

Same-as(open query 245 題):
11. 問句不含 anchor 的 colour(模板天然滿足);場景中恰一個其他物件與 anchor 同色
    (target);另有一個不同色、與 target 同 category 的物件(third object;若無則取任一
    不同色物件,分開報)。c2 = 把 target 當 anchor 命名。

## 附錄 B:probe 的 leakage 控制(H2 與所有 secondary probe 通用)

1. 主量是 c1 對 c0 的差(ΔR²、ΔAcc),不單獨報 R²(c1)。
2. label-shuffled probe(within image)給 chance 分布。
3. category-matched split:train / test 的 anchor category 不重疊。
4. category-only baseline:只用 anchor 的 category 預測同一 label,probe 必須顯著高於它。
5. image-grouped CV:GroupKFold(5) by image。
6. same-as 的 colour probe 另限制到 colour entropy 高的 category(scene graph 中該
   category 的顏色分布 entropy ≥ 1.5 bit)做子集分析。

## 附錄 C:控制組一覽

c0 無問句;transplant 與 patching 的 clean-self 每格 1.00;disjoint 隨機 head(3 seed);
matched-random cumulative curve;norm-matched random vector;random background rows;
label shuffle;category-only baseline;equivalence margin 0.05。

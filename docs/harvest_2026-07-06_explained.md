# 2026-07-06 收成批次解說

> 給沒有跟到當天工作的讀者。本檔逐項說明 2026-07-06 落地的四項結果——**如何量測、
> 數字說了什麼、對論文主張建立了什麼、又沒有建立什麼**。所有數字均取自
> `RESULTS.md`（§6、§11、§13、§14）與 `JOURNAL.md`（07-06 Today's Progress），未經
> artifact 佐證者不入本檔；正式的論文用整合報告另見
> `docs/substrate_fixation_report.md`，本檔不重複其三角敘事，只在需要時指向它。
>
> **命名慣例（v2）**：**Grounding（語言 grounding）** 指整個語言 conditioning（語言
> 條件化）機制——把凍結的視覺表徵，依問題文字轉成能作組合式推理的表徵；其兩個階段依序
> 為 **Binding（綁定：把描述中的屬性綁到被指涉的那個物體）** 與 **Retrieval（讀出：
> 讀出被詢問的屬性值）**。語言 conditioning 透過 **gated cross-attention（GCA，帶閘門的
> 交叉注意力）** 注入問題。這些主張的預先登記版本在 `docs/paper_v2_outline.md`，代號
> A1／A2／A3；下文每項結果都對照其對應主張判讀。

---

## 這一批是什麼

2026-07-06 收掉的是「基底與凝視」證據鏈的最後幾塊拼圖，外加兩項對論文邊界的界定實驗。
核心論證是一個 **凝視三角（fixation triangle）**：預訓練視覺骨幹的原始表徵是否已逐物體
編碼屬性（資訊在不在）、拿掉語言 conditioning 後模型如何失敗（選取斷不斷）、補回完整
grounding 後是否穩健（選取修不修得好）。三角的前兩腿在稍早幾天已落地，這一天補上的是：

- **(a)** 第四個、也是最後一個原始骨幹的探測結果（E8 的 MAE），讓三角的「資訊在不在」這
  一腿站在全部四個骨幹上；
- **(b)** 一項在文字生圖（text-to-image, T2I）模型上檢驗「此機制是否只是 CLEVR 訓練
  artifact」的 zero-shot 實驗，得到**有界的負向結論**；
- **(c)** 一則本地 Flamingo 基線的**量測更正**——先前的數字是量測工具的 bug，不是模型
  的性質；
- **(d)** 撰寫整合報告時被標記出來的**兩處與預先登記的偏離**，需要專案負責人裁決。

以下逐項說明。

---

## (a) E8 — MAE 原始基底探測完成：小的基底差距，大的下游差距

**這是在量什麼。** E8（raw-substrate probing，原始基底探測）問的是一個 claim A1 層級的
問題：**在完全沒有語言 conditioning、也沒有任何 VQA 任務訓練的情況下，預訓練視覺骨幹的
表徵本身，是否已經逐物體（per-object）地把每個物體的四項屬性（顏色 color／材質
material／形狀 shape／大小 size）編碼進去了？** 若答案為是，那基底缺的就不是「屬性資訊」，
而是「在多物體場景中選對那個被描述物體」的能力——這正是 grounding 要補的那一步。

**怎麼量的。** 取四個預訓練骨幹——DINOv2、SigLIP、監督式 ViT（sup-ViT）、MAE——把
GCA 的 tanh 閘門歸零（zero-gated，使前向傳遞退化成純粹的預訓練 ViT，問題文字完全不進入
主幹）。資料是 300 個多物體 CLEVR 場景、共 1,917 個物體。對每個物體，依其像素座標
（`pixel_coords`）在特徵網格上取 3×3 patch 的窗作平均池化，然後在每一個 ViT block 上用
5 折邏輯迴歸（logistic regression）去預測該物體的四項屬性，回報各骨幹在所有 block 上的
**峰值**逐物體可解碼率。這一天落地的是最後一個骨幹 MAE（14:38）；至此 E8 四個骨幹全數
完成。Artifact：`outputs/analysis/raw_backbone_probe/<backbone>/`。RESULTS.md §11。

**數字。** 逐物體屬性可解碼率（各 block 峰值，括號內為達峰的 block）：

| 骨幹 | color | material | shape | size |
|---|---|---|---|---|
| DINOv2 | 0.966 (B7) | 0.987 (B10) | 0.986 (B11) | 0.997 (B8) |
| SigLIP | 0.932 (B2) | 0.953 (B8) | 0.951 (B8) | 0.983 (B3) |
| sup-ViT | 0.938 (B2) | 0.934 (B2) | 0.924 (B7) | 0.983 (B6) |
| MAE | 0.920 (B4) | 0.914 (B6) | 0.914 (B9) | 0.979 (B9) |

四個骨幹、四項屬性全部落在 **0.91–1.00**。這確立 claim A1.2：**多物體場景中，逐物體的
屬性資訊在語言 conditioning 之前、在任何任務訓練之前，就已存在於原始基底裡**——包括
MAE。基底缺的不是資訊，是選取。

**MAE 為什麼逼著「較弱基底」的說法要改寫。** 先看兩組排序。基底可解碼率的排序是
DINOv2 > SigLIP ≈ sup-ViT > MAE，這**恰好**與下游 VQA 準確率的排序一致（同樣 MAE 墊底）。
單看排序，很容易把 MAE 的下游失敗歸因於「它的基底沒把屬性編碼好」。但兩處差距的**尺度**
完全不相稱：

- **基底層面**，MAE 與 DINOv2 的差距很小——例如 color 是 0.920 對 0.966，其餘屬性也都
  只差幾個百分點。
- **下游 VQA 層面**，MAE 與 DINOv2 的差距很大——**0.748 對 0.924**，相差約 17 個百分點。

一個小到只有幾個百分點的基底差距，數學上撐不起一個 17 點的下游差距。**峰值逐物體可解碼率
因此無法單獨解釋 MAE 的失敗。** 這與 §5 的逐題型分析一致：MAE 的缺陷集中在**雙指涉物綁定**
（EqAttr，equal_attribute，比較兩個物體的同一屬性是否相等——MAE 在此崩到 0.586），而不是
單物體的屬性讀出（QryAttr 幾乎正常，0.921）。

**對 claim A3 的意義。** claim A3.1 原本的措辭是「像素重建式預訓練（MAE）是較弱的基底」。
E8 迫使這句話**精確重述**為：MAE 弱的是**可供 Binding 使用的結構**，而不是屬性資訊本身。
MAE 保有屬性資訊，只是這些資訊較不容易被 grounding 的選取／綁定步驟拿去用，尤其在需要
同時綁定並比較多個物體時。這不是把 A3 削弱，而是把它釘到正確的機制層級——並且在 A3
（基底品質）與 A5（失敗模式，MAE 的 EqAttr 崩潰是 H1 形狀的雙指涉物失敗）之間架起一座橋。

---

## (b) T2I timestep sweep — zero-shot grounding 的有界負向結論（PixArt-Σ）

**這是在回應什麼質疑。** 對 claim A2 有一個自然的反駁：Binding→Retrieval 的 grounding
會不會只是我們在 CLEVR 上做 VQA 訓練「訓」出來的 artifact，而非語言 conditioning 的通用
性質？為此，`docs/t2i_experiment_design.md` 預先登記了一項 zero-shot 檢驗：取 PixArt-Σ
（一個 DiT 骨幹 + 凍結 T5-XXL 文字編碼器、**只為文字生圖訓練、從未見過任何 VQA 監督**的
擴散模型），看看在它的 cross-attention 裡，是否**未經任何任務訓練**就自發浮現可量測的
binding 訊號。

**預先登記的雙條件標準。** 判定「有訊號」必須**兩項同時**成立：
- **(a)** referent-local probe（指涉物局部探測）在中／晚層 block 顯著高於 majority
  baseline（多數類基線，即永遠猜最常見答案的準確率）；且
- **(b)** column-normalized 的 cross-attention 質量落在指涉物 3×3 窗上時，顯著高於 chance
  （隨機水準 9/1024 ≈ 0.0088）。

用「雙條件」而非任一單條件，是為了讓負向結論穩健——一項偶然超標不足以翻案。

**為什麼加測 timestep 261 與 400。** 設計原本只登記單一擴散 timestep t=100。落地時把它擴成
t=100／261／400 的 **timestep sweep（掃描）**，作為穩健性檢查：萬一 t=100 太接近乾淨影像、
訊號被抹掉，換幾個雜訊程度不同的 timestep 也許能撈到。（注意：這項擴充本身超出了預先登記的
設計，見下文 (d)。）Artifact：`outputs/analysis/t2i_pixart/`（t=100，n=300/類）與
`outputs/analysis/t2i_pixart_t{261,400}/`（後兩者 n=150/類）。RESULTS.md §13。

**數字。** referent-local probe 最佳 block 準確率（括號內為 majority baseline）：

| t | color | material | shape | size |
|---|---|---|---|---|
| 100 | 0.169 (0.152) | 0.554 (0.566) | 0.421 (0.361) | 0.556 (0.527) |
| 261 | 0.168 (0.159) | 0.689 (0.571) | 0.542 (0.400) | 0.556 (0.556) |
| 400 | 0.230 (0.159) | 0.594 (0.571) | 0.500 (0.400) | 0.632 (0.556) |

cross-attention 指涉物窗質量（各 block 峰值）：t=100 為 0.0095–0.0096、t=261 為
0.0097–0.0099、t=400 為 0.0112–0.0117，換算成 chance 的倍數只有 **1.27–1.33 倍**。

**結果怎麼讀。**
- **probe 這一條件不成立**：對 majority baseline 的超出是**零散**的——散布在不同 block、
  不同屬性，沒有一致的中／晚層結構（例如 t=261 material 0.689 對 0.571 看似超標，但 color
  幾乎貼著基線、其他 timestep 又換一組屬性冒頭；且 t=261/400 每類樣本只有 n=84–133，
  超標幅度落在雜訊帶內）。
- **cross-attention 這一條件更是遠不成立**：定位訊號**從未離開 1.0–1.3× chance 這個帶**。
- **唯一有結構的殘跡**：在 t=400，三個問題類別（color／material／shape 對應的 attr_query
  類別）的 cross-attention **都在同一個 block（B6）達峰**。三類同峰於一處是有結構的跡象，
  但其絕對量（約 1.3× chance）遠低於任何可用的 binding 訊號，只作為一條註記，不足以翻案。

**判讀：在所有受測 timestep 上，依預先登記的雙條件標準皆為負向。** 而且這個負向是**有界**
的：預先登記時就已寫下 domain-mismatch（領域不匹配）警語——CLEVR 的**問句**不是文字生圖
模型習慣的**caption（描述句）**，對 PixArt-Σ 的文字編碼器構成分佈不匹配。因此這個負向
**無法區分**兩種可能：「機制根本不存在於預訓練 T2I」，還是「機制在，只是被 prompt 分佈
不匹配遮住了」。受此約束，唯一能下的結論是：

> **在問題 prompt 下、於任何受測 timestep，PixArt-Σ 的 cross-attention 中都沒有浮現
> zero-shot binding；本文所量測的 grounding 機制，需要任務訓練才會出現。**

**為什麼這是「定位」而非「傷害」論文的貢獻。** 因為 domain-mismatch 警語是**事先**登記的，
這個負向不會被拿去當成「機制不存在」的主張（那會與論文相衝突），也不會被誇大成「T2I 一定
不行」。它反而**正面框定**了論文的貢獻：論文要展示的正是 **VQA 訓練如何從預訓練基底中
誘發並強化這個機制**——如果機制在別處俯拾即是，這個貢獻就不值錢；證明了它不會自發出現，
「訓練把它引出來」這件事才有份量。這是有界的負向結果替論文**定位**貢獻的典型例子。

---

## (c) Flamingo 本地基線 — 量測更正：先前的數字是量測工具的 bug

**背景。** `clevr_flamingo_dinov2_early_s42` 是一個本地訓練的 Flamingo 式基線，原本設計要
訓滿 16 epoch，實際停在 epoch 4/16（當時被降優先級）。它本可作為 I2T（image-to-text，
影像轉文字）路線上「機制是否為訓練 artifact」的另一條證據。我們在它上面跑了 E7 的
add-object 幻覺測試。

**bug 是什麼。** E7 的第一次執行（無後綴的 `add_object_eval_clevr_flamingo_dinov2_early_s42.json`）
**全部無效**——這是量測工具（harness）的 bug，不是模型的量測結果。具體是一個
大小寫不匹配：`generate_answer` 把解碼結果轉成**小寫**，而 adapter 卻用**大寫**的
`"Answer:"` 去切分答案。結果切不出答案，每一筆記錄的預測都變成 prompt 的回音
（prompt echo）`'question:'`，於是所有屬性的準確率都是 0。**那些 0 是量測工具的產物，
不是模型的性質。**

**修正與同源 bug。** 修正落在 `add_object_eval_flamingo.py`。更關鍵的是——**訓練腳本
`train_flamingo_clevr.py:evaluate` 裡有一模一樣的潛伏 bug**，也一併修好了。若不修，之後
計畫中的 Flamingo 重訓會**永遠**回報驗證準確率為 0（訓練其實在進步，儀表卻永遠讀 0）。
依 never-overwrite（絕不覆寫）政策，無效的 JSON 原地保留；修正後的結果是 `*_fixed.json`。
RESULTS.md §14。

**修正後的數字說了什麼。** `..._fixed.json` 的 E7 base 準確率：color 0.13／material 0.54／
shape 0.32／size 0.44。對照各屬性的類別數（8／2／3／2 類），這些全是 **chance（隨機）
水準**，而且輸出**近乎退化**（color 有 98% 全答 "yellow"、size 有 100% 全答 "small"，
material／shape 則在兩個值之間擲硬幣）。預測**仍落在被詢問屬性的答案空間內**（所以那顆
LLM 讀得懂問題的類型），但**視覺路徑目前毫無貢獻**——4/16-epoch 顯然訓練不足。這把
「此 checkpoint 只具定性地位」從一句定性判斷，變成一個**定量**確認。

**為什麼幻覺／誘餌指標在此失去意義。** E7 的核心指標是誘餌捕獲率（hallucination_rate）與
`bait_share_of_errors`（誘餌佔錯誤的比例），用來判斷模型是否「凝視被描述物體、不被誘餌
勾走」。但當模型的預測是 chance 水準、近乎常數時，這些指標變成**無意義的 artifact**：對
二元屬性（material、size 只有兩個值），任何一個錯誤在建構上**必然是誘餌形狀的**——因為
「不是正確值」就只剩「誘餌值」這一個選項。於是 material 的 bait_share 0.96、size 的 1.00
不代表任何「被誘餌勾走」的實質，只是二元 + 亂猜的必然結果。**在這個 regime 下，E7 對
Flamingo 不可判讀。** Flamingo 的 E7 這一腿要等重訓（規劃為 no-LoRA + 預算特徵的方案，
待專案負責人放行）之後才能納入。

---

## (d) 兩處與預先登記的偏離 — 需要專案負責人裁決

撰寫整合報告（`docs/substrate_fixation_report.md`）時，比對預先登記文件
（`docs/paper_v2_outline.md`）後標記出兩處**執行與登記不符**的地方。兩處都不改變任何結論
的方向，但為了論文的預先登記誠信，需要決定「camera-ready 時修改登記還是修改措辭」。

**偏離一：E7 頭條指標的替換。**
- 預先登記（A1.3）指定的頭條指標是 `bait_share_of_errors`（誘餌佔錯誤的比例），理由是它
  能把「幻覺」從「一般分佈位移」中隔離出來。
- 實際整合報告改用 `hallucination_rate`（誘餌捕獲率）作為凝視三角的主軸讀數，只把
  `bait_share_of_errors` 降為輔助的「錯誤確為誘餌形狀」的形狀確認。
- **統計理由**：`bait_share_of_errors` 在**錯誤稀有時不穩定**——完整訓練模型好到幾乎不出錯，
  於是分母極小，例如 size 只有少數幾個錯誤就得出 bait_share 1.00，material 甚至 0 個錯誤而
  完全無定義。一個在零錯誤處爆掉的比值，撐不起「捕獲率」這個量。相對地，hallucination_rate
  提供了三角所需的因果對比（+CA 模型 0–7% 對 −CA 模型 24–59%）。
- **待決**：camera-ready 要嘛修改 outline 的登記（承認頭條指標改為 hallucination_rate），
  要嘛修改報告措辭讓它與原登記對齊。實質結論不受影響。

**偏離二：T2I sweep 超出登記的單一 timestep 設計。**
- 預先登記只登記**單一 timestep t=100、每類 300 題**。
- 實際執行擴成 **t=100／261／400 的 sweep，且 t=261/400 每類只有 n=150 題**（樣本數
  縮減）。
- 這是對預登記設計的**擴充**。它讓負向結論更穩健（多個雜訊程度都看不到訊號），但嚴格
  而言，t=261/400 屬於預先登記範圍之外的探索，且縮減的 n 使其超標判定的統計檢定力較低。
- **待決**：camera-ready 是把 sweep 補進登記（承認這是預登記後的穩健性擴充），還是只報
  登記的 t=100、把 sweep 列為附錄的探索性檢查。負向判讀本身不變。

兩處偏離都已記入 `JOURNAL.md` 的 TODO，也寫進 `substrate_fixation_report.md` 的「限制」節。

---

## 當天的兩項圖表交付（支援性 artifact）

除了上述四項結果，當天還產出兩份圖：

- **A/B/C 的 CA-share 對比圖（E9）**：`abc_localization.py` 新增輸出
  `outputs/analysis/abc_localization/clevr_dinov2_decoder1l_scratch/abc_contrast.png`——
  每個屬性一組 cross-attention 佔比（CA-share）的長條，並畫出 0.5 的「無偏好」參考線，
  讓「文字側擾動經 cross-attention 復原（A>B>C 的梯度）」一眼可見（shape 的 C 情形最低，
  0.199）。底層 JSON 經驗證位元相同，只是把圖補上（RESULTS.md §6 的圖註）。
- **v2 標籤重繪（E10）**：`grounding_manipulation.py` 新增 `--replot-from`，用已存的 JSON
  重繪圖（不動 GPU），把 Retrieval 階段的子訊號標籤更新為 **Retrieval (object)** 與
  **Retrieval (answer)**，輸出到**新目錄**
  `outputs/analysis/grounding_manipulation/clevr_dinov2_decoder1l_scratch_v2labels/`，
  原圖原地保留未動。

兩者都是支援性 artifact——不引入新數字，只把既有結果以正確的 v2 命名與對比視覺化呈現。

---

## 小結

這一批把「基底與凝視」的證據鏈補完，並替論文的邊界立了兩根樁。(a) E8 讓凝視三角的「資訊
在不在」這一腿站在全部四個骨幹上，並把 MAE 的故事從「基底較弱」精修為「可供 Binding 使用
的結構較弱、而非缺屬性資訊」——這是對 claim A3 的精確化，不是削弱。(b) T2I sweep 是一個
**事先設好護欄**的負向結果：它不宣稱機制不存在，而是正面定位論文的貢獻在於「VQA 訓練把
機制從預訓練基底中引出來」。(c) Flamingo 的更正是一堂量測教訓——先前的全零準確率是
harness 的大小寫 bug，不是模型；修正後看到的是 chance 水準的訓練不足模型，E7 對它暫不可
判讀，同源 bug 也已在訓練腳本裡一併修好以防重訓時儀表永遠讀 0。(d) 兩處預先登記偏離
（E7 頭條指標、T2I sweep 範圍）都不影響結論方向，但需要專案負責人在 camera-ready 前決定
是修改登記還是修改措辭。正式的整合敘事見 `docs/substrate_fixation_report.md`。

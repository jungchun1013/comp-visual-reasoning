# 2026-07-06 實驗結果解說

> 本檔說明 2026-07-06 落地的四項結果，對象為未跟到當天工作的讀者。每項結果說明其
> 量測方法、數字，以及對論文主張建立了什麼、沒有建立什麼。所有數字取自
> `RESULTS.md`（§6、§11、§13、§14）與 `JOURNAL.md`（07-06 的 Today's Progress），
> 未經 artifact 佐證者不入本檔。正式的論文用整合報告為
> `docs/substrate_fixation_report.md`；本檔不重複其內容，僅在需要處指向它。
>
> **命名慣例（v2）**：**Grounding（語言 grounding）** 指整個語言 conditioning（語言
> 條件化）機制——依問題文字對凍結的視覺表徵作條件化，使其可用於組合式推理；其兩個階段
> 依序為 **Binding（綁定：把描述中的屬性綁到被指涉的那個物體）** 與 **Retrieval（讀出：
> 讀出被詢問的屬性值）**。語言 conditioning 透過 **gated cross-attention（GCA，帶閘門的
> 交叉注意力）** 注入問題。這些主張的預先登記版本在 `docs/paper_v2_outline.md`，代號
> A1／A2／A3；下文每項結果都對照其對應主張判讀。

---

## 本批結果的範圍

`RESULTS.md` 以「fixation triangle」為名記錄一個三部分論證；該論證由三項獨立量測構成：
**證據一（E8）**——未經語言 conditioning 的原始骨幹表徵是否已逐物體編碼屬性；
**證據二（E5/E7-on-nogca）**——移除 cross-attention 的訓練模型如何失敗；
**證據三（E7-trained）**——具備完整 grounding 的訓練模型在相同刺激下是否穩健。
證據二與證據三在此前數日已完成。2026-07-06 完成的項目為：

- **(a)** E8 的第四個、也是最後一個骨幹（MAE）的探測結果，使證據一涵蓋全部四個骨幹；
- **(b)** 一項在文字生圖（text-to-image, T2I）模型上檢驗「此機制是否只是 CLEVR 訓練
  artifact」的 zero-shot 實驗，結論為有界的負向（bounded negative）；
- **(c)** 一則本地 Flamingo 基線的量測更正——先前的數字由量測工具的 bug 產生，不反映
  模型的性質；
- **(d)** 撰寫整合報告時標記出的兩處與預先登記的偏離，需要專案負責人裁決。

以下逐項說明。

---

## (a) E8 — MAE 原始基底探測完成：基底差距小，下游差距大

**這是在量什麼。** E8（raw-substrate probing，原始基底探測）檢驗一個 claim A1 層級的
問題：**在完全沒有語言 conditioning、也沒有任何 VQA 任務訓練的情況下，預訓練視覺骨幹的
表徵本身，是否已經逐物體（per-object）地編碼每個物體的四項屬性（顏色 color／材質
material／形狀 shape／大小 size）？** 若答案為是，則基底缺的不是屬性資訊，而是在多物體
場景中選出被描述物體的能力——後者正是 grounding 要補的步驟。

**怎麼量的。** 取四個預訓練骨幹——DINOv2、SigLIP、監督式 ViT（sup-ViT）、MAE——把
GCA 的 tanh 閘門歸零（zero-gated，使前向傳遞等同於純粹的預訓練 ViT，問題文字完全不進入
主幹）。資料是 300 個多物體 CLEVR 場景、共 1,917 個物體。對每個物體，依其像素座標
（`pixel_coords`）在特徵網格上取 3×3 patch 的窗作平均池化，然後在每一個 ViT block 上用
5 折邏輯迴歸（logistic regression）預測該物體的四項屬性，回報各骨幹在所有 block 上的
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

**為何此結果要求改寫「MAE 是較弱基底」的表述。** 先看兩組排序。基底可解碼率的排序是
DINOv2 > SigLIP ≈ sup-ViT > MAE，與下游 VQA 準確率的排序一致（同樣 MAE 墊底）。單看
排序，容易把 MAE 的下游失敗歸因於「基底沒有把屬性編碼好」。但兩處差距的**尺度**不相稱：

- **基底層面**，MAE 與 DINOv2 的差距小——例如 color 是 0.920 對 0.966，其餘屬性也都
  只差幾個百分點。
- **下游 VQA 層面**，MAE 與 DINOv2 的差距大——**0.748 對 0.924**，相差約 17 個百分點。

幾個百分點的基底差距不足以解釋 17 個百分點的下游差距。**峰值逐物體可解碼率因此無法單獨
解釋 MAE 的失敗。** 這與 §5 的逐題型分析一致：MAE 的缺陷集中在**雙指涉物綁定**（EqAttr，
equal_attribute，判斷兩個物體的同一屬性是否相等——MAE 在此降到 0.586），而不是單物體的
屬性讀出（QryAttr 接近正常，0.921）。

**對 claim A3 的意義。** claim A3.1 原本的措辭是「像素重建式預訓練（MAE）是較弱的基底」。
E8 要求把這句話**精確重述**為：MAE 弱的是**可供 Binding 使用的結構**，而不是屬性資訊
本身。MAE 保有屬性資訊，但這些資訊較不易被 grounding 的選取／綁定步驟利用，尤其在需要
同時綁定並比較多個物體時。此改寫並未削弱 A3，而是把它限定在正確的機制層級；同時它連結了
A3（基底品質）與 A5（失敗模式——MAE 的 EqAttr 下降屬於 H1 形狀的雙指涉物失敗）兩項主張
的證據。

---

## (b) T2I timestep sweep — zero-shot grounding 的有界負向結論（PixArt-Σ）

**這是在回應什麼質疑。** 對 claim A2 有一個自然的反駁：Binding→Retrieval 的 grounding
可能只是在 CLEVR 上做 VQA 訓練產生的 artifact，而非語言 conditioning 的通用性質。為此，
`docs/t2i_experiment_design.md` 預先登記了一項 zero-shot 檢驗：取 PixArt-Σ（一個 DiT
骨幹 + 凍結 T5-XXL 文字編碼器、**只為文字生圖訓練、從未見過任何 VQA 監督**的擴散模型），
檢驗其 cross-attention 是否在未經任何任務訓練的情況下出現可量測的 binding 訊號。

**預先登記的雙條件標準。** 判定「有訊號」必須**兩項同時**成立：

- **(a)** referent-local probe（指涉物局部探測）在中／晚層 block 顯著高於 majority
  baseline（多數類基線，即永遠猜最常見答案的準確率）；且
- **(b)** column-normalized 的 cross-attention 質量落在指涉物 3×3 窗上時，顯著高於 chance
  （隨機水準 9/1024 ≈ 0.0088）。

採雙條件而非任一單條件，是為了使負向結論穩健——單一項偶然超標不足以構成正向判定。

**為什麼加測 timestep 261 與 400。** 設計原本只登記單一擴散 timestep t=100。實際執行
擴成 t=100／261／400 的 timestep sweep（掃描），作為穩健性檢查：若訊號僅在特定雜訊程度
（特定 timestep）下可量測，只測 t=100 會遺漏它。（此擴充本身超出預先登記的設計，見下文
(d)。）Artifact：`outputs/analysis/t2i_pixart/`（t=100，n=300/類）與
`outputs/analysis/t2i_pixart_t{261,400}/`（後兩者 n=150/類）。RESULTS.md §13。

**數字。** referent-local probe 最佳 block 準確率（括號內為 majority baseline）：

| t | color | material | shape | size |
|---|---|---|---|---|
| 100 | 0.169 (0.152) | 0.554 (0.566) | 0.421 (0.361) | 0.556 (0.527) |
| 261 | 0.168 (0.159) | 0.689 (0.571) | 0.542 (0.400) | 0.556 (0.556) |
| 400 | 0.230 (0.159) | 0.594 (0.571) | 0.500 (0.400) | 0.632 (0.556) |

cross-attention 指涉物窗質量（各 block 峰值）：t=100 為 0.0095–0.0096、t=261 為
0.0097–0.0099、t=400 為 0.0112–0.0117，即 chance 的 **1.27–1.33 倍**。

**結果怎麼讀。**

- **probe 條件不成立**：對 majority baseline 的超出是**零散**的——分布在不同 block、
  不同屬性，沒有一致的中／晚層結構。例如 t=261 的 material 為 0.689 對 0.571，但同一
  timestep 的 color 與基線幾乎相同；在不同 timestep 超出基線的屬性又不相同。且 t=261/400
  每類樣本僅 n=84–133，超出幅度在雜訊範圍內。
- **cross-attention 條件不成立**：定位訊號在所有 timestep、所有 block 上始終落在 chance
  的 1.0–1.3 倍範圍內。
- **唯一有結構的殘跡**：在 t=400，三個問題類別的 cross-attention **都在同一個 block
  （B6）達峰**。三類同峰於一處具有結構性，但其絕對量（約 1.3× chance）遠低於任何可用的
  binding 訊號，只作為註記，不足以改變負向判定。

**判讀：在所有受測 timestep 上，依預先登記的雙條件標準皆為負向。** 此負向是**有界**的：
預先登記時已寫下 domain-mismatch（領域不匹配）警語——CLEVR 的**問句**不是文字生圖模型
訓練所見的**caption（描述句）**，對 PixArt-Σ 的文字編碼器構成分佈不匹配。因此此負向結果
**無法區分**兩種可能：機制不存在於預訓練 T2I 模型，或機制存在但因 prompt 分佈不匹配而
未被量到。受此約束，能下的結論限於：

> **在問題 prompt 下、於任何受測 timestep，PixArt-Σ 的 cross-attention 中都未出現
> zero-shot binding；本文所量測的 grounding 機制，需要任務訓練才會出現。**

**此負向結果對論文貢獻的意義。** 因為 domain-mismatch 警語是**事先**登記的，此負向不會
被當成「機制不存在」的主張（該主張超出資料所能支持的範圍），也不會被擴大為對 T2I 模型的
一般性否定。它界定了論文的貢獻：論文要展示的是 **VQA 訓練如何從預訓練基底中誘發並強化
這個機制**。若該機制普遍自發存在於預訓練模型中，此貢獻的重要性會降低；證明它在此設定下
不自發出現，則「訓練誘發此機制」構成論文的實質貢獻。因此這個負向結果界定而非削弱論文的
貢獻。

---

## (c) Flamingo 本地基線 — 量測更正：先前的數字由量測工具的 bug 產生

**背景。** `clevr_flamingo_dinov2_early_s42` 是一個本地訓練的 Flamingo 式基線，原設計
訓滿 16 epoch，實際停在 epoch 4/16（當時被降低優先級）。它原可作為 I2T（image-to-text，
影像轉文字）路線上「機制是否為訓練 artifact」的另一條證據。我們在它上面執行了 E7 的
add-object 幻覺測試。

**bug 是什麼。** E7 的第一次執行（無後綴的
`add_object_eval_clevr_flamingo_dinov2_early_s42.json`）**全部無效**——原因是量測工具
（harness）的 bug，不是模型的量測結果。具體是一處大小寫不匹配：`generate_answer` 把
解碼結果轉成**小寫**，而 adapter 用**大寫**的 `"Answer:"` 切分答案。切分因此永遠失敗，
每一筆記錄的預測都成為 prompt 的回音（prompt echo）`'question:'`，所有屬性的準確率
記為 0。**這些 0 是量測工具的產物，不是模型的性質。**

**修正與同源 bug。** 修正落在 `add_object_eval_flamingo.py`。此外，**訓練腳本
`train_flamingo_clevr.py:evaluate` 含有相同的潛伏 bug**，也一併修正。若不修正，計畫中的
Flamingo 重訓在整個訓練過程中回報的驗證準確率將恆為 0。依 never-overwrite（絕不覆寫）
政策，無效的 JSON 原地保留；修正後的結果為 `*_fixed.json`。RESULTS.md §14。

**修正後的數字說了什麼。** `..._fixed.json` 的 E7 base 準確率：color 0.13／material 0.54／
shape 0.32／size 0.44。對照各屬性的類別數（8／2／3／2 類），這些全部是 **chance（隨機）
水準**，且輸出**近乎退化**（color 有 98% 全答 "yellow"、size 有 100% 全答 "small"，
material／shape 在兩個值之間接近隨機分佈）。預測**落在被詢問屬性的答案空間內**（語言
模型能辨識問題的類型），但**視覺路徑目前沒有貢獻**——4/16-epoch 的訓練不足。這把「此
checkpoint 只具定性地位」從定性判斷變成定量確認。

**為什麼幻覺／誘餌指標在此不可判讀。** E7 的核心指標是誘餌捕獲率（hallucination_rate）
與 `bait_share_of_errors`（誘餌佔錯誤的比例），用於判斷模型是否穩定選取被描述物體、不改
選誘餌物體。當模型的預測是 chance 水準、近乎常數時，這些指標成為**無意義的 artifact**：
對二元屬性（material、size 只有兩個值），任何錯誤在建構上**必然是誘餌形狀的**——不是
正確值，就只剩誘餌值這一個選項。因此 material 的 bait_share 0.96、size 的 1.00 不代表
任何選取誘餌的實質行為，只是二元屬性加上隨機預測的必然結果。**在這個 regime 下，E7 對
Flamingo 不可判讀。** Flamingo 的 E7 部分要等重訓（規劃為 no-LoRA + 預先計算特徵的方案，
待專案負責人放行）之後才能納入。

---

## (d) 兩處與預先登記的偏離 — 需要專案負責人裁決

撰寫整合報告（`docs/substrate_fixation_report.md`）時，比對預先登記文件
（`docs/paper_v2_outline.md`）後標記出兩處**執行與登記不符**。兩處都不改變任何結論的
方向，但為維持論文的預先登記誠信，需要決定 camera-ready 時修改登記還是修改措辭。

**偏離一：E7 頭條指標的替換。**

- 預先登記（A1.3）指定的頭條指標是 `bait_share_of_errors`（誘餌佔錯誤的比例），理由是它
  能把幻覺與一般分佈位移分開。
- 實際整合報告改用 `hallucination_rate`（誘餌捕獲率）作為主軸讀數，把
  `bait_share_of_errors` 降為輔助指標，只用於確認錯誤確為誘餌形狀。
- **統計理由**：`bait_share_of_errors` 在**錯誤稀有時不穩定**——完整訓練模型的錯誤極少，
  分母極小：size 只有少數幾個錯誤即得 bait_share 1.00，material 為 0 個錯誤而完全無定義。
  此指標在錯誤數為零時無定義、錯誤數極小時變異極大，無法作為捕獲率的量測。相對地，
  hallucination_rate 提供了論證所需的因果對比（+CA 模型 0–7% 對 −CA 模型 24–59%）。
- **待決**：camera-ready 或者修改 outline 的登記（頭條指標改為 hallucination_rate），
  或者修改報告措辭使其與原登記對齊。實質結論不受影響。

**偏離二：T2I sweep 超出登記的單一 timestep 設計。**

- 預先登記只登記**單一 timestep t=100、每類 300 題**。
- 實際執行擴成 **t=100／261／400 的 sweep，且 t=261/400 每類為 n=150 題**（樣本數縮減）。
- 這是對預先登記設計的**擴充**。它使負向結論更穩健（多個雜訊程度下皆無訊號），但嚴格
  而言，t=261/400 屬於預先登記範圍之外的探索，且縮減的 n 使其超標判定的統計檢定力較低。
- **待決**：camera-ready 或者把 sweep 補進登記（載明其為預登記後的穩健性擴充），或者只
  報告登記的 t=100、把 sweep 列為附錄的探索性檢查。負向判讀本身不變。

兩處偏離都已記入 `JOURNAL.md` 的 TODO，也寫進 `substrate_fixation_report.md` 的「限制」節。

---

## 當天的兩項圖表交付（支援性 artifact）

除上述四項結果外，當天另產出兩份圖：

- **A/B/C 的 CA-share 對比圖（E9）**：`abc_localization.py` 新增輸出
  `outputs/analysis/abc_localization/clevr_dinov2_decoder1l_scratch/abc_contrast.png`——
  每個屬性一組 cross-attention 佔比（CA-share）的長條圖，並標示 0.5 的「無偏好」參考線；
  A>B>C 的梯度可直接自圖中讀出（shape 的 C 情形最低，0.199）。底層 JSON 經驗證位元相同，
  僅新增圖檔（RESULTS.md §6 的圖註）。
- **v2 標籤重繪（E10）**：`grounding_manipulation.py` 新增 `--replot-from`，用已存的 JSON
  重繪圖（不需 GPU），輸出到**新目錄**
  `outputs/analysis/grounding_manipulation/clevr_dinov2_decoder1l_scratch_v2labels/`，
  原圖保留未動。（2026-07-07 命名更新：該批圖使用的「Retrieval (object)/(answer)」
  標籤已被使用者裁定廢除——中間鏈層級即 Retrieval 階段，答案層級為原有的
  classification 讀出；標籤修正版重繪於
  `.../clevr_dinov2_decoder1l_scratch_v3labels/`。）

兩者皆為支援性 artifact：不引入新數字，只把既有結果以正確的 v2 命名與對比方式呈現。

---

## 小結

本批結果完成上述三部分論證所需的最後一項量測，並界定了兩項論文邊界。(a) E8 現已涵蓋
全部四個骨幹；MAE 的結論由「基底較弱」改寫為「可供 Binding 使用的結構較弱，屬性資訊並未
缺失」，是對 claim A3 的精確化，不是削弱。(b) T2I sweep 在預先登記的雙條件標準下為負向，
並受預先登記的 domain-mismatch 警語約束為有界結論；其作用是界定論文貢獻——VQA 訓練誘發
此機制，而非機制自發存在。(c) Flamingo 第一次 E7 的全零準確率由 harness 的大小寫 bug
產生；修正後的量測顯示 chance 水準、近乎退化的預測，E7 指標在此 regime 不可判讀；同源
bug 已在訓練腳本的評估函數中修正，避免重訓時回報恆為 0 的驗證準確率。(d) 兩處預先登記
偏離（E7 頭條指標、T2I sweep 範圍）不影響結論方向，但需要專案負責人在 camera-ready 前
決定修改登記或修改措辭。整合敘事見 `docs/substrate_fixation_report.md`。

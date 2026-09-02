# 問題怎麼改變物件的 patch token：語言條件實驗整理報告

整理日期 2026-08-31。本報告描述語言條件實驗套件全部十一個 run（registry 條目 X21 A 到 K，編號只在 `docs/experiment_registry.md` 內部使用）、程式
`scripts/analysis/patch_language_condition.py`、結果目錄
`outputs/analysis/patch_language_condition/`，數據總表見文末（由
`scripts/analysis/language_condition_table.py` 產生，`summary_table.{md,csv}`）。
讀者設定為第一次接觸這個專案的人；本文用到的每個量在第一次出現時定義。

## 1. 研究問題

模型是凍結的 ViT-B backbone，加上六層 gated cross-attention（接在 block
1、3、5、7、9、11 之後）把問題文字寫進影像的 patch token，再由一層
transformer decoder 從全部 patch token 讀出答案。先前已經證明：不給問題時，
一個物件所在的 patch token 等於「該位置的背景 token」加上「一個只和物件有
關、和位置無關的向量」。本報告回答的問題是：給了一句 referring question（例如
What color is the cube?）之後，這個向量怎麼變、變化發生在哪幾個 block、最後
decoder 從哪些 token 讀出答案，以及這些答案是否隨 queried attribute 與
backbone 而不同。

文獻裡有兩種說法需要放在同一組座標上比較。Song、Lepori 與 Pavlick（2025）在
vision-language model 上量到，問題會讓 queried attribute 在物件 token 上的成
分變大；我們先前在自己的模型上量到的則是，問題會讓 non-referent（沒被問到的
那個物件）的向量整體變小。

## 2. 設定

影像為 324 張 CLEVR 合成場景，每張恰有兩個物件：被問的物件稱為 target，另
一個稱為 distractor；兩者顏色不同且都不是灰色，以便用像素分割把每個 patch
標成 target、distractor 或背景三組之一。另有 324 張只含一個物件的場景，用來
估計屬性方向與逐位置背景平均。

每張影像跑四次前向：(1) 不給問題；(2) 問 target 的 queried attribute，
referring expression 取 target 與 distractor 第一個不同的屬性（順序 shape、
size、material、colour，排除 queried attribute 本身）；(3) 問 distractor 的
queried attribute；(4) 不指涉物件的問題 What {attribute} is the object?。

Queried attribute 分別為 colour、shape、material、size（DINOv2），另以
SigLIP 與 MAE 兩個 backbone 重跑 colour 的整套。三個 backbone 的網格分別為
24×24（DINOv2，336 像素）、16×16（SigLIP，256）、14×14（MAE，224）。每個
run 的不給問題條件都與先前的加性結構分析在 30 張影像上逐項重現（例如
DINOv2 在 block 11 同型別跨位置的 cosine 0.912、不同型別 0.624）。

本文用到的三個量：「物件的 patch 平均」是某物件所屬全部 patch token 在某個
block 輸出的平均向量（768 維）。「屬性方向」是從單物件影像估出來的單位向
量，例如 V_red = 正規化（所有紅色物件的 patch 平均 − 所有物件的 patch 平
均），每個屬性值各一個。「投影」是物件的 patch 平均與某個方向的內積，代表
這個物件的 token 沿該方向有多少成分。此外「物件向量」指物件的 patch 平均減
去逐位置背景模板（每個 patch 位置在其他影像中屬於背景時的平均 token）。

## 3. 量測方法

**RSA。** 取 324 張影像的 target 物件向量，算 324×324 的兩兩 cosine 距離矩
陣，與三種模型矩陣做 Spearman 相關：位置矩陣（兩張影像 target 位置的歐氏距
離）、屬性矩陣（兩張影像的 target 在該屬性上不同記 1、相同記 0，每個屬性各
一個）、四屬性全同矩陣。相關高表示物件向量的差異跟著該因素走。

**屬性方向上的投影差。** 對 target 與 distractor 各量三個投影：對自己的
queried attribute 方向、對其他值方向的平均、對另一個屬性（對照屬性）的方
向。比較「問 target − 問 distractor」（只跟哪個物件被問有關，稱為
selection contrast）與「給問題 − 不給問題」（包含問題帶來的所有變化）。

**條件之間的 activation patching。** 對同一張影像跑前向 A（問 target）與前
向 B（問 distractor 或不給問題）。在前向 A 第 ℓ 個 block 的輸出，把某一組
patch token（背景、兩個物件、只 target、只 distractor）換成前向 B 在同一
block 的對應 token，其餘不動，之後照常執行，看 decoder 的答案是 target 的
值、distractor 的值或其他。ℓ 掃 0 到 11。四項檢查：逐詞生成與第一個詞的
argmax 一致；兩種問題的準確率；decoder 注意力每列加總為 1；換入前向 A 自己
的 token 時 12 個 block 都重現 baseline。只計入兩種問題都答對、且兩物件在
queried attribute 上不同的影像。

**加性介入。** 從單物件影像算 queried attribute 的差向量 Δ(A→B) = 值為 B 的
物件 patch 平均 − 值為 A 的，在第 ℓ 個 block 加到 target 的 patch 上，看答
案由 A 變成 B 的比例；對照組為同長度的隨機向量、加在背景子集、加在
distractor 上（問 target 時）、加在 target 上（問 distractor 時）。

**Decoder 注意力。** decoder 只有一層，起始 token 對全部 patch 的注意力權
重依三組加總、除以組內 patch 數，得到每個 patch 平均分到的注意力。

**Head 歸零。** 逐一把一個 attention head 的輸出歸零（self-attention 12 個
block × 12 個 head、GCA 6 層 × 16 個 head），或把一整層的 head 全部歸零，量
selection contrast 在 block 9、10、11 的變化與準確率；另做 GCA 第 7、9 層的
head 組合（只保留單 head 效應最大的四個、只歸零那四個、隨機保留四個、隨機
歸零八個）。

**單一 patch 的 linear probe。** 從單一 patch token 讀出所屬物件的屬性與
「是否為 referent」，切分方式含依影像分組的隨機五折與留出空間位置。

## 4. 結果

### 4.1 位置不變（DINOv2，colour）

用整張影像的背景平均當基準時，位置矩陣的相關在 block 0 到 8 維持 0.6 到
0.8，屬性訊號被蓋住。改用逐位置背景模板後，位置相關降到不給問題時的 0.1
到 0.3，給問題時從 block 5 起接近 0。物件向量和位置無關這件事因此同時有
cosine 與 RSA 兩種證據。

### 4.2 selection 的形式：從 non-referent 移除 queried attribute（DINOv2，四個屬性）

Selection contrast（target 對自己 queried attribute 方向的投影，問 target −
問 distractor）在 block 4 以前為 0，中段起上升，block 11 為 colour +11.4、
shape +14.7、material +10.3、size +4.9；distractor 的數值是鏡像。RSA 的屬性
矩陣在 block 11 的相關：target 為 referent 時 colour 0.59、shape 0.81、
material 0.86、size 0.73；target 為 non-referent 時 0.06、0.05、0.04、
0.09。四個屬性一致：non-referent 的 queried attribute 資訊從它自己的 patch
上消失，referent 的維持。

### 4.3 queried attribute 是否被抬起，取決於 backbone 預設保留多少

「給問題 − 不給問題」的走向依屬性而異。不給問題時，backbone 對各屬性的保
留程度（RSA 相關由 block 0 到 11）為 colour 0.43 → 0.01、shape 0.02 →
0.77、material 0.09 → 0.19、size 0.09 → 0.10。問到該屬性時，referent 的相
關升到 colour 0.59、material 0.86、size 0.73；shape 本來就有 0.77，問了之
後 0.81，幾乎沒有抬升。投影上的抬升則只有 colour 明顯（block 8 為 +12.8，
兩個物件一起），material 與 size 在 +0.7 到 +2.8 之間，shape 為負。因此
「queried attribute 在兩個物件上一起放大」不是通則；抬升的幅度與 backbone
預設保留該屬性的程度相反。這句是推論；抬升出現在 RSA 的組織而不在原始投
影上的差異照實記錄、未解釋。沒被問到的屬性在所有 run 裡都不被維持。

### 4.4 decoder 從哪些 token 讀出答案（DINOv2）

從問 distractor 的前向換入 token 時，換掉兩個物件的 patch 使答案變成
distractor 值的比例在 block 7 為 0.74 到 0.99、block 9 到 10 為 0.92 到
0.99、block 11 掉到 0.04 到 0.25；換掉背景 patch 在 block 10 以前不超過
0.05、block 11 為 0.67 到 0.97（四個屬性）。只換 distractor 的 patch 有
0.3 到 0.5 的效果，只換 target 的幾乎沒有。也就是在 block 7 到 10，「哪個
物件被問到」由物件 token（主要是 non-referent 的）決定；在 block 11 這個訊
息被複製到背景 token，decoder 從那裡讀。Decoder 的注意力仍集中在 referent
（每 patch 10 到 15×10⁻³，背景 1.5，non-referent 2 到 3）。

### 4.5 加性介入（DINOv2）

顏色差向量加在 target 的 patch 上，block 0 到 10 使答案翻轉的比例為 0.80
到 0.99，block 11 只有 0.27；加在全部背景 patch 上時 block 11 反而 0.79，
與 4.4 一致。形狀差向量從 block 5 才開始有效（0.10），block 8 到 10 為
0.92 到 0.96，對應單一 patch 的 shape probe 在 block 5 起才到 1.00。所有對
照組在每個 block 都是 0。

### 4.6 哪些 head 寫入 selection（DINOv2，colour）

240 個單 head 歸零裡，block 11 的 selection contrast 變化中位數 0.0、第 5
百分位 −0.7，沒有一個使準確率低於 0.97；例外是 block 11 的 self-attention
第 7 個 head（11.4 → 5.1，block 9、10 不變）。把 GCA 第 7 層或第 9 層整層歸
零，block 9、10 的 selection 降到約 0、準確率約 0.7；第 1、3、11 層無影響。
組合實驗：只保留第 7 層效應最大的四個 head 保住 +3.6／+4.2（未介入
+5.3／+6.0），隨機四個只剩 +1.3 到 +2.0；只歸零那四個剩約一半。寫入集中在
該層約四分之一的 head 上、其餘分散；沒有任何四個 head 是充分或必要的。逐
head 與 activation patching 的 recovery 相關只有 −0.02 到 +0.28。

### 4.7 換 backbone

SigLIP：機制與 DINOv2 相同，selection contrast 從 block 5 出現、block 11 為
+17.7；non-referent 的 colour RDM 0.09 對 referent 0.60；換掉物件 token 從
block 5 到 11 都改變答案（0.80 到 0.88），背景不超過 0.21——沒有 block 11
的背景複製；差向量在每個 block 都能翻轉（0.83 到 0.95）。

MAE：selection 但不移除。selection contrast 最大只有 +1.6，colour RDM 在每
個 block、每種條件都停在 0.48 到 0.55（non-referent 0.51 對 referent
0.54）。referent 卻被標記：單一 patch 的 referent probe 由 block 7 的 0.79
升到 block 11 的 0.99，decoder 注意力每個 referent patch 128×10⁻³、
non-referent 1.3、背景 2.9。換掉物件 token 在 block 7 到 10 只改變 0.13 到
0.15 的答案、block 11 為 0.90；背景不超過 0.12。差向量每個 block 都能翻轉
（0.54 到 0.95）。

## 5. 判讀

對四個 queried attribute（DINOv2）都成立的：selection 的形式是把
non-referent 身上 queried attribute 的成分從它自己的 token 移除、referent 維
持；這個 selection 由物件 token 決定，寫入位置是 GCA 第 7、9 層，集中在約四
分之一的 head；queried attribute 的成分是可加的；沒被問到的屬性不被維持；
最後一個 block 把區分物件的訊息複製到背景 token。

依屬性而異的：queried attribute 在被問時是否被抬起，取決於 backbone 預設保
留它的程度（colour、material、size 有、shape 沒有）。

依 backbone 而異的：SigLIP 與 DINOv2 用同一種移除式 selection（早兩層、無
背景複製）；MAE 保留全部屬性、標記 referent、由 decoder 注意力選擇。因此
「移除」不是 gated cross-attention 本身的性質，而是它在特定 backbone 上的實
現；block 11 的背景複製只有 DINOv2 有。MAE 的標記式 selection 是否對應它保
留全部屬性的預設表徵，是推論。

未解：material 與 size 的抬升只在 RSA 組織上可見、投影上很小，兩種量測的
差異沒有解釋；Sup-ViT 尚未重跑；SigLIP 與 MAE 的 head 分佈未量；物件數超過
兩個的情形未測。

## 6. 數據總表

見 `outputs/analysis/patch_language_condition/summary_table.md`（同一份數字
也併入 `docs/results_tables.md`）。欄位：sel9／sel11 為 selection contrast
在 block 9／11；rise8 為 referent 的 queried attribute 投影「給問題 − 不給問
題」在 block 8；rsa_none11／rsa_ref11／rsa_nonref11 為 queried attribute 矩
陣的 RSA 相關在 block 11（不給問題／target 為 referent／target 為
non-referent）；swap_obj9／swap_obj11／swap_bg11 為換掉物件 token（block 9、
11）或背景 token（block 11）後答案變成 distractor 值的比例；attn_ref／
attn_nonref 為 decoder 對每個 referent／non-referent patch 的平均注意力
（×10⁻³）；flip5／flip11 為差向量介入在 block 5／11 的翻轉比例；
probe_ref7 為 block 7 的單一 patch referent probe；acc 為問 target 時的
baseline 準確率。

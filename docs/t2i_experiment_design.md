# 實驗設計：T2I 模型上的 zero-shot grounding 分析

> 論文中文草稿用節。撰寫：Opus agent，2026-07-06（user 指示：實驗報告由 Opus 撰寫）。
> 事實來源：`scripts/analysis/t2i_pixart_probe.py`、`docs/paper_v2_outline.md` §T2I。

## 研究問題

本文主實驗主張：語言 conditioning（cross-attention）在一組凍結的預訓練視覺表徵之上，
實作出一套 Binding → Retrieval 的 grounding 機制。一個自然的質疑是，此機制或許只是
我們在 CLEVR 上進行 VQA 訓練所誘發的 artifact，而非語言 conditioning 本身的通用性質。
本節設計一項 zero-shot 檢驗來回應此質疑：我們取一個從未接受任何 VQA 訓練、僅為文字
生圖（text-to-image）目標訓練的模型，檢驗其 cross-attention 是否在不經任何微調的情況
下，就已表現出 grounding 機制的兩項可觀測特徵——(i) referent 的空間定位（對應 Binding
階段），以及 (ii) 答案屬性在視覺表徵中的可解碼性（對應 Retrieval 階段）。若這兩項特徵
在一個與 CLEVR 訓練完全無關的模型上 zero-shot 地出現，則機制屬於 cross-attention 預訓
練通用副產品的解釋將獲得支持。

## 模型與資料

受測模型為 PixArt-Σ。其骨幹為 DiT，含 28 個 transformer blocks，每個 block 內同時具備
self-attention 與 cross-attention；文字端為凍結的 T5-XXL 編碼器。此設定與主實驗共享
「凍結視覺骨幹 + 語言 conditioning 經由 cross-attention 注入」的結構，但其
cross-attention 從未見過問答監督訊號，故構成一個乾淨的對照。

資料取自 CLEVR val set，選用 RETRIEVAL_CATEGORIES 中的三個 attr_query 類別（direct、
same、spatial），每類 300 題。對每一題，我們以其 program 的執行結果定位 referent 物體，
取最後一個 unique step 的輸出作為 referent；再由該物體的 pixel_coords 映射到 32×32
latent-token 網格上，取以其中心為核的 3×3 patch 窗作為 referent-local 區域。

## 方法

特徵抽取採 DIFT-style 的單步流程，固定 seed=42：

- 將 CLEVR 影像 resize 至 512×512，經 VAE 編碼為 latent；
- 於 timestep t=100/1000 對 latent 加噪；
- 以「該題的問題文字」為 prompt，執行單步去噪 forward；
- 透過 hook 擷取全部 28 個 block 的 hidden states 與 cross-attention 機率圖。

在此之上進行三項分析。**分析一（per-block linear probing）**：對每個 block 取兩種特徵
——全圖 mean-pool 與 referent 3×3 窗 mean-pool——分別以 5-fold logistic regression 預測
答案屬性。其中 referent-local 特徵的抽取協議與主實驗 E8 的 raw-backbone probe 一致，故
其結果可與主實驗直接比較。**分析二（cross-attention localization）**：量測各 block 的
cross-attention 在 referent 窗上的質量。**分析三（frozen 1-layer decoder readout）**：
此為獨立的後續工作，協議對齊論文 Table 1，本節僅作預告，不在此報告結果。

## 量測指標

分析一報告每個 block 的 probe 準確率，並以 majority-class baseline 作為對照下限。
分析二須留意 DiT 的 cross-attention 以 image token 為 query、text token 為 key，每一列
（每個 image token 對所有 text token）的機率和為 1；因此定位訊號不可直接沿列讀取，而須
以 column 正規化：先將每個 text token 的注意力在所有 image patches 上正規化，再對 text
tokens 取平均，得到「各 image patch 相對於其他 patch 受到多少 text 注意力」的分佈，最後
量測其落在 referent 3×3 窗上的質量。此指標的 chance 水準為 9/1024 ≈ 0.0088。

## 預期結果與判讀

判讀採預先登記的雙條件標準。若同時滿足 (a) referent-local probe 在中晚層顯著高於
majority baseline，且 (b) cross-attention 質量在 referent 窗上顯著高於 chance，則結果
支持「binding 式的語言—視覺對接是 cross-attention 預訓練的通用副產品」，從而反駁「該
機制僅為 CLEVR 訓練 artifact」的質疑。反之，若兩項訊號皆缺，則結論須限縮為「此機制需要
task 訓練方能浮現」，主實驗的貢獻據此定位為揭示訓練如何誘發並強化該機制，而非其自發
存在。

## 限制（預先登記）

本設計有兩項須事先聲明的限制。其一，CLEVR 的問句並非 caption，對 T2I 的文字編碼器構成
domain mismatch；因此負向結果無法區分「機制不存在」與「prompt 分佈不匹配」兩種可能，
而正向結果則因跨越此 mismatch 仍成立，反而更為保守可信。其二，512×512 的 resize 相對於
CLEVR 原生渲染尺寸造成非等比變形，使 referent 座標映射至 patch 網格時帶有量化誤差；
我們以 3×3 窗而非單一 patch 作為 referent 區域來緩解此誤差。

（全節術語遵循命名規範：Grounding 指整個語言 conditioning 機制，其階段依序為 Binding
與 Retrieval，不使用「object grounding」或「Object match」作為階段名稱。）

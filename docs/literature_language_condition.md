# 文獻對照：語言條件下的 selection 機制與 backbone 的屬性保留

整理日期 2026-08-31。兩次網路搜尋（各一個 agent，2026-08-31），只收錄有
arXiv 編號或 DOI 可核對的論文。每一條先寫文獻量到什麼，再寫與本專案哪一項結
果對應（結果編號依 `docs/language_condition_report.md` 與 registry X21）。

## 一、selection 是「移除」還是「標記」

**與 MAE 的標記式 selection 對應（X21 K）**

- Song, Lepori & Pavlick (2026). *Linguistic Context Recodes Visual Representations in Vision-Language Models.* arXiv:2608.00035。在 Qwen2.5-VL-7B 與 InternVL3-8B 上量到影像 token 上有可線性讀出的「reference」標記（中後層達峰，steering 可翻轉答案），以及 referent 的 shape／colour 投影在後層上升。對應：MAE 的 referent probe 由 block 7 的 0.79 升到 0.99，是同一種標記；他們的「屬性放大」與我們 DINOv2／SigLIP 的「從 non-referent 移除」方向相反，但他們沒有量 non-referent 的投影是否下降，所以兩者並不矛盾；本專案在四個屬性上量到的「抬升幅度隨 backbone 預設保留程度而變」，文獻中沒有找到先例。
- Assouel, Campbell, Bengio & Webb (2025). *Visual symbolic mechanisms.* arXiv:2506.15871。七個 VLM 中找到與內容無關的位置 ID：ID-retrieval head（第 12–16 層）、ID-selection head（18–19）、feature-retrieval head（20–27）以 ID 為指標取屬性。對應：指標式 selection 加讀出端取值、不改動 non-target 的內容，與 MAE 的路徑同一類；三階段順序與「中段標記、最後由 decoder 注意力取值」平行。
- Hasani et al. (2025). *Uncovering Grounding IDs.* arXiv:2509.24072。外部分割線索在 object patch 上誘發潛在識別碼（第 20–27 層），交換 patch 活化後預測跟著識別碼走（0.98）。對應：同樣是把 selection 寫成 token 上的識別碼而非屬性內容。
- Haputhanthri, Campbell, Assouel, Cohen & Webb (2026). *Binding Visual Features Point by Point.* arXiv:2605.25427。訓練 VLM 輸出座標會誘發序列式的視覺搜尋。對應：把 selection 框成一次注意一個物件，與 decoder 注意力集中在 referent 的讀出一致。

**與 DINOv2／SigLIP 的移除式 selection 對應（X21 E、G、J）**

- Golovanevsky et al. (2024/2025). *What Do VLMs NOTICE?* arXiv:2406.16320。在 BLIP 類模型上以 activation patching 找到中層 cross-attention head 做「object inhibition」「outlier inhibition」。對應：唯一明確以 cross-attention head 的「抑制」描述物件層級 selection 的機制文獻，但沒有量屬性內容是否從 patch token 移除；本專案補上了這一步。
- Cui et al. (2026). *The Dual Mechanisms of Spatial Variable Binding in VLMs.* arXiv:2603.22278。binding 用的空間資訊分佈在包含背景在內的全部視覺 token。對應：DINOv2 在 block 11 把 selection 資訊複製到背景 token。
- Campbell et al. (2024). *Understanding the Limits of VLMs Through the Lens of the Binding Problem.* arXiv:2411.00238（NeurIPS 2024）。從認知科學論證多物件失效來自共享表徵需要序列注意力。對應：本專案「物件數超過兩個」的規劃依據；biased-competition 傳統（Desimone & Duncan 1995）在 2024–2026 的 ML 文獻中只經由此篇連結。
- *Latent Noise Mask for Reducing Visual Redundancy in MLLMs* (2026). arXiv:2606.30168。與問題無關的視覺 token 會與證據 token 競爭，需另外學一個依問題的相關性遮罩。對應：工程面把「壓掉無關 token」當成模型預設做不好的事；本專案顯示 gated cross-attention 自己學會了這件事（DINOv2／SigLIP），而 MAE 上則改為標記。

**讀出端的 selection（X21 D、K）**

- Neo et al. (2025). ICLR 2025, arXiv:2410.07149：LLaVA 中物件資訊留在物件自己的視覺 token，後層由最後一個位置取出。對應：答案由物件 token 而非全域摘要決定。
- Zhang, Yadav, Han & Shutova (2025). CVPR 2025, arXiv:2411.18620：中層只搬運與問題相關那個物件的資訊。對應：中段的 selection。
- Kaduri, Bagon & Dekel (2025). CVPR 2025, arXiv:2411.17491：跨模態流集中在中間約四分之一的層。對應：移除的起點在中段。
- Salazar et al. (2026). arXiv:2607.03358：直接路徑與經文字中介的路徑並存、依任務切換。對應：DINOv2／SigLIP 的原地移除與 MAE 的 decoder 端集中，是兩條並存的路徑。
- Kim et al. (2025). arXiv:2509.17588：少數 head 承擔影像到文字的流動，背景 token 也帶影像資訊。對應：head 組合實驗與背景複製。
- Kang et al. (2025). ICLR 2025, arXiv:2503.03321；Luo et al. (2025). arXiv:2510.08510；Jiang, Dravid, Efros & Gandelsman (2025). NeurIPS 2025, arXiv:2506.08010：高 norm 的 sink／register token 不論問題都吸引注意力。對應：**需要補一項對照**——decoder 對 referent 的注意力（每 patch 約 100 倍於背景）要對 token norm 檢查，確認集中來自物件而非高 norm token；DINOv2 無 register，是已知易出現 sink 的 backbone。
- Liu et al. (2025). arXiv:2510.04819：LM 內部的影像 value token 可零樣本做 referring-expression 偵測。對應：referent selection 可由語言條件下的影像 token 讀出。

## 二、預訓練目標與 patch token 保留哪些屬性（X19、X21 G、J、K）

- Park et al. (2023). ICLR 2023, arXiv:2305.00729：contrastive 模型後層偏向低頻、形狀導向的全域樣式，masked-image-modelling 偏向高頻、紋理導向且主要在淺層。對應：DINOv2／SigLIP「shape 隨深度增加、colour 消失」對 MAE「全部保留」的分野最接近的先例。
- Dorszewski et al. (2025). arXiv:2503.24071：逐層神經元標記；所有模型第一層以顏色概念為主、到最後一層幾乎消失；MAE 中層近半神經元對應材質與紋理。對應：與本專案的 colour RSA 曲線（0.43 → 0.01）一致；**但他們的顏色消失包含 MAE，與本專案 MAE 到 block 11 仍保留顏色（RSA 0.52）相反**，寫報告時要明說這個分歧（他們量的是神經元概念標記，我們量的是 patch 向量的 RSA，量法不同）。
- Wagner & Harmeling (2025). arXiv:2503.09867：CLIP／DINOv2／DINOv3 擅長邊緣導出的幾何（shape、size），保留不了表面線索（colour、material、texture）。對應：DINOv2／SigLIP 在 block 11 的屬性保留輪廓（shape 高、colour 與 material 低）。
- Walmer et al. (2023). CVPR 2023, arXiv:2212.03862：局部與全域資訊的處理順序取決於訓練目標。對應：把預訓練目標當成決定 patch 保留什麼的變因。
- Vanyan et al. (2024). arXiv:2401.00463：MAE 的 patch token 含高變異、非語意的維度。對應：MAE token 保留原始外觀而非抽象類別碼。
- Zhou et al. (2026). ECCV 2026, arXiv:2607.01987：DINOv2 把幾何訊號壓在緊湊的線性子空間，MAE 散在較多維度。對應：逐屬性 RSA 的方法學近親。
- Li, Salehi, Ungar & Kording (2025). NeurIPS 2025, arXiv:2510.24709：IsSameObject 在 DINO／CLIP／supervised 超過 90%，在 MAE 明顯較弱。對應：MAE 的兩物件可分是逐 patch 的屬性保留，不是學到的同物件碼。
- Lepori et al. (2024). NeurIPS 2024, arXiv:2406.15955；Vilas et al. (2023). NeurIPS 2023, arXiv:2310.18969；Gandelsman, Efros & Steinhardt (2024). ICLR 2024, arXiv:2310.05916；Darcet et al. (2024). ICLR 2024, arXiv:2309.16588；Oquab et al. (2023). arXiv:2304.07193：屬性子空間、patch token 逐層累積類別資訊、影像表徵的加性分解、register token、DINOv2 的 PCA 前景分離。對應：加性物件向量與背景基準的依據。（找不到「Dai et al. 2024」的 patch PCA 論文；PCA 前景分離出自 DINOv2 論文本身。）
- Tong et al. (2024). CVPR 2024, arXiv:2401.06209；Kar et al. (2024). ECCV 2024, arXiv:2404.07204；Tong et al. (2024). NeurIPS 2024, arXiv:2406.16860；Jiang et al. (2024). arXiv:2310.08825；Yang et al. (2025). COLM 2025, arXiv:2408.16357；Shi et al. (2024). arXiv:2408.15998；Arias, Baldrich & Vanrell (2025). arXiv:2502.04470；Gurung, Hoffmann & Brox (2025). arXiv:2507.07985：frozen encoder 丟掉的屬性（尤其 colour、counting）直接變成下游 VLM 的失效，換或混合 encoder 可改善。對應：「queried attribute 只在 backbone 會丟掉時才被抬起」的下游意義。
- Huang & Chang (2025). arXiv:2510.09794：可解碼與因果使用的資訊在各層分歧。對應：本專案同時做 RSA 與介入的理由；單看 RSA 不足以下因果結論。
- Zhang et al. (2022). arXiv:2210.11470（i-MAE）：MAE 潛在表徵線性可分性較低但資訊完整。對應：MAE 保留全部屬性的解讀。

## 三、由文獻直接導出的補充實驗

1. Decoder 注意力對 token norm 的對照（Kang 2025；Jiang 2025）：把 decoder 的每 patch 注意力與該 patch 的 norm 做相關，並在排除高 norm token 後重算 referent 對背景的比值。用現有 cache（token_norm_stats）與 readout 結果，CPU。
2. 對 Song 等人的直接比較：在同一批影像上量 non-referent 的屬性投影是否下降，他們的資料沒有這一項；本專案的結果可以直接寫成對他們的補充。
3. Dorszewski 等人與本專案在 MAE 顏色保留上的分歧：用他們的神經元標記法在我們的 MAE 上重跑，或用我們的 RSA 在 ImageNet 影像上重跑，判斷是量法差異還是資料差異。

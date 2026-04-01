# 影像模型實驗成果與比較分析報告

本報告針對目前在影像分支 (Image Branch) 所進行的三種主要電腦視覺模型（ResNet50, Swin-Transformer, Vision Transformer (ViT)）之訓練與微調結果進行統整，包含資料準備、模型設定以及最終效能結果比較。

## 1. 資料來源與分割設定

本次實驗所使用的主要資料集以及其取樣設定如下：

*   **資料來源：** [ChestMNIST](https://medmnist.com/) (隸屬於 MedMNIST 資料集)，任務為預測肺部 X 光影像中是否患有**肺浸潤 (Infiltration)** 症狀的二元分類問題。
*   **影像大小：** 224 x 224。
*   **總資料量與分割 (Subset 10k Binary)**：為了加速實驗並確保正負樣本比例，採用了採樣機制 (Subsampling)，抽取出總共 10,000 筆資料。
    *   **訓練集 (Training Set)：** 8,000 筆
    *   **驗證集 (Validation Set)：** 1,000 筆
    *   **測試集 (Test Set)：** 1,000 筆
*   **類別比例：** 此子集被設定為固定含有 40% 的正樣本 (`SUBSAMPLE_POS_RATIO = 0.4`)，以解決原資料集潛在的不平衡問題。

## 2. 訓練之模型與策略

本次實驗使用了標準卷積神經網路 (CNN) 與近年主流的視覺 Transformer 進行比較，並導入了多種預訓練及 Parameter-Efficient Fine-Tuning (PEFT) 策略。

### 訓練模型架構
1.  **ResNet50:** 作為 CNN 的強勢基準模型 (Baseline)。
2.  **Swin-Transformer (Tiny / Swin-T):** 具備階層式結構的區域性 Transformer。
3.  **Vision Transformer (ViT):** 標準基於全域自注意力機制的視覺 Transformer。

### 預訓練策略 (Pre-training)
所有模型皆比較了兩種初始權重：
*   **ImageNet-1K:** 標準的大規模影像資料集預訓練權重。
*   **DAPT (Domain-Adaptive Pre-Training):** 針對醫療/專屬領域資料進行的無監督或自監督適應性預訓練。 ResNet 採用的 Spark，Swin-T 採用的 SimMIM，以及 ViT 採用的 MAE (Masked Autoencoders) 等產出的局部最佳權重。預訓練權重所使用的資料集為 ChestMNIST。

### 微調機制 (Fine-Tuning, FT)
對於 Transformer 架構 (ViT & Swin-T) 的 DAPT 權重，進一步嘗試了以下幾種進階 FT 機制：
1.  **標準全參數微調 (Standard FT):** 預設策略。
2.  **LLRD (Layer-Wise Learning Rate Decay):**
    *   **參數：** 衰減率 `LLRD_DECAY = 0.85`。
    *   **方法：** 針對深層網路，學習率從頂層到底層逐漸遞減，有助於保留預訓練底層特徵，同時讓頂層更好地適應新任務。
3.  **LoRA (Low-Rank Adaptation):**
    *   **參數：** `R (Rank) = 16`, `Alpha = 32`, `Dropout = 0.1`, `Learning Rate = 5e-4`。
    *   **方法：** 凍結原始預訓練權重，僅在注意力矩陣旁插入可訓練的低秩分解矩陣，大幅減少訓練參數，同時避免 Catastrophic Forgetting，節省 GPU 資源。

---

## 3. 結果分析與比較

評估結果與圖表請參考 [`image_comparison_results`](./image_comparison_results/)資料夾中的圖表與 csv 檔案。

### 模型曲線比較 (ROC & PR Curves)

<div align="center">
  <img src="./image_comparison_results/Integrated_ROC_Curve.png" width="48%" alt="Integrated ROC Curve">
  <img src="./image_comparison_results/Integrated_PR_Curve.png" width="48%" alt="Integrated PR Curve">
</div>

### 評估指標 (Metrics) 

參考目前的評估指標 (`Model_Comparison_Metrics.csv`) 進行分析：

| Model | Accuracy | Precision | Recall | AUC | AP (Avg Precision) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet50_ImageNet** | 0.691 | 0.6086 | 0.6375 | **0.7486** | 0.6402 |
| **Swin-T_ImageNet** | **0.715** | 0.6696 | 0.5675 | 0.7378 | 0.6273 |
| Swin-T_DAPT | 0.645 | 0.5843 | 0.3900 | 0.6783 | 0.5504 |
| Swin-T_DAPT_LLRD | 0.655 | 0.6022 | 0.4050 | 0.6821 | 0.5598 |
| Swin-T_DAPT_LoRA | 0.589 | 0.4225 | 0.0750 | 0.5533 | 0.4336 |
| ViT_ImageNet | 0.698 | 0.6512 | 0.5275 | 0.7252 | 0.6179 |
| ViT_DAPT | 0.703 | 0.6604 | 0.5300 | 0.7291 | **0.6434** |
| ViT_DAPT_LLRD | 0.707 | 0.6499 | **0.5800** | 0.7404 | 0.6346 |
| **ViT_DAPT_LoRA** | 0.705 | 0.6606 | 0.5400 | **0.7422** | 0.6390 |

### 分析結論：

1.  **曲線離散度反映模型特性 (ROC 與 PR 圖表解釋)：**
    從 ROC 與 PR 曲線圖中可以非常顯著地觀察到各個模型的落差形態：
    *   **領先群：** **ResNet50** 與經過完整微調的 **ViT (DAPT+LoRA)** 曲線弧度最為飽滿，包覆在最外側，且在 PR 曲線中能長效維持較高的 Precision 緩步下降。
    *   **失常陷阱：** 反觀 **Swin-T (DAPT+LoRA)** 的 ROC 曲線幾乎貼近 45 度角的對角線（如同盲猜），PR 曲線更是呈現斷崖式墜落，完全喪失鑑別力。這些視覺化曲線十分直觀地反映了底層不同網路結構處理醫學影像特徵時出現的適應性懸殊差異。

2.  **基準模型依然強大：**
    **ResNet50 (ImageNet)** 在未經任何複雜微調機制的狀況下，達到了最高的 AUC (0.7486) 以及良好的召回率 (0.6375)。這顯示傳統 CNN 對於此類醫學影像（尤其是資料量受限於 10k 時）具有極強的穩定性與歸納偏置 (Inductive Bias)。
    
3.  **ViT 與微調策略的有效結合：**
    *   ViT 在套用 DAPT 後，效能穩定超越單純的 ImageNet 權重。
    *   在 DAPT 基礎上加入 **LLRD** 或 **LoRA** 後，不但維持了優異的準確率 (~0.705)，其 AUC 更是提升至 ~0.74 (逼近 ResNet50 baseline)。
    *   這證明了對於 ViT，使用 DAPT 搭配 Parameter-Efficient 的微調方法 (尤其是 LLRD 與 LoRA) 是一個相對成功且穩健的策略。

4.  **Swin-Transformer 的領域適應困境：**
    *   **Swin-T (ImageNet)** 的原始表現相當出眾 (Accuracy高達 0.715)，但一旦換用 **Swin-T (DAPT)**，所有指標皆出現顯著下滑 (AUC 從 0.7378 跌至 0.6783)。
    *   將 **LoRA** 套用於 Swin-T_DAPT 時幾乎崩潰，Recall 降至 0.075 (模型幾乎無法預測出正樣本)，AUC 掉至 0.5533 (近乎隨機猜測)。
    *   **可能原因：** Swin-Transformer 複雜的 Window Attention 階層式結構，在進行 DAPT 預訓練時可能並未良好收斂，或是產生的特徵空間與 LoRA 預設配置的適應性極差（例如 LoRA rank 或插入的位置不適合 Window Attention），導致微調失敗。


### 後續建議：
*   **針對 Swin-T:** 如果後續要繼續使用 Swin-Transformer，建議放棄目前的 DAPT 權重，退回 ImageNet 進行實驗，或是重新檢視並調整用於 Swin 的領域預訓練 (Domain Pre-training) 方法及 LoRA 的設定 (例如將 LoRA 套用到其他投影矩陣)。
*   **針對整體模型選擇:** 目前 **ViT + DAPT + LoRA** 是在使用進階微調技術下表現最好且最均衡的配置，可以做為後續開發基於 Transformer 模型的主力方向。同時 **ResNet50** 由於其實現簡單與高 AUC，仍不失為一個堅實的備案或對照組。

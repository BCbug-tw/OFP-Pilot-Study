# Osteoporosis Fracture Prediction - Pilot Study

## 專案背景 (Project Background)
本專案為「骨質疏鬆性骨折預測 (Osteoporosis Fracture Prediction)」的多模態模型前導實驗 (Pilot Study)。

在實際的臨床診療中，影響骨質疏鬆相關骨折的因素複雜且多元，病患的資料常常同時存在多種不同的模態（例如：包含年齡、病史、抽血數值的「結構化電子病歷資料」，以及 X 光的「醫學影像資料」）。本研究的目標是希望整合這些不同來源的特徵，打造一個多模態融合 (Multimodal Fusion) 的精準預測模型。

在進入複雜的多模態架構之前，為釐清並最大化「單一資料模態」的預測潛力，本前導實驗劃分為兩個主要的目錄分支，各自獨立進行不同領域的技術堆疊驗證與基準線 (Baseline) 建立。

## 目錄結構與分支說明

### 1. 結構化資料分支 ([Tabular Branch](./Tabular%20Branch/))
*   **目的**：專注於處理一維結構化數據（Tabular Data），探討如何最佳地學習到病患的生理指數與類別特徵。
*   **測試模型**：包含產業界主流的梯度提升樹模型 (GBDT： XGBoost, LightGBM, CatBoost)，以及基於深度學習的架構 (FT-Transformer)。
*   **詳細成果**：完整的超參數設定、訓練集分割策略以及各種模型彼此間的效能長短處比較（如 AUC、Accuracy 等），請參閱 [`Tabular_Branch_Report`](./Tabular%20Branch/tabular_branch_report.md)。

### 2. 影像分支 ([Image Branch](./Image%20Branch/))
*   **目的**：專注於處理胸腔X光醫學影像資料，探討不同神經網絡架構在擷取影像關聯特徵上的表現，並特別研究領域自適應預訓練 (DAPT, Domain-Adaptive Pretraining) 與參數高效微調 (如 LoRA) 等技術如何影響成效。
*   **測試模型**：包含卷積神經網絡 (ResNet50) 以及多種視覺 Transformer (Vision Transformer, Swin-Transformer)。
*   **詳細成果**：實驗中所使用的影像處理手法、各階段微調機制的差異、以及最終影像分類指標的比較，請參閱 [`Image_Branch_Report`](./Image%20Branch/image_branch_report.md)。



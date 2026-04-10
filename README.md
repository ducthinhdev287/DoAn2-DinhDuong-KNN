# 🥗 Đồ Án 2 — Hệ Thống Gợi Ý Chế Độ Dinh Dưỡng bằng Machine Learning

> Xây dựng mô hình dự đoán lượng Calo và gợi ý chế độ dinh dưỡng cá nhân hóa  
> dựa trên thành phần dinh dưỡng thực phẩm, sử dụng KNN và các thuật toán ML hiện đại.

---

## 📋 Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Kiến trúc Pipeline](#kiến-trúc-pipeline)
3. [Thành phần dự án](#thành-phần-dự-án)
4. [Kết quả đạt được](#kết-quả-đạt-được)
5. [Hướng dẫn cài đặt & Chạy](#hướng-dẫn-cài-đặt--chạy)
6. [Công nghệ sử dụng](#công-nghệ-sử-dụng)
7. [Tác giả](#tác-giả)

---

## Giới thiệu

Dự án tập trung xây dựng hệ thống **gợi ý dinh dưỡng cá nhân hóa** dựa trên dữ liệu thực phẩm thực tế.  
Mô hình học máy được huấn luyện để:

- **Dự đoán lượng Calo** (`Caloric Value`) từ các thành phần dinh dưỡng
- **Phân nhóm thực phẩm** theo mức năng lượng (Thấp / Trung bình / Cao)
- **Gợi ý món ăn** phù hợp với nhu cầu dinh dưỡng từng người

**Dataset:** `FOOD-DATA-GROUP1.csv` — 551 thực phẩm × 36 chỉ số dinh dưỡng  
(Protein, Fat, Carbohydrates, Vitamins A/B/C/D/E/K, Minerals, ...)

---

## Kiến trúc Pipeline

```
📂 Raw Data (CSV)
      │
      ▼
① Setup môi trường  →  Cài thư viện, Mount Drive, CONFIG
      │
      ▼
② Load & Kiểm tra   →  551 dòng × 37 cột, không có NaN, không duplicate
      │
      ▼
③ EDA               →  Phân phối, Correlation Heatmap, Outlier, SHAP
      │
      ▼
④ Làm sạch          →  Winsorize, Log1p transform, Encode categorical
      │
      ▼
⑤ Feature Engineering → Tỷ lệ dinh dưỡng, Binning, sklearn Pipeline
      │
      ▼
⑥ Train & Evaluate  →  6 models, RandomizedSearchCV, SHAP explainability
      │
      ▼
📦 Export (model.pkl + preprocessor.pkl + predictions.csv)
```

---

## Thành phần dự án

```
DoAn2-DinhDuong-KNN/
│
├── 📓 Do_An_2.ipynb          # Notebook chính — toàn bộ pipeline 6 bước
│
├── 📁 exports/
│   ├── best_model_*.pkl      # Model tốt nhất đã huấn luyện
│   ├── preprocessor.pkl      # Pipeline tiền xử lý (StandardScaler + Imputer)
│   ├── feature_meta.json     # Metadata: tên cột, target, task
│   ├── predictions_*.csv     # Kết quả dự đoán trên Test set
│   ├── df_cleaned.csv        # Dữ liệu sau làm sạch
│   └── 03_eda_todo.csv       # TODO list từ EDA
│
└── 📄 README.md
```

---

## Kết quả đạt được

### So sánh các mô hình (Cross-Validation 5-Fold)

| Model | R² Score (CV) | Ghi chú |
|-------|:---:|---------|
| Linear Regression | 0.9286 | Baseline đơn giản |
| Ridge Regression | 0.9290 | Regularization L2 |
| Lasso Regression | 0.9291 | Regularization L1 |
| Random Forest | 0.9767 | Ensemble trees |
| XGBoost | 0.9866 | Gradient boosting |
| **LightGBM** ⭐ | **0.9897** | **Best model** |

### Kết quả mô hình tốt nhất (LightGBM)

| Chỉ số | Validation Set |
|--------|:--------------:|
| **R² Score** | **0.9897** |
| **MAE** | **12.35 kcal** |

> **Nhận xét:** R² = 0.9897 có nghĩa mô hình giải thích được **98.97%** sự biến động  
> của lượng Calo — kết quả rất tốt, phù hợp vì Calo được tính từ công thức Atwater  
> (Fat × 9 + Protein × 4 + Carbs × 4).

### Top features quan trọng nhất (theo SHAP)

1. 🥩 `fat` — Chất béo
2. 🍞 `carbohydrates` — Tinh bột
3. 💪 `protein` — Đạm
4. 💧 `water` — Hàm lượng nước
5. 🌿 `dietary_fiber` — Chất xơ

---

## Hướng dẫn cài đặt & Chạy

### Bước 1: Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/DoAn2-DinhDuong-KNN.git
cd DoAn2-DinhDuong-KNN
```

### Bước 2: Mở notebook trên Google Colab

1. Vào [colab.research.google.com](https://colab.research.google.com)
2. Chọn **File → Open notebook → GitHub**
3. Dán link repo vào và chọn file `Do_An_2.ipynb`

Hoặc nhấn thẳng vào badge:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/DoAn2-DinhDuong-KNN/blob/main/Do_An_2.ipynb)

### Bước 3: Cài đặt thư viện

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap joblib missingno openpyxl
```

Hoặc chạy cell đầu tiên trong notebook — đã tích hợp sẵn `!pip install`.

### Bước 4: Chạy toàn bộ pipeline

Trong Colab: **Runtime → Run all** (hoặc `Ctrl + F9`)

> ⚠️ **Lưu ý:** Chạy lần lượt từ cell đầu — một số cell phụ thuộc vào biến của cell trước.

---

## Công nghệ sử dụng

| Nhóm | Thư viện |
|------|----------|
| Xử lý dữ liệu | `pandas`, `numpy` |
| Trực quan hóa | `matplotlib`, `seaborn`, `missingno` |
| Machine Learning | `scikit-learn`, `xgboost`, `lightgbm` |
| Giải thích mô hình | `shap` |
| Lưu trữ | `joblib` |
| Môi trường | Google Colab, Google Drive |

---

## Tác giả

| Thông tin | Chi tiết |
|-----------|----------|
| **Họ tên** | *Đỗ Đức Thịnh* |
| **MSSV** | *10123307* |
| **Lớp** | *124231* |
| **Giảng viên** | *ThS Đào Minh Tuấn* |
| **Trường** | *Sư Phạm Kỹ Thuật Hưng Yên* |

---

*Dự án thực hiện trong khuôn khổ môn học — Đợt kiểm tra 2*

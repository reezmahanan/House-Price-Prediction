# 🏡 House Price Prediction Project (Google Colab)

A machine learning project built in **Google Colab** to predict residential property prices in Indian Rupees (₹). This project features a full machine learning lifecycle including Exploratory Data Analysis, Baseline Modelling, Log Transformations, Gradient Boosting (XGBoost), Custom Feature Engineering, and Model Benchmarking.

---

## 📊 Project Overview & Highlights

- **Dataset**: 545 residential properties with 13 original attributes (area, bedrooms, bathrooms, stories, parking, amenities, and furnishing status).
- **Target Variable**: Property price (`price`) in Indian Rupees (₹), ranging from ₹1,750,000 to ₹13,300,000.
- **Top Performing Model**: **Linear Regression with Feature Engineering** achieved the highest accuracy with **$R^2 = 0.6611$**, lowest **MAE of ₹968,819**, and lowest **RMSE of ₹1,308,818**.
- **Key Insight**: Linear regression outperformed complex ensemble methods (Random Forest & XGBoost), indicating a strong linear relationship between square footage, room proportions, and property value in this dataset.

---

## 📁 Repository Structure

```
House Price Prediction/
├── Housing.csv                    # Dataset (CSV format)
├── Housing.xlsx                   # Dataset (Excel format, if uploaded to Colab)
├── House_Price_Prediction.ipynb   # Executed Google Colab Notebook with all outputs & plots
└── README.md                      # Comprehensive project documentation & benchmark results
```

---

## 📈 Benchmark & Experimental Results

All models were evaluated on an identical **80/20 train-test split** (`random_state=42`, 436 training samples, 109 test samples).

### 1. Model Progression Summary

| Stage | Model | MAE (₹) | RMSE (₹) | $R^2$ Score | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline** | **Linear Regression** | **₹970,043** | **₹1,324,507** | **0.6529** | Strong linear baseline |
| **Baseline** | Random Forest (100 trees) | ₹1,021,546 | ₹1,400,566 | 0.6119 | Tends to overfit on smaller tabular sample |
| **Experiment 1** | Random Forest (Log Target) | ₹1,021,111 | ₹1,434,740 | 0.5927 | `y_log = np.log1p(y)` reduced variance but slightly lower test $R^2$ |
| **Experiment 2** | XGBoost (300 trees, lr=0.05) | ₹979,223 | ₹1,353,929 | 0.6373 | Outperformed Random Forest baseline |
| **Feature Eng.** | **Linear Regression + FE** | **₹968,819** | **₹1,308,818** | **0.6611** | 🏆 **Best Overall Performance** |
| **Feature Eng.** | Random Forest + FE (200 trees) | ₹989,526 | ₹1,376,546 | 0.6251 | +1.32% improvement from new features |
| **Feature Eng.** | XGBoost + FE (300 trees) | ₹1,021,711 | ₹1,377,814 | 0.6244 | Competitive ensemble score |

---

## 🛠️ Feature Engineering Strategy

Four domain-specific interaction features were engineered to capture property utility and layout efficiency:

1. **`area_per_bedroom`**: `area / bedrooms`  
   Measures living space density and bedroom spaciousness.
2. **`total_rooms`**: `bedrooms + bathrooms`  
   Represents total private indoor functional spaces.
3. **`amenities_score`** (Range: 0 – 6):  
   Sum of affirmative binary infrastructure flags:
   $$\text{amenities\_score} = \text{mainroad} + \text{guestroom} + \text{basement} + \text{hotwaterheating} + \text{airconditioning} + \text{prefarea}$$
4. **`area_per_bathroom`**: `area / bathrooms`  
   Reflects luxury ratio and plumbing infrastructure scale.

> **Impact**: Adding these 4 features increased total feature dimensions from **13 to 17**, raising the Linear Regression $R^2$ score to **0.6611** and reducing RMSE by **₹15,689**.

---

## 🔍 Feature Importance Ranking

Based on the Random Forest feature importance analysis:
1. **`area`**: Dominant driver of property price (> 45% importance weight).
2. **`bathrooms`**: Significant indicator of home class and valuation.
3. **`airconditioning_yes`**: Highest premium among individual amenities.
4. **`stories`**: Multi-level properties command noticeable premiums.
5. **`parking`**: Urban utility space directly correlating with price.

---

## 🚀 How to Run in Google Colab

1. **Launch Google Colab**: Open [colab.research.google.com](https://colab.research.google.com/).
2. **Upload Notebook**: Go to `File` > `Upload notebook` and upload [`House_Price_Prediction.ipynb`](file:///c:/Users/HANAN/Downloads/House%20Price%20Prediction/House_Price_Prediction.ipynb).
3. **Upload Dataset**:
   - In Colab's left sidebar, click the **Folder icon (Files)**.
   - Upload `Housing.csv` (or `Housing.xlsx`).
4. **Run Cells**:
   - Run cells sequentially from top to bottom.
   - The notebook installs `xgboost` automatically (`!pip install xgboost -q`).

---

## 💡 Notes & Recommended Fixes

When executing the notebook on a fresh Colab runtime:
1. **File Format Support (Cell 3)**:
   The notebook currently calls `pd.read_excel("Housing.xlsx")`. If you are using `Housing.csv`, change cell 3 to:
   ```python
   import os
   if os.path.exists("Housing.xlsx"):
       df = pd.read_excel("Housing.xlsx")
   else:
       df = pd.read_csv("Housing.csv")
   ```
2. **Scaling & Tuned Comparison (Cell 26)**:
   Cell 26 references `X_train_scaled`, `X_test_scaled`, `best_rf`, and `best_xgb`. To prevent a `NameError` on a clean restart:
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)

   best_rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train_scaled, y_train)
   best_xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42).fit(X_train_scaled, y_train)
   ```

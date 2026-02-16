# House Price Prediction — Machine Learning Pipeline

This project builds a complete machine learning regression pipeline to predict housing prices using structured real-estate data.  
The workflow includes data cleaning, feature engineering, model comparison, hyperparameter tuning, and ensemble learning.

The project was implemented using Python and multiple regression algorithms to identify the best-performing model.

---

## Dataset
The dataset contains housing information such as:

- area_type
- availability
- location
- size (BHK / Bedroom)
- total_sqft
- bath
- balcony
- price (target variable)

Files:
- train.csv — training dataset
- test.csv — test dataset
- sample_submission.csv — submission format

---

## Project Workflow

### 1. Data Preprocessing
- Loaded dataset using Pandas
- Identified numerical and categorical features
- Handled missing values using median imputation
- Checked duplicates
- Detected outliers using IQR method

---

### 2. Exploratory Data Analysis (EDA)
- Price distribution analysis
- Correlation heatmap
- Price comparison across:
  - location
  - area type
  - size
  - availability

Key observation:
Total square footage and number of bathrooms strongly influence price.

---

### 3. Feature Engineering
Created 20+ new features, including:

- BHK extraction from size column
- sqft_per_bhk
- bath_per_bhk
- balcony_per_bhk
- total_rooms
- sqft_per_room
- polynomial features
- interaction features
- luxury property indicator
- location target encoding
- area_type encoding
- size target encoding

These features significantly improved model performance.

---

### 4. Models Trained
Multiple regression models were trained and compared:

- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- HistGradientBoosting Regressor

Evaluation metrics:
- RMSE
- R² Score

---

### 5. Hyperparameter Tuning
RandomizedSearchCV was used to tune XGBoost parameters, improving validation performance.

Best model:
XGBoost (Tuned)

Validation Performance:
- RMSE ≈ 56
- R² ≈ 0.82

---

### 6. Ensemble Learning
A weighted ensemble of top models was implemented:

- XGBoost (Tuned)
- XGBoost
- CatBoost
- Extra Trees

Final validation performance:

RMSE ≈ 55  
R² ≈ 0.83

---

## Output
The pipeline generates a Kaggle submission file:


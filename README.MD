# 🏠 Advanced House Price Prediction Using Machine Learning

An end-to-end **Machine Learning regression project** for predicting residential house prices using property characteristics such as location, overall quality, living area, basement, garage, kitchen quality, and other structural features.

The project focuses not only on model building but also on **domain analysis, statistical testing, missing-value treatment, outlier analysis, multicollinearity detection, feature engineering, feature encoding, model comparison, hyperparameter tuning, model explainability, and buyer-oriented insights**.

---

## 📌 Project Overview

### Business Objective

The objective of this project is to predict the **selling price of a house (`SalePrice`)** based on its available property features.

### Machine Learning Problem

* **Problem Type:** Regression
* **Target Variable:** `SalePrice`
* **Dataset Size:** 1,460 rows × 81 columns
* **Train/Test Split:** 80% / 20%
* **Training Samples:** 1,168
* **Testing Samples:** 292

### Evaluation Metrics

The models were evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)

The target variable was transformed using `log1p(SalePrice)`, so the reported regression errors and R² values are evaluated on the transformed/log scale.

---

# 🎯 Key Highlights

* Performed detailed **domain analysis** of real-estate features.
* Analyzed **81 original features**.
* Identified numerical and categorical variables.
* Converted incorrectly typed categorical variables back to categorical representation.
* Standardized inconsistent neighborhood labels.
* Performed comprehensive missing-value analysis.
* Used **median/mean imputation** based on distribution characteristics.
* Used **mode imputation** for categorical variables.
* Investigated skewness and kurtosis.
* Applied **Yeo-Johnson Power Transformation** to highly skewed numerical variables.
* Investigated numerical multicollinearity using **Pearson correlation and VIF**.
* Investigated categorical relationships using **Cramér's V and Chi-Square testing**.
* Investigated categorical relationships with `SalePrice` using **Mann-Whitney U and Kruskal-Wallis tests**.
* Evaluated the impact of removing outliers.
* Used **RobustScaler** because removing outliers with the IQR rule would eliminate approximately 48.49% of the observations.
* Performed feature engineering and feature selection.
* Used **Neighborhood Target Encoding**.
* Used **Ordinal Encoding** for ordered quality variables.
* Used **One-Hot Encoding** for selected categorical variables.
* Compared **11 regression models**.
* Evaluated baseline, intermediate, and advanced models.
* Tested Voting and Stacking ensembles.
* Performed **GridSearchCV hyperparameter tuning**.
* Selected tuned Linear Regression as the final model.
* Performed 5-fold cross-validation.
* Used **SHAP** for model explainability.
* Generated actual-vs-predicted and residual diagnostic plots.
* Derived practical buyer recommendations from the analysis.

---

# 🏡 Domain Understanding

House prices are influenced by multiple categories of property characteristics.

### Property & Land

Important features include:

* `LotArea`
* `LotFrontage`
* `Neighborhood`
* `LotShape`
* `LandContour`
* `LandSlope`
* `LotConfig`

Location was identified as an important factor because properties in desirable neighborhoods can command significantly higher prices.

### House Quality

Important quality-related features include:

* `OverallQual`
* `OverallCond`
* `ExterQual`
* `KitchenQual`
* `BsmtQual`
* `HeatingQC`

Among these, `OverallQual` was identified as one of the strongest price-related variables.

### Living Area

Important size-related features include:

* `GrLivArea`
* `TotalBsmtSF`
* `GarageArea`
* `WoodDeckSF`
* `OpenPorchSF`

Larger usable living areas generally correspond to higher property prices.

### Garage

Garage-related features include:

* `GarageCars`
* `GarageArea`
* `GarageYrBlt`
* `GarageFinish`

Garage capacity and size were identified as meaningful value-adds.

---

# 📊 Dataset

The project uses the Ames Housing dataset containing detailed information about residential properties.

The original dataset contains:

```text
Rows    : 1,460
Columns : 81
Target  : SalePrice
```

The dataset includes numerical and categorical information describing:

* Property size
* Property quality
* Location
* Construction
* Basement
* Garage
* Kitchen
* Bathrooms
* Bedrooms
* Exterior
* Heating
* Fireplaces
* Outdoor features
* Sale information

---

# 🔎 Exploratory Data Analysis

The project performs several levels of exploratory analysis.

## 1. Data Validation

The dataset was inspected for:

* Data types
* Unique values
* Numerical ranges
* Missing values
* Duplicate records
* Skewness
* Kurtosis
* Categorical distributions

---

## 2. Data Type Correction

Several variables represented categorical information even though they were initially stored as numerical values.

Examples include:

```text
MSSubClass
OverallQual
OverallCond
BsmtFullBath
BsmtHalfBath
FullBath
HalfBath
BedroomAbvGr
KitchenAbvGr
TotRmsAbvGrd
Fireplaces
GarageCars
PoolArea
MiscVal
MoSold
```

These variables were converted into categorical representations before the preprocessing stage.

---

# 🧹 Data Cleaning

## Neighborhood Standardization

An inconsistent neighborhood label was identified:

```text
NAmes → NWAmes
```

This prevented the same neighborhood from being treated as two separate categories.

---

# 🕳️ Missing Value Handling

Missing values were analyzed based on the distribution of each variable.

### Numerical Features

For numerical variables:

* **Median imputation** was used when the variable was highly skewed.
* **Mean imputation** was used when the distribution was relatively normal.

### Categorical Features

Categorical missing values were replaced using the **mode**.

This approach preserved observations instead of removing rows containing missing values.

---

# 📈 Skewness & Kurtosis Analysis

Numerical distributions were analyzed using:

* Skewness
* Kurtosis
* Histograms

Several variables showed significant right-skewness and extreme values.

Examples include:

* `LotArea`
* `3SsnPorch`
* `LowQualFinSF`
* `BsmtFinSF2`
* `ScreenPorch`
* `EnclosedPorch`
* `OpenPorchSF`

Instead of aggressively deleting these observations, the project applied a transformation-based approach.

---

# 🔄 Power Transformation

Highly skewed numerical variables were transformed using:

```python
PowerTransformer(method="yeo-johnson")
```

The Yeo-Johnson transformation was selected because it can work with values that may include zero or negative values.

The project compared the original and transformed distributions to understand the effect of the transformation.

---

# 🚨 Outlier Analysis

Outliers were initially investigated using the **IQR method**.

The analysis showed:

```text
Original Rows       : 1,460
Rows Remaining      : 752
Rows Removed        : 708
Data Loss           : 48.49%
```

Removing outliers would therefore eliminate almost half of the available dataset.

### Decision

Instead of removing these observations, the project used:

```text
Power Transformation
+
Robust Scaling
```

This approach preserves more training data while reducing the influence of extreme values.

---

# 🔗 Multicollinearity Analysis

Multicollinearity was investigated among numerical variables using:

* Pearson correlation
* Variance Inflation Factor (VIF)

Some features were found to represent information already contained in other features.

### Basement Relationship

The project verified:

```text
TotalBsmtSF =
BsmtFinSF1 +
BsmtFinSF2 +
BsmtUnfSF
```

Because these variables contain overlapping information, the following features were removed:

```text
BsmtFinSF1
BsmtFinSF2
BsmtUnfSF
```

while retaining:

```text
TotalBsmtSF
```

### Living Area Relationship

Similarly:

```text
GrLivArea =
1stFlrSF +
2ndFlrSF +
LowQualFinSF
```

Therefore:

```text
1stFlrSF
2ndFlrSF
LowQualFinSF
```

were removed while retaining:

```text
GrLivArea
```

This reduced redundant information and helped control multicollinearity.

---

# 📊 Numerical Feature Relationship With SalePrice

Several numerical variables showed meaningful relationships with `SalePrice`.

Selected correlations from the analysis include:

| Feature       | Correlation |
| ------------- | ----------: |
| `GrLivArea`   |       0.709 |
| `GarageArea`  |       0.623 |
| `TotalBsmtSF` |       0.614 |
| `MasVnrArea`  |       0.473 |
| `GarageYrBlt` |       0.470 |
| `LotFrontage` |       0.335 |
| `WoodDeckSF`  |       0.324 |
| `OpenPorchSF` |       0.316 |
| `LotArea`     |       0.264 |

`GrLivArea` showed the strongest numerical correlation with `SalePrice` among the variables examined in this analysis.

---

# 🔤 Categorical Multicollinearity

Categorical feature relationships were investigated using:

### Cramér's V

A Cramér's V matrix was created to identify strongly related categorical variables.

Some notable relationships included:

| Feature 1     | Feature 2      | Cramér's V |
| ------------- | -------------- | ---------: |
| `MSSubClass`  | `BldgType`     |      0.896 |
| `MSSubClass`  | `HouseStyle`   |      0.852 |
| `Exterior1st` | `Exterior2nd`  |      0.762 |
| `MSZoning`    | `Neighborhood` |      0.652 |
| `OverallQual` | `ExterQual`    |      0.619 |
| `ExterQual`   | `KitchenQual`  |      0.547 |
| `OverallQual` | `KitchenQual`  |      0.545 |

These relationships were investigated before the final feature set was constructed.

---

# 📐 Statistical Testing

Categorical variables were analyzed against `SalePrice` using statistical tests.

### Two Groups

For variables containing two groups:

```text
Mann-Whitney U Test
```

was used.

### Multiple Groups

For variables containing more than two groups:

```text
Kruskal-Wallis Test
```

was used.

Effect sizes were also calculated to identify variables with stronger relationships with the target.

---

# 🛠️ Feature Engineering

Several transformations were performed.

### 3-Season Porch

Because most observations had the same value, `3SsnPorch` was converted into a binary feature:

```text
0 → No 3-season porch
1 → Has 3-season porch
```

### Selected Features

Statistical analysis was used to construct an important categorical feature set containing variables such as:

```text
OverallQual
Neighborhood
GarageCars
ExterQual
KitchenQual
BsmtQual
FullBath
GarageFinish
MSSubClass
Foundation
CentralAir
TotRmsAbvGrd
Fireplaces
HeatingQC
```

---

# 🔀 Train-Test Split

The target variable was transformed using:

```python
df["SalePrice"] = np.log1p(df["SalePrice"])
```

The data was then split using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Result:

```text
Training Set : 1,168 samples
Testing Set  :   292 samples
```

---

# ⚙️ Feature Encoding

Different encoding strategies were selected according to the nature of each variable.

## Target Encoding

`Neighborhood` was encoded using:

```python
category_encoders.TargetEncoder
```

with smoothing:

```text
smoothing = 10
```

The encoder was fitted only on the training data and then applied to the test data.

---

## One-Hot Encoding

One-hot encoding was applied to:

```text
Foundation
CentralAir
```

with:

```python
handle_unknown="ignore"
drop="first"
```

---

## Ordinal Encoding

Ordinal encoding was used for ordered quality-related features:

```text
ExterQual
KitchenQual
BsmtQual
GarageFinish
HeatingQC
```

The ordinal relationships were explicitly defined.

---

# 📏 Numerical Preprocessing Pipeline

Numerical variables were processed using:

```text
Yeo-Johnson Power Transformation
            ↓
RobustScaler
```

### Why?

The dataset contains highly skewed variables and significant outliers.

`RobustScaler` was selected because it is less sensitive to extreme observations than standard scaling.

---

# 🧩 Preprocessing Architecture

The project uses a `ColumnTransformer` containing separate pipelines:

```text
                    Input Features
                          │
              ┌───────────┴───────────┐
              │                       │
        Numerical Features      Categorical Features
              │                       │
       Yeo-Johnson Transform     ┌────┴─────┐
              │                  │          │
         RobustScaler       One-Hot      Ordinal
              │             Encoding     Encoding
              │                  │          │
              └──────────┬───────┴──────────┘
                         │
                  Final Preprocessed Data
```

---

# 🎯 Feature Selection

Random Forest feature importance was used to investigate the relative importance of the transformed features.

The project also considered:

* Statistical relationships
* Correlation
* Multicollinearity
* Domain knowledge
* Feature redundancy
* Model importance

This resulted in a more focused feature representation.

---

# 🤖 Machine Learning Models

The project compared **11 regression models**.

## Baseline Models

* Linear Regression
* Lasso Regression
* Ridge Regression
* ElasticNet
* K-Nearest Neighbors Regressor

## Intermediate Models

* Random Forest Regressor
* Decision Tree Regressor
* Support Vector Regressor

## Advanced Models

* XGBoost Regressor
* CatBoost Regressor
* LightGBM Regressor

---

# 📊 Model Comparison

| Model             | Train R² |   Test R² |  Test MAE | Test RMSE |
| ----------------- | -------: | --------: | --------: | --------: |
| Linear Regression |    0.864 |     0.864 |     0.112 |     0.159 |
| Lasso             |    0.847 |     0.853 |     0.114 |     0.165 |
| Ridge             |    0.864 |     0.864 |     0.112 |     0.159 |
| ElasticNet        |    0.856 |     0.861 |     0.112 |     0.161 |
| KNN               |    0.853 |     0.794 |     0.133 |     0.196 |
| Random Forest     |    0.977 |     0.859 |     0.110 |     0.162 |
| Decision Tree     |    1.000 |     0.754 |     0.153 |     0.214 |
| SVR               |    0.825 |     0.828 |     0.123 |     0.179 |
| XGBoost           |    0.999 |     0.851 |     0.114 |     0.167 |
| CatBoost          |    0.985 | **0.882** | **0.099** | **0.148** |
| LightGBM          |    0.980 |     0.870 |     0.105 |     0.156 |

> **Note:** The metrics above are calculated on the log-transformed `SalePrice` target.

---

# ⚠️ Model Overfitting Analysis

Tree-based and boosting models achieved extremely high training scores.

For example:

```text
Decision Tree
Train R² ≈ 1.000
Test R²  ≈ 0.754
```

and:

```text
XGBoost
Train R² ≈ 0.999
Test R²  ≈ 0.851
```

This indicates substantial overfitting.

The project identified the relatively small dataset size as one contributing factor.

---

# 🏆 CatBoost Performance

Among the untuned models, CatBoost achieved the highest raw test R²:

```text
Test R²   : 0.882
Test MAE  : 0.099
Test RMSE : 0.148
```

However, CatBoost also showed a considerably larger train-test performance gap compared with the linear models.

Therefore, the project prioritized **generalization and stability**, rather than simply selecting the highest raw test score.

---

# 🧪 Ensemble Learning

Two ensemble approaches were tested:

### Voting Regressor

```text
Test R²   : 0.864
Test MAE  : 0.111
Test RMSE  : 0.159
```

### Stacking Regressor

```text
Test R²   : 0.864
Test MAE  : 0.111
Test RMSE  : 0.159
```

The ensemble approaches did not outperform the individual linear models.

### Decision

A single tuned Linear Regression model was therefore preferred.

---

# 🔧 Hyperparameter Tuning

`GridSearchCV` with 10-fold cross-validation was used for:

* Linear Regression
* Ridge
* ElasticNet
* SVR

## Best Parameters

### Linear Regression

```python
{
    "fit_intercept": True,
    "positive": True
}
```

Best CV R²:

```text
0.8584
```

### Ridge

```python
alpha ≈ 16.7683
```

Best CV R²:

```text
0.8476
```

### ElasticNet

```python
alpha ≈ 0.00695
l1_ratio = 0.1
```

Best CV R²:

```text
0.8477
```

### SVR

```python
C = 10
epsilon = 0.1
gamma = "scale"
kernel = "rbf"
```

Best CV R²:

```text
0.8450
```

---

# 🏆 Final Model

The final selected model is:

```python
LinearRegression(
    fit_intercept=True,
    positive=True
)
```

### Final Training Performance

```text
R²    : 0.8719
MAE   : 0.0986
MSE   : 0.0195
RMSE  : 0.1397
```

### Final Testing Performance

```text
R²    : 0.8809
MAE   : 0.1042
MSE   : 0.0222
RMSE  : 0.1491
```

### Final Result

```text
Test R² ≈ 0.881
```

The tuned model improved the linear regression performance from approximately:

```text
86% → 88%
```

in terms of R².

---

# 🔄 Cross-Validation

The final model was further evaluated using 5-fold cross-validation.

### Fold Scores

```text
0.8765
0.8472
0.8267
0.8931
0.8674
```

### Mean CV R²

```text
0.8622
```

### Standard Deviation

```text
0.0231
```

The relatively small standard deviation indicates reasonable stability across folds.

---

# 🔍 Model Explainability With SHAP

SHAP was used to investigate how features contribute to the model predictions.

The project used:

```python
shap.LinearExplainer
```

to calculate SHAP values for the final Linear Regression model.

This provides an interpretable view of how individual features influence predicted house prices.

---

# 📉 Model Diagnostics

The final model was evaluated using several diagnostic visualizations:

### Actual vs Predicted

Used to determine how closely predictions follow the actual values.

### Residuals vs Predicted

Used to inspect systematic residual patterns.

### Residual Distribution

Used to understand the distribution of prediction errors.

### Q-Q Plot

Used to inspect whether residuals approximately follow a normal distribution.

---

# 💡 Key Business Insights

The analysis produced several important insights.

### 1. Location Matters

`Neighborhood` was identified as a major source of price variation.

A desirable location can significantly influence property value.

### 2. Overall Quality Is Critical

`OverallQual` was identified as one of the most important price drivers.

Higher-quality construction and materials generally correspond to higher selling prices.

### 3. Living Area Matters

`GrLivArea` showed a strong relationship with `SalePrice`.

Larger above-ground living space generally increases property value.

### 4. Basement Adds Value

`TotalBsmtSF` demonstrated a meaningful positive relationship with house price.

### 5. Garage Features Matter

Garage size and capacity can contribute to property value.

### 6. Quality Features Matter

Features such as:

```text
KitchenQual
ExterQual
BsmtQual
HeatingQC
```

provide additional information about property quality.

### 7. Central Air Matters

Properties with central air conditioning were associated with a price premium in the analysis.

### 8. Not Every Amenity Has Equal Impact

Features such as:

```text
Porches
Screened porches
Pools
```

showed comparatively weaker relationships with price.

---

# 🏠 Buyer Recommendations

Based on the analysis, buyers can use the following strategy:

### Step 1 — Start With Neighborhood

Compare properties within the same neighborhood before comparing price.

### Step 2 — Check Overall Quality

Use `OverallQual` as one of the first indicators of property quality.

### Step 3 — Compare Living Area

Use `GrLivArea` when comparing the usable above-ground living space.

### Step 4 — Evaluate Basement Separately

Consider `TotalBsmtSF` as an independent value-add.

### Step 5 — Evaluate Garage Capacity

Compare:

```text
GarageCars
GarageArea
```

rather than simply checking whether a property has a garage.

### Step 6 — Inspect Individual Quality Ratings

Check:

```text
KitchenQual
ExterQual
BsmtQual
HeatingQC
```

instead of relying only on the overall quality score.

### Step 7 — Compare Asking Price With Model Prediction

A listing's asking price can be sanity-checked using:

```text
Neighborhood
OverallQual
GrLivArea
Garage
Basement
```

---

# 🧰 Technology Stack

### Programming

* Python

### Data Manipulation

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* CatBoost
* LightGBM

### Statistical Analysis

* SciPy
* Statsmodels

### Feature Engineering

* Category Encoders
* PowerTransformer
* RobustScaler
* OneHotEncoder
* OrdinalEncoder

### Model Explainability

* SHAP

### Model Persistence

* Pickle

---

# 📚 Python Libraries

Main libraries used in the project include:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
statsmodels
xgboost
catboost
lightgbm
category_encoders
shap
```

---

# 📂 Project Structure

A recommended GitHub repository structure is:

```text
Advanced-House-Price-Prediction-Using-Machine-Learning/
│
├── Data/
│   └── data.csv
│
├── house_price_prediction_regression.ipynb
│
├── linear_regression_model.pkl
│
├── preprocesor.pkl
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone https://github.com/ramnallapati/Advanced-House-Price-Prediction-Using-Machine-Learning.git
```

## 2. Navigate to the Project

```bash
cd Advanced-House-Price-Prediction-Using-Machine-Learning
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
house_price_prediction_regression.ipynb
```

Run the notebook cells sequentially.

---

# 🔁 End-to-End Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Data Validation
     │
     ▼
Domain Analysis
     │
     ▼
EDA
     │
     ├── Missing Value Analysis
     ├── Skewness / Kurtosis
     ├── Categorical Analysis
     └── Outlier Analysis
     │
     ▼
Data Cleaning
     │
     ├── Type Correction
     ├── Category Standardization
     └── Constant Feature Removal
     │
     ▼
Multicollinearity Analysis
     │
     ├── Pearson Correlation
     ├── VIF
     └── Cramér's V
     │
     ▼
Feature Engineering
     │
     ├── Feature Transformation
     ├── Feature Selection
     └── Redundant Feature Removal
     │
     ▼
Train-Test Split
     │
     ▼
Target Transformation
     │
     ▼
Feature Encoding
     │
     ├── Target Encoding
     ├── One-Hot Encoding
     └── Ordinal Encoding
     │
     ▼
Power Transformation
     │
     ▼
Robust Scaling
     │
     ▼
Model Training
     │
     ├── Linear Models
     ├── Tree Models
     └── Boosting Models
     │
     ▼
Model Comparison
     │
     ▼
Ensemble Evaluation
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Final Linear Regression
     │
     ▼
Cross Validation
     │
     ▼
SHAP Explainability
     │
     ▼
Model Diagnostics
     │
     ▼
Business Insights
```

---

# 📈 Final Results Summary

| Metric          |             Final Model |
| --------------- | ----------------------: |
| Model           | Tuned Linear Regression |
| `fit_intercept` |                    True |
| `positive`      |                    True |
| Train R²        |                   0.872 |
| Test R²         |               **0.881** |
| Test MAE        |                   0.104 |
| Test MSE        |                   0.022 |
| Test RMSE       |                   0.149 |
| Mean CV R²      |                   0.862 |
| CV Std          |                   0.023 |

---

# 🎓 Skills Demonstrated

This project demonstrates practical experience in:

* Python Programming
* Data Cleaning
* Exploratory Data Analysis
* Statistical Analysis
* Feature Engineering
* Feature Selection
* Missing Value Treatment
* Outlier Analysis
* Multicollinearity Detection
* VIF Analysis
* Cramér's V
* Chi-Square Testing
* Mann-Whitney U Testing
* Kruskal-Wallis Testing
* Target Encoding
* One-Hot Encoding
* Ordinal Encoding
* Power Transformation
* Robust Scaling
* Regression Modeling
* Ensemble Learning
* Hyperparameter Optimization
* Cross-Validation
* Model Evaluation
* SHAP Explainability
* Residual Analysis
* Business Interpretation

---

# 📌 Project Conclusion

This project demonstrates an end-to-end approach to solving a real-estate price prediction problem.

Rather than selecting a model solely based on the highest training or test score, the project investigated **data quality, feature relationships, multicollinearity, outliers, overfitting, model stability, and interpretability**.

Although CatBoost achieved the highest raw test R² among the untuned models, the project selected a **hyperparameter-tuned Linear Regression model** because of its strong generalization, small train-test performance gap, interpretability, and stable cross-validation performance.

The final model achieved approximately:

```text
Test R² = 0.881
```

with a 5-fold cross-validation mean R² of approximately:

```text
0.862
```

This makes the final solution both **predictive and interpretable**, while providing useful insights into the factors that influence residential property prices.

---

# 👨‍💻 Author

**Rama Rao Nallapati**

Aspiring **AI Engineer | Machine Learning Engineer | Data Scientist**

### Areas of Interest

```text
Artificial Intelligence
Machine Learning
Generative AI
Agentic AI
Deep Learning
Data Science
Python
SQL
```

---

⭐ If you found this project useful, consider giving the repository a **star**!

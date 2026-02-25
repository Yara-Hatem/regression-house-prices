# Pakistan House Price Prediction — Regression

A machine learning project that predicts property prices in Pakistan's real estate market using listing data scraped from [Zameen.com](https://www.zameen.com). Three regression models are trained, evaluated, and compared.

---

## Project Overview

This is a supervised regression problem built on a dataset of ~168,000 property listings across five major Pakistani cities. The goal is to predict the continuous `price` variable from property attributes such as size, location, type, and number of rooms.

**Target variable:** `price` (in PKR)

---

## Dataset

| Property | Value |
|---|---|
| Rows | 168,446 |
| Features | 20 |
| Task | Regression |
| Source | Zameen.com listings |
| Cities | Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad |

**Feature types:**
- **Numerical:** price (target), latitude, longitude, baths, bedrooms, Area Size
- **Categorical:** property_type, city, province, location, purpose (For Sale / For Rent), Area Type, Area Category, agency, agent
- **Identifiers (dropped):** property_id, location_id, page_url

**Notable data characteristics:**
- ~11.85% missing values overall (artificially augmented with 10% nulls for preprocessing practice); imputed using median
- Significant outliers: prices up to 2 billion PKR, area sizes up to 800 Marla — handled during preprocessing
- Label encoding applied to categorical features
- Feature selection performed using Sequential Backward Selection (SBS)

---

## Key EDA Findings

- **Lahore** has the highest average property prices among the five cities
- **Karachi** is the second most expensive market
- **Faisalabad and Rawalpindi** are the most affordable markets
- Price distribution is heavily right-skewed due to luxury property outliers

---

## Models

Three regressors were trained and evaluated, each with SBS-based feature selection:

- **Linear Regression**
- **Decision Tree Regressor** (tuned via GridSearchCV on `max_depth`, `min_samples_split`, `min_samples_leaf`)
- **K-Nearest Neighbors Regressor**

---

## Results

| Model | R² Score | MAE (PKR) |
|---|---|---|
| **KNN** | **0.7666** | — |
| Decision Tree | 0.6365 | 6,776,667 |
| Linear Regression | 0.1967 | 14,825,863 |

KNN achieved the best R² score after feature selection, using: `property_type`, `latitude`, `longitude`, `purpose`, `Area Type`, and `Area Size`. Linear Regression struggled significantly due to the non-linear nature of real estate pricing. Learning curves were generated for all models to diagnose bias/variance trade-offs.

---

## Tech Stack

- Python 3.12
- pandas, NumPy
- scikit-learn
- mlxtend (Sequential Feature Selector)
- matplotlib, seaborn
- Jupyter Notebook

---

## Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn mlxtend matplotlib seaborn jupyter
   ```

3. **Add the dataset**  
   Download the [Pakistan House Prices dataset](https://www.kaggle.com/datasets/your-link) and place it in the project directory.

4. **Launch the notebook**
   ```bash
   jupyter notebook Regression_-_House_Prices.ipynb
   ```





## 📌 Key Takeaways

- KNN outperforms linear models substantially, confirming that house price patterns in this dataset are non-linear.
- Location-related features (`latitude`, `longitude`, `area`) and property size (`Area Size`, `Area Type`) are the most predictive.
- Linear Regression's low R² (≈0.20) highlights the limitations of linear assumptions on real estate data with heavy skew and outliers.

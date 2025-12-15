# Crop Yield Forecasting: An End-to-End Machine Learning Project

This repository documents my complete workflow for a comprehensive **crop yield forecasting project**. The goal was to build a robust machine learning model capable of predicting agricultural yields by integrating diverse global datasets. The project spans from data acquisition and extensive preprocessing to advanced model training, hyperparameter tuning, and final deployment in an interactive web application.

---

## Project Summary

I developed a predictive pipeline that leverages agricultural, economic, and climatic data to forecast crop yields. The final model, a **Random Forest Regressor**, achieved an overall **R² score of 0.920**, demonstrating a significant improvement over baseline models and proving its effectiveness in capturing the complex interactions between various factors. The project culminates in an interactive Streamlit dashboard for easy exploration and prediction.

### The Workflow at a Glance

1.  **Data Acquisition:** Fetched data from multiple APIs, including FAOSTAT for agricultural inputs and the World Bank for climatic and economic indicators.
2.  **Preprocessing & Integration:** Cleaned, standardized, and merged disparate datasets into a single, analysis-ready master file.
3.  **Feature Engineering:** Created new, impactful features, such as pivoting fertilizer data and grouping hundreds of crops into logical categories.
4.  **Exploratory Data Analysis (EDA):** Visualized trends, correlations, and distributions to uncover key relationships within the data.
5.  **Baseline Modeling:** Established initial performance metrics using Linear Regression, Decision Trees, and Random Forest.
6.  **Advanced Modeling:** Trained and compared sophisticated gradient boosting models (XGBoost, LightGBM, CatBoost) against an optimized Random Forest.
7.  **Hyperparameter Tuning:** Systematically optimized the best models using Bayesian Optimization and Grid Search to maximize performance.
8.  **Deployment:** Built an interactive dashboard with Streamlit to visualize historical data and showcase the model's predictive power.

---

## Data Acquisition and Integration

The foundation of this project is a master dataset constructed from two primary sources:

| Data Source | Datasets Acquired | Scope |
| :--- | :--- | :--- |
| **FAOSTAT API** | Crop Yield, Fertilizer Use (by nutrient), Pesticide Use | Agricultural production and inputs from 2015-2023. |
| **World Bank API**| Historical Temperature & Rainfall, GDP, Agricultural Land % | Climatic and economic indicators from 1901-2022. |

The integration process was critical. I performed several key steps to create a unified and clean dataset:
*   **Standardized Country Names:** Mapped different naming conventions between FAOSTAT and the World Bank to ensure accurate merging.
*   **Filtered Regional Aggregates:** Removed summary entries (e.g., 'Africa', 'World') to focus the analysis on individual countries.
*   **Handled Missing Values:** Employed a multi-step strategy of interpolation, back-filling/forward-filling by country, and finally imputing with regional medians to ensure data integrity without introducing significant bias.

---

## Feature Engineering & Exploratory Data Analysis

To enhance the model's predictive power, I engineered several new features and performed a deep exploratory analysis.

### Key Feature Engineering Steps
*   **Pivoted Fertilizer Data:** Transformed the long-format fertilizer data into a wide format, creating distinct columns for each primary nutrient (`Nitrogen N`, `Phosphate P2O5`, `Potash K2O`).
*   **Aggregated Pesticide Data:** Focused on the total pesticide usage in tonnes, creating a single, powerful feature for pesticide impact.
*   **Categorized Crops:** Grouped over 190 unique crops into 10 broader categories (e.g., Cereals, Fruits, Vegetables). This helps the model learn generalized patterns instead of memorizing patterns for each specific crop.

### Exploratory Data Analysis (EDA) Highlights
My analysis revealed several important trends:
*   A clear upward trend in global average yield over the years.
*   Significant correlations between yield and factors like fertilizer use, temperature, and rainfall.
*   Distinct yield distributions across different crop categories and geographical regions.

<!-- Placeholder for EDA visualizations -->
**Top 10 Crops by Data Volume**
`![Top 10 Crops](path/to/your/top_crops_plot.png)`

**Smoothed Yield Trends for Major Crops (3-Year Rolling Avg)**
`![Yield Trends](path/to/your/yield_trends_plot.png)`

**Global Consumption Trends**
`![Consumption Trends](path/to/your/consumption_trends_plot.png)`

---

## Modeling & Evaluation

I adopted a category-specific modeling approach, training a separate model for each major crop category. This strategy allows each model to specialize in the unique factors that influence yields for its respective category.

### Baseline Models
First, I established baseline performance to measure against. The Random Forest model showed early promise.

| Model | Overall R² | Overall RMSE (kg/ha) | Overall MAE (kg/ha) |
| :--- | :---: | :---: | :---: |
| Linear Regression | 0.626 | 7,535 | 3,805 |
| Decision Tree | 0.673 | 7,040 | 3,623 |
| **Random Forest (Initial)** | **0.904** | **3,811** | **1,655** |

### Advanced Models & Final Selection
Next, I trained four powerful tree-based models on the engineered features and compared their performance across all crop categories.

| Model | Overall R² | Overall RMSE (kg/ha) | Overall MAE (kg/ha) |
| :--- | :---: | :---: | :---: |
| **Random Forest** | **0.920** | **3,487** | **1,437** |
| XGBoost | 0.896 | 3,979 | 1,822 |
| LightGBM | 0.825 | 5,147 | 2,445 |
| CatBoost | 0.789 | 5,665 | 2,711 |

The **Random Forest** model consistently outperformed the others, achieving the highest overall R² score and the lowest error metrics. This model was selected for the final dashboard.

<!-- Placeholder for Model Comparison Bar Charts -->
**Overall Model Performance Comparison**
`![Overall Performance](path/to/your/overall_performance_plot.png)`

**R² Score by Crop Category**
`![R2 by Category](path/to/your/r2_by_category_plot.png)`

---

## Model Interpretation with SHAP

To understand what drives the model's predictions, I used SHAP (SHapley Additive exPlanations). The analysis confirmed that **Crop Type** was the most influential feature, followed by **Country**. This highlights the success of the category-specific modeling and the importance of location-based factors. Climatic and agricultural input features like **Temperature**, **Rainfall**, and **Fertilizer Use** also ranked highly.

<!-- Placeholder for SHAP Summary Plots -->
**SHAP Feature Importance (Bar Plot)**
`![SHAP Bar Plot](path/to/your/shap_bar_plot.png)`

---

## Interactive Dashboard

The final output is an interactive dashboard built with Streamlit. It allows users to:
*   Filter data by **Country**, **Crop**, and **Year Range**.
*   View key performance indicators like average yield and recent trends.
*   Analyze historical yield charts with trend lines.
*   Explore the relationship between yield and environmental factors through interactive scatter plots.
*   Examine correlations between different features.
*   Review the raw data in a clean, filterable table.

<!-- Placeholder for Dashboard Screenshot -->
**Dashboard Preview**
`![Dashboard Screenshot](path/to/your/dashboard_screenshot.png)`

---

## Setting Up the Environment

Make **"agro-climatic-forecasting"** as your current working directory.

```bash
cd agro-climatic-forecasting
```

To avoid package conflicts, it is highly recommended to create a **virtual environment**.

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

---

## Install Dependencies

With the virtual environment activated, install all required packages using the requirements.txt file.

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit Dashboard

Once the environment is set up and all dependencies are installed, you can launch the interactive dashboard.

```bash
streamlit run dashboard.py
```
---

## Datasets

You can find all the data files required for this project in the link provided below.

[Download Datasets from Google Drive](https://drive.google.com/drive/folders/1ANgX_G0LGUk5ZOjarP3l4t6qfedRIbMT?usp=sharing)

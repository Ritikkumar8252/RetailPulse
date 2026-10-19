# RetailPulse
### AI-Powered Customer Analytics & Demand Forecasting Platform

RetailPulse is a data analytics and demand forecasting platform designed to help businesses understand customer purchasing behaviour, analyse sales performance, segment customers, and estimate future product demand.

The project combines data preprocessing, exploratory data analysis, RFM-based customer segmentation, time-series forecasting, model evaluation, and an interactive Streamlit dashboard.

---

## 1. Project Objectives

The main objectives of RetailPulse are:

- Analyse historical retail sales and revenue trends.
- Identify high-value, regular, occasional, and inactive customers.
- Segment customers using Recency, Frequency, and Monetary analysis.
- Analyse product-level sales and demand patterns.
- Forecast future product demand using time-series methods.
- Compare forecasting models using MAE and RMSE.
- Present business insights through an interactive dashboard.
- Support data-informed customer retention and inventory planning.

---

## 2. Dashboard Features

RetailPulse currently includes seven dashboard sections.

### Business Overview
- Total revenue
- Total orders
- Known customers
- Units sold
- Monthly revenue trends

### Sales Analytics
- Interactive date-range filters
- Country selection
- Revenue and order metrics
- Top products by revenue
- Revenue by country
- Filtered sales records
- CSV export

### Customer Segmentation
- Customer distribution by segment
- RFM metrics
- Segment-level spending analysis
- Customer filtering
- Downloadable customer segment data

### Demand Forecasting
- Product selection
- Forecast horizons of 7, 14, and 30 days
- Historical daily demand visualisation
- Future demand estimates
- Forecast tables
- CSV export

### Model Evaluation
- Comparison of five forecasting approaches
- Average MAE and RMSE
- Model performance charts
- Evaluation summary
- Downloadable evaluation results

### Business Recommendations
- Customer retention recommendations
- Identification of inactive or at-risk customer segments
- Historical product performance analysis
- Demand forecasting recommendations

### Data Quality
- Missing-value analysis
- Duplicate record checks
- Transaction validation
- Revenue calculation validation
- Transaction date coverage
- Downloadable validation report

---

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly |
| Dashboard | Streamlit |
| Machine Learning | Scikit-learn |
| Customer Segmentation | K-Means Clustering |
| Time-Series Forecasting | Holt-Winters, Moving Average, Seasonal Naive |
| Statistical Modelling | Statsmodels |
| Development Environment | VS Code, Jupyter Notebook |

---

## 4. Dataset

The project uses the Online Retail transactional dataset.

The dataset contains historical retail transactions, including information such as:

- Invoice number
- Product code
- Product description
- Quantity
- Invoice date
- Unit price
- Customer ID
- Country

The original dataset contains 541,909 transaction records.

The cleaned positive-sales dataset contains 524,878 records.

### Data Coverage

- Start date: 1 December 2010
- End date: 9 December 2011
- Unique invoices: approximately 19,962
- Unique products: 3,922
- Known customers: 4,338
- Countries represented: 38

December 2011 is a partial month in the dataset and should be interpreted accordingly.

---

## 5. Data Cleaning and Preprocessing

The preprocessing workflow included:

1. Removing exact duplicate records.
2. Examining missing customer IDs and product descriptions.
3. Investigating cancelled transactions.
4. Identifying non-positive quantities and prices.
5. Calculating transaction revenue.
6. Creating a cleaned transaction dataset.
7. Creating a positive-sales dataset for sales analytics and forecasting.
8. Aggregating transactions into daily product demand.
9. Preparing customer-level RFM features.
10. Generating datasets for dashboard visualisations.

Revenue was calculated as:

Revenue = Quantity × Unit Price

The positive-sales dataset excludes cancellations, non-positive quantities, and non-positive prices.

Missing customer IDs were retained for overall sales analysis but excluded from customer-level RFM segmentation.

---

## 6. Customer Segmentation

RetailPulse uses RFM analysis to understand customer behaviour.

### RFM Features

**Recency:** Number of days since a customer's most recent purchase.

**Frequency:** Number of distinct invoices associated with a customer.

**Monetary:** Total historical spending by a customer.

The RFM features were transformed using log1p and standardised before applying K-Means clustering.

The selected configuration uses four clusters.

### Customer Segments

| Segment | Customers | Median Recency | Median Frequency | Median Monetary Value |
|---|---:|---:|---:|---:|
| High-Value Active | 713 | 8 days | 10 | £3,722.82 |
| Inactive / At-Risk | 1,622 | 174.5 days | 1 | £296.09 |
| Recent Occasional | 837 | 17 days | 2 | £480.31 |
| Regular Customers | 1,166 | 57 days | 4 | £1,345.20 |

These profiles support customer retention planning, identification of valuable customers, and exploration of purchasing behaviour.

The segment labels are business interpretations of the resulting clusters rather than externally validated customer classifications.

---

## 7. Demand Forecasting

RetailPulse explores multiple approaches to predicting future product demand.

The forecasting methods evaluated include:

- Mean Baseline
- Simple Exponential Smoothing
- 7-Day Moving Average
- Seasonal Naive
- Holt-Winters Exponential Smoothing

The interactive dashboard supports product selection and forecasts for the next 7, 14, or 30 days.

Holt-Winters is used when sufficient daily history is available and the model fits successfully. The dashboard falls back to a recent-average baseline when necessary.

Forecasts are estimates based on historical patterns. They are not guarantees of future sales.

---

## 8. Forecast Model Evaluation

Five forecasting approaches were evaluated across 20 products.

The following results are from the saved product-level model evaluation experiment.

| Model | Products Evaluated | Average MAE | Median MAE | Average RMSE | Median RMSE |
|---|---:|---:|---:|---:|---:|
| Holt-Winters | 20 | 87.03 | 69.91 | 160.45 | 139.97 |
| 7-Day Moving Average | 20 | 90.86 | 72.68 | 163.42 | 129.24 |
| Seasonal Naive | 20 | 90.94 | 76.25 | 175.32 | 152.91 |
| Simple Exponential Smoothing | 20 | 93.15 | 69.51 | 169.36 | 144.57 |
| Mean Baseline | 20 | 93.19 | 68.31 | 171.11 | 149.01 |

### Evaluation Metrics

**Mean Absolute Error (MAE)** measures the average absolute difference between actual and predicted demand.

**Root Mean Squared Error (RMSE)** measures prediction error while penalising larger errors more heavily.

Lower values indicate better performance.

### Experimental Findings

Holt-Winters achieved the lowest average MAE and average RMSE among the five evaluated approaches.

Compared with the Mean Baseline, its average MAE was approximately 6.4% lower.

The 7-Day Moving Average was a competitive alternative.

These results apply to the evaluated products and experimental setup. They do not establish that Holt-Winters will be the best model for every product or forecast horizon.

---

## 9. Project Structure

The main project files are organised as follows:

```text
RetailPulse_project/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── processed/
│       ├── retailpulse_cleaned_transactions.csv
│       └── retailpulse_positive_sales.csv
│
├── outputs/
│   ├── retailpulse_customer_segments.csv
│   ├── retailpulse_daily_product_demand.csv
│   ├── retailpulse_product_summary.csv
│   ├── retailpulse_forecast_evaluation.csv
│   ├── retailpulse_product_model_evaluation.csv
│   └── retailpulse_product_model_summary.csv
│
├── notebooks/
└── screenshots/
```

The `notebooks` and `screenshots` directories can be added as project documentation is prepared.

---

## 10. Installation and Setup

### Prerequisites

- Python
- pip
- Git (optional, for repository management)

### Step 1: Clone or Download the Project

Open a terminal in the project directory.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Environment

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Run the Dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

---

## 11. Expected Data Files

The application expects its processed datasets and model outputs in the following locations:

```text
data/processed/retailpulse_positive_sales.csv
data/processed/retailpulse_cleaned_transactions.csv

outputs/retailpulse_customer_segments.csv
outputs/retailpulse_daily_product_demand.csv
outputs/retailpulse_product_summary.csv
outputs/retailpulse_forecast_evaluation.csv
outputs/retailpulse_product_model_evaluation.csv
outputs/retailpulse_product_model_summary.csv
```

Ensure the required files are present before running the dashboard.

---

## 12. Limitations and Future Improvements

Potential future improvements include:

- Automated ingestion of new retail transactions.
- Scheduled model retraining.
- Forecast accuracy monitoring as new data arrives.
- More robust product-specific model selection.
- Inventory replenishment recommendations incorporating lead times and safety stock.
- Customer retention campaign measurement.
- Cloud deployment and automated updates.
- Automated data validation pipelines.
- Unit and integration testing.
- Experiment tracking and model versioning.
- Containerisation and production deployment.

The current forecasting experiments use historical data. Further evaluation on genuinely future, unseen observations is needed before using the forecasts for operational inventory decisions.

---

## 13. Acknowledgements

This project uses historical online retail transaction data for educational and analytical purposes.

---

## 14. Disclaimer

RetailPulse is an analytics and forecasting project. Its outputs are intended to support business analysis and decision-making.

Forecasts and customer segments should be validated before being used for real-world business operations.
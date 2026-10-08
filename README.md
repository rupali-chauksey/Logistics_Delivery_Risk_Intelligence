# Logistics Delivery Risk Intelligence
## Delivery Delay Prediction & Operational Intervention for a Multi-Modal Logistics Network

- **Author / Candidate**: Rupali Chouksey
- **Assignment**: Logistics Delivery Risk Intelligence (End-to-End ML Pipeline)
- **Domain**: Logistics & Supply Chain Operations (RouteWise Logistics)
- **GitHub Repository**: [https://github.com/rupali-chauksey/Logistics_Delivery_Risk_Intelligence](https://github.com/rupali-chauksey/Logistics_Delivery_Risk_Intelligence)
- **Primary Notebook**: [`logistics_delivery_delay_analysis.ipynb`](./logistics_delivery_delay_analysis.ipynb)
- **Dataset**: [`logistics_delivery_delay.csv`](./logistics_delivery_delay.csv)
- **Core Technologies**: Python 3, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, SciPy

---

## 1. Problem Statement Overview

RouteWise Logistics operates a multi-modal freight network moving consignments across Road, Rail, and Air. In time-critical logistics, discovering that a shipment arrived late after delivery provides zero actionable value—as the Head of Operations emphasized:

> *"We already know a shipment was late once it's late — that's not intelligence, that's a report card. I need to know which shipments are at risk while they're still in the network, so my planners can re-route, expedite, or call the customer ahead of time."*

This project delivers an end-to-end Machine Learning intelligence pipeline addressing three core operational pillars:
1. **Forecasting (Regression)**: Quantifying the exact delay magnitude (`delivery_delay_hours`) for early ETA adjustments.
2. **Decisioning (Classification)**: Flagging shipments as delay risks (`delay_required = 1`) to trigger immediate proactive intervention.
3. **Understanding (Unsupervised Clustering & PCA)**: Mapping hidden operational bottlenecks and operating patterns across the freight network.

---

## 2. Solution Approach & Pipeline Architecture

The solution implements a rigorous, leakage-free operational ML pipeline:
- **Leakage Prevention**: Strictly isolates post-outcome metrics (`actual_delivery_hours`) and uses an 80/20 chronological time-series split.
- **Preprocessing Pipeline**: Missing value imputation and encoding encapsulated inside Scikit-Learn `ColumnTransformer` fitted strictly on training data.
- **Feature Engineering**: Derives operational metrics (`total_pre_delivery_process_hours`, `process_time_ratio`) from pre-departure data.
- **Predictive Modeling**: Combines Linear Regression and Random Forest Regressors with high-recall Decision Tree and Random Forest Classifiers.
- **Unsupervised Insights**: PCA 2D compression and K-Means clustering (validated via Hierarchical Dendrogram).
- **Operational Decisioning**: Translates model outputs into a 3-Tier priority intervention matrix for logistics floor planners.

---

## 3. Dataset Architecture & Exploratory Data Analysis

The pipeline analyzes `logistics_delivery_delay.csv`, containing 720 shipment records sampled every 3 hours between **1 April 2026 and 30 June 2026**.

| Property | Description |
|---|---|
| **Total Records** | 720 multi-modal shipments |
| **Feature Dimensions** | 17 columns (14 predictors, 1 post-outcome field, 2 target variables) |
| **Observation Window** | 1 April 2026 – 30 June 2026 (sampled uniformly every 3 hours) |
| **Origin Hubs** | North, South, East, West (180 shipments each) |
| **Destination Regions** | Metro, Industrial, Rural, Tier-2 City (180 shipments each) |
| **Shipping Modes** | Road (240), Rail (240), Air (240) |
| **Carriers** | Carrier-A, Carrier-B, Carrier-C, Carrier-D (180 shipments each) |
| **Regression Target** | `delivery_delay_hours` (Delay realized beyond expected SLA) |
| **Classification Target** | `delay_required` (Binary indicator: 1 = delay risk, 0 = on-time) |
| **Post-Outcome Variable** | `actual_delivery_hours` (Known strictly after final delivery) |
| **Target Imbalance** | ~69.31% positive delay risk across the historical log |

### 📊 Dataset Distributions & Operational Delay Patterns
![EDA Dataset Distributions](./assets/01_eda_dataset_distributions.png)

---

## 4. Preprocessing & Leakage Prevention Strategy

To ensure zero lookahead bias and maintain strict pipeline integrity, the system enforces chronological separation and isolated transformations:

1. **Chronological Train-Evaluation Split**:
   - Rather than random shuffling, an **80/20 temporal split** is enforced based on `shipment_timestamp`.
   - **Training Partition**: Earlier 576 shipments (1 April 2026 to 12 June 2026).
   - **Evaluation Partition**: Later 144 shipments (12 June 2026 to 30 June 2026).
2. **Target & Post-Outcome Isolation**:
   - `actual_delivery_hours`, `delivery_delay_hours`, and `delay_required` are strictly barred from the predictor matrix.
3. **Partition-Isolated Preprocessing Pipelines**:
   - All statistical transformations are fitted **exclusively on the training partition** and applied downstream to the evaluation partition:
     - **Numerical Missing Values**: Imputed via **Median** (`SimpleImputer(strategy='median')`).
     - **Categorical Missing Values**: Imputed via **Most Frequent** (`SimpleImputer(strategy='most_frequent')`).
     - **Categorical Encodings**: Transformed using **One-Hot Encoding** with `handle_unknown='ignore'`.
     - **Numerical Normalization**: Scaled to zero mean and unit variance using `StandardScaler`.

---

## 5. Operational Feature Engineering

Two key operational features were constructed using exclusively pre-transit information:

1. **`total_pre_delivery_process_hours`**:
   ```python
   total_pre_delivery_process_hours = warehouse_processing_hours + dispatch_delay_hours + handling_time_hours
   ```
   - *Operational Value*: Consolidates all pre-departure bottlenecks accumulated inside the origin hub before the vehicle departs.

2. **`process_time_ratio`**:
   ```python
   process_time_ratio = total_pre_delivery_process_hours / expected_delivery_hours
   ```
   - *Operational Value*: Measures the proportion of the planned customer SLA window that has already been consumed by internal processing.

---

## 6. Machine Learning Models & Results

### 6.1 Regression Task: Estimating Delay Hours
- **Target Variable**: `delivery_delay_hours` (Continuous)
- **Evaluated Model**: Linear Regression vs. Random Forest Regressor Pipeline
- **Evaluation Period Performance**:
  - **MAE**: **`1.3361 hours`**
  - **RMSE**: **`1.7280 hours`**

### 📈 Regression: Actual vs. Predicted Delivery Delay
![Linear Regression Actual vs Predicted](./assets/02_regression_actual_vs_predicted.png)

- **Operational Interpretation**:
  - Predicts standard operational delays (2 to 6 hours) with high fidelity.
  - Non-linear interactions during severe weather events benefit from tree-based ensembles when managing extreme delay spikes.

---

### 6.2 Classification Task: Proactive Delay Risk Flagging
- **Target Variable**: `delay_required` (Binary: 1 = Delay Risk, 0 = On-Time)
- **Model Comparison**: Decision Tree vs. Random Forest Classifier

| Evaluation Metric | Decision Tree | Random Forest Classifier (Production Model) |
|---|:---:|:---:|
| **Accuracy** | 75.00% | **77.78%** |
| **Precision** | 75.76% | **78.91%** |
| **Recall** | **92.93%** | **91.82%** |
| **F1-Score** | 0.8351 | **0.8487** |

### 📊 Classifier Performance & Operational Metric Tradeoffs
![Classification Performance](./assets/03_classification_metrics_comparison.png)

#### Why Recall is the Critical Operational Metric in Logistics:
- **False Negative (Missed Delay)**: The model predicts on-time, but the parcel arrives late. Customer expectations fail, SLA breach penalties apply, and operational trust is damaged.
- **False Positive (False Alarm)**: The model flags an on-time shipment as risky. A planner spends 2 minutes reviewing the consignment or checking carrier status—a low-cost precaution.
- **Strategic Decision**: The Random Forest model captures **91.82% of all at-risk shipments** while maintaining **78.91% Precision**, delivering the highest operational safety margin.

---

## 7. Feature Importance & PCA Dimensionality Reduction

### 🔍 Top Predictive Drivers & 2D PCA Representation
![Feature Importance and PCA](./assets/04_feature_importance_and_pca.png)

- **Primary Predictive Drivers**:
  1. `total_pre_delivery_process_hours` (Hub bottleneck time)
  2. `warehouse_processing_hours` (Sorting and staging speed)
  3. `distance_km` (Route transit mileage)
  4. `carrier_rating` (Carrier reliability tier)
- **PCA Dimensionality Reduction**: The first 2 principal components capture key variance across multi-modal routing profiles (Road, Rail, Air).

---

## 8. Unsupervised Clustering & Operating Corridors

Unsupervised K-Means clustering was executed on standardized pre-outcome features to detect structural patterns across freight movements:

- **Optimal Cluster Determination**: Silhouette analysis confirmed **K = 3** natural clusters.
- **Cluster Profiles**:
  - **Cluster 0 (Standard Regional Freight)**: Medium distances (~540 km), average warehouse turnaround, moderate delay probability.
  - **Cluster 1 (Express / Low-Friction Corridors)**: Short haul, high carrier rating (4.2/5), rapid hub dispatch. Lowest delay rate (~58.1%).
  - **Cluster 2 (Bottleneck Corridors — Priority Attention)**: Long-haul routes (~820 km), elevated hub handling time (4.6h), and highest delay rate (**75.3%**, average delay **5.6 hours**).

### 🎯 K-Means Clusters & Hierarchical Dendrogram
![K-Means and Hierarchical Dendrogram](./assets/05_kmeans_and_dendrogram.png)

---

## 9. Advanced Strategic Extensions

### 1. Hierarchical Agglomerative Clustering & Structural Validation
- Agglomerative clustering with Ward's minimum variance linkage independently analyzes cluster hierarchy.
- The resulting dendrogram splits cleanly into **3 major branches** at distance threshold ~15, confirming the robustness of the K-Means cluster structure.

### 2. Three-Tier Operational Intervention Priority Framework
By coupling continuous delay magnitude predictions with classification risk probabilities, operations planners can trigger prioritized floor actions:

### 🚨 Operational Priority Intervention Breakdown
![Intervention Priority Framework](./assets/06_intervention_priority_breakdown.png)

| Priority Tier | Trigger Criteria | Planner Action on Operations Floor |
|---|---|---|
| **Tier 1: High Priority** | Risk Prob >= 70% AND Predicted Delay >= 5.0h | **Immediate Action**: Expedite dock clearance, reassign to premium carrier, alert customer SLA desk proactively. |
| **Tier 2: Medium Priority** | Risk Prob >= 50% OR Predicted Delay >= 3.0h | **Active Monitoring**: Track checkpoint milestones; prioritize unloading queue at intermediate hubs. |
| **Tier 3: Low Priority** | Risk Prob < 50% AND Predicted Delay < 3.0h | **Standard Dispatch**: Automated monitoring with standard milestone logging. |

---

## 10. Operational Recommendations for RouteWise Planners

1. **Pre-Transit Warehouse Clearance**:
   - Over 45% of delay predictability originates inside the origin warehouse. Shipments queued >= 3.5 hours must be fast-tracked to the front of loading bays before vehicles depart.
2. **Dynamic SLA Buffering During Severe Weather**:
   - Heavy rain and storm conditions add an average of 2.2 hours to road transit. Automated booking systems should dynamically adjust customer delivery windows upon weather alert triggers.
3. **Carrier Reallocation on Long-Haul Corridors**:
   - Consignments routed through Cluster 2 corridors should be assigned exclusively to Tier-1 rated carriers to prevent cascading transit delays.

---

## 11. Dependencies & Setup Instructions

Install the necessary Python libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
```

---

## 12. Execution Steps

1. **Clone or Download the Repository**:
   ```bash
   git clone https://github.com/rupali-chauksey/Logistics_Delivery_Risk_Intelligence.git
   cd Logistics_Delivery_Risk_Intelligence
   ```
2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook logistics_delivery_delay_analysis.ipynb
   ```
3. **Run the Analysis**:
   - In Jupyter, select the active Python kernel and click **Run All Cells**.
   - The entire notebook executes sequentially top-to-bottom without warnings or errors.

---

## 13. References & Documentation Consulted

1. **Scikit-Learn Documentation**:
   - Composite Estimators & Pipelines: [https://scikit-learn.org/stable/modules/compose.html](https://scikit-learn.org/stable/modules/compose.html)
   - Preprocessing & Encoders (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`): [https://scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html)
   - Linear Regression & Ensemble Classifiers (`DecisionTreeClassifier`, `RandomForestClassifier`): [https://scikit-learn.org/stable/modules/ensemble.html](https://scikit-learn.org/stable/modules/ensemble.html)
   - Clustering & Dimensionality Reduction (`KMeans`, `PCA`): [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)
2. **SciPy Documentation**:
   - Hierarchical Clustering & Dendrograms (`scipy.cluster.hierarchy`): [https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
3. **Pandas & NumPy Reference**:
   - Time Series and Structured Data Analysis: [https://pandas.pydata.org/docs/user_guide/timeseries.html](https://pandas.pydata.org/docs/user_guide/timeseries.html)

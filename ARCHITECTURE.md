# 🏛️ OmniInsight AI System Architecture & Technical Specifications

## 1. System Philosophy

Small and Medium Businesses (SMBs) operate under constraints very different from large enterprises:
1. **Low Tolerance for Ambiguity**: Business owners need clear answers ("What will our revenue be next month?", "Which customers might cancel?", "Can we raise prices by 5%?"), not raw probability curves.
2. **Messy, Fragmented Input**: Data rarely adheres to perfect schemas; column names vary from `Sales_USD` to `Total Amount` to `gross_rev`.
3. **Execution Gap**: Knowing an insight is useless without a prioritized operational playbook allocating departmental responsibility.

To address these principles, OmniInsight decouples data analysis into a **Cognitive Agent Pipeline** backed by deterministic, mathematically validated statistical models.

---

## 2. Cognitive Agent Pipeline

```
  [Raw Input: CSV/Excel]
           │
           ▼
  [1. ProfilerAgent]
           │ ➔ Evaluates Completeness, Uniqueness, Consistency (0-100 Score)
           │ ➔ Infers Column Semantics (Datetime, Target, CustomerKey, Category, Quantity)
           │ ➔ Imputes Missing Values & Drops Zero-Variance Attributes
           ▼
  [2. HypothesisAgent]
           │ ➔ Evaluates Business Domain (Retail, SaaS, Restaurant, General)
           │ ➔ Generates 4-6 Empirical Business Questions
           ▼
  [3. SpecialistAgent]
           │ ➔ Selects Appropriate Models for Dataset Attributes
           │ ┌──────────────────────┬──────────────────────┐
           │ ▼                      ▼                      ▼
           │ Forecaster       Segmentation           Churn Radar
           │ Elasticity       Anomaly Radar
           ▼
  [4. ExecutiveAgent]
           │ ➔ Integrates Findings Across All Active Models
           │ ➔ Quantifies Financial Opportunity ($ ROI)
           │ ➔ Constructs 7-Day, 30-Day, 90-Day Tactical Playbook
           ▼
  [Standardized AuditReport & Interactive Views]
```

---

## 3. Mathematical Foundations of the Model Suite

### 3.1. Demand & Time Series Forecaster (`TimeSeriesForecaster`)
- **Aggregation**: Groups raw transaction timestamps to daily frequency $y_t$.
- **Feature Matrix**:
  $$X_t = [t, \text{is\_weekend}_t, \mathbf{DOW}_t]$$
  where $\mathbf{DOW}_t$ is the one-hot encoded vector of days of the week.
- **Model**: Ridge Regression with $L_2$ regularization:
  $$\hat{\beta} = \arg\min_\beta \left( \|y - X\beta\|_2^2 + \alpha \|\beta\|_2^2 \right)$$
- **Uncertainty Bounds**:
  $$\text{Upper}_t = \hat{y}_t + 1.96 \cdot \hat{\sigma}_\epsilon, \quad \text{Lower}_t = \max(0, \hat{y}_t - 1.96 \cdot \hat{\sigma}_\epsilon)$$
  where $\hat{\sigma}_\epsilon = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$.

### 3.2. Customer RFM & K-Means Clustering (`CustomerSegmentationEngine`)
- **Feature Vector**:
  $$\vec{x}_i = \left[ \ln(R_i + 1), \ln(F_i + 1), \ln(M_i + 1) \right]$$
  where $R_i$ is recency in days, $F_i$ is transaction frequency, and $M_i$ is cumulative monetary spend.
- **Scaling**: Standardized with zero mean and unit variance.
- **Cluster Selection**: Evaluates Silhouette score $S(k)$ across $k \in \{3, 4, 5\}$:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad k^* = \arg\max_k \bar{s}(k)$$

### 3.3. Churn Prediction & Risk Scoring (`ChurnPredictionEngine`)
- **Model**: Ensemble Random Forest with Gini impurity splitting and maximum tree depth $D=6$ to prevent overfitting on small datasets.
- **Feature Importance**: Evaluates Mean Decrease in Impurity (MDI):
  $$I(f) = \sum_{T} \sum_{t \in T: v(s_t) = f} p(t) \Delta i(s_t, t)$$
- **Risk Threshold**: Flags active accounts with attrition probability $P(\text{churn} \ge 0.65)$ and calculates exposed revenue:
  $$\text{Revenue at Risk} = \sum_{i \in \text{High Risk}} \text{Spend}_i$$

### 3.4. Econometric Price Elasticity of Demand (`PriceElasticityEngine`)
- **Equation**:
  $$\ln(Q_i) = \alpha + \epsilon \cdot \ln(P_i) + \varepsilon_i$$
- **Interpretation**:
  - $\epsilon > -0.8$: **Inelastic** (Low sensitivity; business has pricing power; recommended price hike: +4% to +8%).
  - $\epsilon < -1.4$: **Elastic** (High sensitivity; price hikes will compress total revenue; recommend value bundling).
  - $-1.4 \le \epsilon \le -0.8$: **Unitary**.

### 3.5. Operational Anomaly Radar (`AnomalyDetectionEngine`)
- **Algorithm**: Isolation Forest with tree isolation depth.
- Contamination parameter $\gamma = 0.03$ (identifying top 3% operational outliers).
- Generates Z-score diagnostic signatures explaining which individual operational metrics contributed to the anomaly flag.

# 🌐 OmniInsight AI REST API Reference

The OmniInsight API provides high-performance, asynchronous endpoints for data profiling, multi-agent audits, and conversational business queries.

- **Base URL:** `http://localhost:8000/api/v1`
- **Interactive OpenAPI Documentation:** `http://localhost:8000/docs`
- **Specification:** OpenAPI 3.1.0

---

## 1. System Health Check

### `GET /api/v1/health`
Verify server status and version information.

**Sample Request:**
```bash
curl -X GET http://localhost:8000/api/v1/health
```

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "app": "OmniInsight AI SMB Data Intelligence",
  "version": "1.0.0"
}
```

---

## 2. List Benchmark Samples

### `GET /api/v1/samples`
Returns available preloaded SMB benchmark datasets.

**Sample Request:**
```bash
curl -X GET http://localhost:8000/api/v1/samples
```

**Response (`200 OK`):**
```json
[
  {
    "id": "retail",
    "name": "Retail & E-Commerce Transactions",
    "business_type": "retail",
    "description": "3,500 multi-channel retail orders with products, categories, margins, discounts, and customer IDs.",
    "rows": 3500,
    "columns": 13
  },
  {
    "id": "saas",
    "name": "SaaS Customer Subscriptions & Retention",
    "business_type": "saas",
    "description": "1,200 B2B subscription accounts with MRR, seat usage, login frequency, support tickets, and churn labels.",
    "rows": 1200,
    "columns": 14
  },
  {
    "id": "restaurant",
    "name": "Hospitality & Restaurant Operations",
    "business_type": "restaurant",
    "description": "650 operational days with covers, average check, dine-in vs delivery sales, labor, food costs, and weather.",
    "rows": 650,
    "columns": 16
  }
]
```

---

## 3. Autonomous Multi-Agent Audit on Sample Dataset

### `POST /api/v1/audit/sample/{sample_id}`
Triggers the full multi-agent cognitive pipeline on a benchmark SMB dataset (`retail`, `saas`, or `restaurant`).

**Sample Request:**
```bash
curl -X POST http://localhost:8000/api/v1/audit/sample/retail
```

**Response (`200 OK`):**
Returns a complete `AuditReport` object containing:
- `audit_id`: Unique identifier (e.g., `AUDIT-D0AA6E29`)
- `dataset_profile`: Shape, memory usage, semantic types, 0-100 hygiene score
- `hypotheses`: Prioritized business questions and strategic rationales
- `model_results`: Quantitative findings, forecast tables, RFM segments, churn watchlist, elasticity coefficients, anomaly radar
- `executive_summary`: Plain-English narrative, quantified dollar ROI, 7/30/90-day prioritized action playbook

---

## 4. Multi-Agent Audit on File Upload

### `POST /api/v1/audit/file`
Upload a proprietary CSV or Excel (`.xlsx`, `.xls`) file to execute the autonomous multi-agent audit.

**Sample Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/audit/file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@my_sales_data.csv" \
  -F "business_type=retail"
```

---

## 5. Conversational Natural Language Query

### `POST /api/v1/query`
Query business data using natural language questions.

**Sample Request:**
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is our total revenue?",
    "sample_dataset": "retail"
  }'
```

**Response (`200 OK`):**
```json
{
  "question": "What is our total revenue?",
  "answer": "Total cumulative gross revenue across all records is **$856,364.28** (with an average transaction value of **$244.68**).",
  "metric_value": 856364.28,
  "table_data": null
}
```

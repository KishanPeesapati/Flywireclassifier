# Credit Card Intent Classification Service

A containerized microservice that classifies credit card-related customer queries into intent categories using machine learning. Built with FastAPI and sentence transformers for fast, accurate intent detection to help customer support teams triage queries efficiently.

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Code Components](#code-components)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

Get the service running in under 2 minutes:

```bash
# Build and run with Docker
docker build -t credit-card-classifier .
docker run -p 8000:8000 credit-card-classifier
```

**Expected Output:**
```
INFO: Loaded 10 intent categories
INFO: Loaded 1734 labeled queries
INFO: Created embeddings with shape: (1734, 384)
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Verify it's working:**
```bash
For Swagger UI
http://0.0.0.0:8000/docs

curl http://localhost:8000/health
```

---

## Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **Docker**: 20.0+ (for containerized deployment)
- **Memory**: Minimum 2GB RAM (4GB recommended)
- **Storage**: 500MB free space for model and dependencies

### Platform Support
- macOS (Intel & Apple Silicon)
- Linux (Ubuntu 18.04+, CentOS 7+)
- Windows 10/11 with WSL2

---



## Installation & Setup

### Option A: Docker (Recommended)

**Step 1: Clone and navigate to project**
```bash
# One command to get the code
git clone https://github.com/KishanPeesapati/Flywireclassifier.git
cd Flywireclassifier
```

**Step 2: Build the container**
```bash
docker build -t credit-card-classifier .
```

**Step 3: Run the service**
```bash
docker run -p 8000:8000 credit-card-classifier
```

The service will be available at `http://localhost:8000`

### Option B: Local Development

**Step 1: Create virtual environment**
```bash
python3 -m venv classifier_env
source classifier_env/bin/activate  # On Windows: classifier_env\Scripts\activate
```

**Step 2: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Verify data files**
Ensure these files exist in `data_files/`:
- `credit_card_intent_labels 1.xlsx`
- `credit_card_queries_labeled.xlsx`

**Step 4: Start the service**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Usage

### Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Service is running and model is loaded",
  "model_loaded": true
}
```

### Classify a Query
```bash
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"query": "My card was declined at the store"}'
```

**Response:**
```json
{
  "intent_label": 25,
  "intent_description": "declined_card_payment",
  "confidence": 0.944,
  "processing_time_ms": 45.2,
  "is_confident": true
}
```

### Interactive API Documentation
Open your browser to `http://localhost:8000/docs` for Swagger UI documentation.

---

## Testing

### Run Complete Test Suite
```bash
cd tests
python run_tests.py
```

**Expected Output:**
```
Happy Path: 4/4 passed [PASS]
Input Validation: 6/6 passed [PASS]
Confidence Threshold: 1/1 passed [PASS]
Error Handling: 2/2 passed [PASS]
Business Logic: 1/1 passed [PASS]

Overall: 14/14 tests passed
```

### Individual Test Categories
```bash
# Run only endpoint tests
python test_classify_endpoint.py
```

### Manual Testing Examples
```bash
# Test valid query
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"query": "I cannot activate my credit card"}'

# Test invalid query (should be rejected)
curl -X POST "http://localhost:8000/classify" \
     -H "Content-Type: application/json" \
     -d '{"query": "1222"}'
```

---

## API Reference

### POST /classify

Classify a credit card query into intent categories.

**Request Schema:**
```json
{
  "query": "string (3-500 characters, must contain letters)"
}
```

**Success Response (200):**
```json
{
  "intent_label": 0,
  "intent_description": "activate_my_card", 
  "confidence": 0.854,
  "processing_time_ms": 67.3,
  "is_confident": true
}
```

**Low Confidence Response (200):**
```json
{
  "message": "I'm not confident about this classification (confidence: 45%). Please rephrase...",
  "intent_label": 12,
  "intent_description": "card_delivery",
  "confidence": 0.45,
  "processing_time_ms": 71.2,
  "suggestions": ["Try rephrasing with more details", "Use specific credit card terms"]
}
```

**Error Responses:**
- **400**: Bad Request (malformed JSON)
- **422**: Validation Error (invalid query format)
- **500**: Internal Server Error

### GET /health

Service health check.

**Response (200):**
```json
{
  "status": "healthy",
  "message": "Service is running and model is loaded",
  "model_loaded": true
}
```

### GET /intents

List all available intent categories.

**Response (200):**
```json
{
  "0": "activate_my_card",
  "12": "card_delivery", 
  "15": "card_payment_fee_charged",
  "16": "card_payment_not_recognised",
  "17": "card_payment_wrong_exchange_rate",
  "18": "card_swallowed",
  "22": "compromised_card",
  "25": "declined_card_payment",
  "41": "lost_or_stolen_card",
  "45": "pending_card_payment"
}
```

---

## Code Components

### `classifier.py`
**Core ML Classification Engine**
- Loads training data from Excel files (1,734 labeled queries)
- Initializes sentence-transformers model (all-MiniLM-L6-v2)
- Creates 384-dimensional embeddings for all training queries
- Performs intent classification using cosine similarity
- Returns confidence scores and matched intents

**Key Functions:**
- `load_data()` - Loads Excel training data
- `initialize_model()` - Loads pre-trained sentence transformer
- `create_embeddings()` - Generates embeddings for training data
- `classify_query()` - Classifies new queries

### `main.py`
**FastAPI REST Service**
- Implements REST endpoints with automatic OpenAPI documentation
- Input validation using Pydantic models
- Confidence thresholding (60% default)
- Error handling with proper HTTP status codes
- Structured logging for monitoring

**Key Features:**
- Async application startup with model pre-loading
- Request/response models with validation
- Custom error messages for better UX

### `tests/`
**Automated Test Suite**
- `test_classify_endpoint.py` - Comprehensive endpoint testing
- `run_tests.py` - Test runner with summary reporting
- Validates happy path, error handling, and edge cases

### `data_files/`
**Training Data (Banking77 Dataset)**
- `credit_card_intent_labels 1.xlsx` - Intent definitions (10 categories)
- `credit_card_queries_labeled.xlsx` - Labeled examples (1,734 queries)

---

## Project Structure

```
Project-Flywire/
├── classifier.py              # Core ML classification logic
├── main.py                    # FastAPI service & endpoints
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container configuration
├── README.md                  # Project documentation
├── data_files/               # Training data
│   ├── credit_card_intent_labels.xlsx
│   └── credit_card_queries_labeled.xlsx
└── tests/                    # Test suite
    ├── __init__.py
    ├── test_classify_endpoint.py
    └── run_tests.py
```

---

## Performance

### Response Times
- **Average classification time**: ~50ms
- **Model loading time**: ~10-15 seconds (startup only)
- **Embedding creation**: ~10 seconds (startup only)

### Accuracy Metrics
- **Test query confidence**: 85-95% on validation set
- **Classification method**: Cosine similarity with threshold 0.6
- **Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)

### Resource Usage
- **Memory**: ~800MB (including model and embeddings)
- **CPU**: Moderate during inference, high during startup
- **Storage**: ~200MB (model + dependencies)

---

## Troubleshooting

### Common Issues

**Problem: Service won't start**
```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill process using port 8000
kill -9 $(lsof -t -i:8000)
```

**Problem: Docker build fails**
```bash
# Clear Docker cache and rebuild
docker system prune -f
docker build --no-cache -t credit-card-classifier .
```

**Problem: Model download fails**
```bash
# Check internet connection and disk space
df -h
ping huggingface.co
```

**Problem: Import errors in local setup**
```bash
# Verify virtual environment is activated
which python
pip list | grep fastapi

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

**Problem: Tests fail with connection errors**
```bash
# Ensure service is running first
curl http://localhost:8000/health

# Check if correct port is being used
docker ps
```

### Performance Issues

**Slow first request:**
- Expected behavior - model loads on first classification
- Subsequent requests will be much faster

**Memory usage high:**
- Normal - embeddings for 1,734 queries stored in memory
- Consider restart if memory grows beyond 1GB

### Error Messages

**HTTP 422 - Validation Error:**
- Query too short (< 3 characters)
- Query contains only numbers
- Query has no alphabetic characters

**HTTP 500 - Internal Server Error:**
- Model not loaded properly
- Data files missing or corrupted
- Check logs for detailed error information

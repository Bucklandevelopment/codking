

# CodKing Inference API Guide

Complete guide for deploying and using the CodKing inference API for cybersecurity data processing.

---

## Quick Start

### 1. Start the Server

```bash
# With default checkpoint (checkpoints/best.ckpt)
python src/codking/api/inference_server.py

# With specific checkpoint
python src/codking/api/inference_server.py \
  --checkpoint checkpoints/codking-epoch=40-val_accuracy=0.8891.ckpt \
  --port 8000 \
  --device cuda
```

Server starts on `http://localhost:8000`

### 2. Check Health

```bash
curl http://localhost:8000/health
```

### 3. Use Python Client

```python
from codking.api.client import CodKingClient

client = CodKingClient("http://localhost:8000")

# Detect threat
result = client.detect_threat("Suspicious SSH login attempt")
print(result.is_threat, result.threat_score)
```

---

## Server Deployment

### Local Deployment

```bash
# Development
python src/codking/api/inference_server.py \
  --checkpoint checkpoints/best.ckpt \
  --host 127.0.0.1 \
  --port 8000

# Production (bind to all interfaces)
python src/codking/api/inference_server.py \
  --checkpoint checkpoints/best.ckpt \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY src/codking /app/codking
COPY checkpoints/best.ckpt /app/checkpoints/best.ckpt

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "codking.api.inference_server", \
     "--checkpoint", "/app/checkpoints/best.ckpt", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

Build and run:
```bash
docker build -t codking-api .
docker run -p 8000:8000 --gpus all codking-api
```

### Production with Gunicorn

```bash
# Install gunicorn
pip install gunicorn[standard]

# Run with 4 workers
gunicorn codking.api.inference_server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

---

## API Endpoints

### 1. Health Check

**GET** `/health`

Returns server health and metrics.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "uptime_seconds": 3600.5,
  "total_requests": 1250,
  "avg_latency_ms": 6.8
}
```

### 2. Load Model

**POST** `/load_model`

Load model from checkpoint (useful for hot-swapping models).

```bash
curl -X POST http://localhost:8000/load_model \
  -H "Content-Type: application/json" \
  -d '{
    "checkpoint_path": "checkpoints/new_model.ckpt",
    "device": "cuda"
  }'
```

### 3. Threat Detection

**POST** `/detect_threat`

Detect threat in single log entry.

```bash
curl -X POST http://localhost:8000/detect_threat \
  -H "Content-Type: application/json" \
  -d '{
    "log_content": "CRITICAL: Multiple failed SSH attempts from 203.0.113.42",
    "source": "firewall",
    "timestamp": "2025-10-07T12:34:56Z"
  }'
```

Response:
```json
{
  "is_threat": true,
  "threat_score": 0.923,
  "confidence": 0.957,
  "prediction_class": "threat",
  "metadata": {
    "latency_ms": 6.2,
    "compression_ratio": 5.8,
    "num_chunks": 42,
    "num_cycles": 8,
    "batch_size": 1,
    "device": "cuda:0"
  }
}
```

### 4. Batch Threat Detection

**POST** `/detect_threats_batch`

Process multiple logs in batch (up to 100).

```bash
curl -X POST http://localhost:8000/detect_threats_batch \
  -H "Content-Type: application/json" \
  -d '[
    {"log_content": "Normal user login", "source": "auth"},
    {"log_content": "SQL injection attempt", "source": "web"}
  ]'
```

Response:
```json
{
  "results": [
    {
      "is_threat": false,
      "threat_score": 0.123,
      "confidence": 0.877,
      "prediction_class": "normal"
    },
    {
      "is_threat": true,
      "threat_score": 0.891,
      "confidence": 0.925,
      "prediction_class": "threat"
    }
  ],
  "total_processed": 2,
  "processing_time_ms": 8.5,
  "throughput_samples_per_sec": 235.3
}
```

### 5. Text Classification

**POST** `/classify`

General text classification.

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Sample text to classify",
    "return_scores": true,
    "return_metadata": true
  }'
```

Response:
```json
{
  "predicted_class": 3,
  "class_label": "class_3",
  "confidence": 0.847,
  "scores": [0.023, 0.089, 0.041, 0.847],
  "metadata": {
    "latency_ms": 5.9,
    "compression_ratio": 6.1,
    "num_chunks": 38,
    "num_cycles": 7
  }
}
```

### 6. Batch Classification

**POST** `/classify_batch`

Batch text classification (up to 100 texts).

```bash
curl -X POST http://localhost:8000/classify_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Text 1", "Text 2", "Text 3"],
    "return_scores": true,
    "return_metadata": true
  }'
```

### 7. File Upload

**POST** `/analyze_log_file`

Upload and analyze log file.

```bash
curl -X POST http://localhost:8000/analyze_log_file \
  -F "file=@/path/to/logfile.log"
```

Response:
```json
{
  "filename": "logfile.log",
  "is_threat": true,
  "threat_score": 0.856,
  "confidence": 0.912,
  "metadata": {
    "latency_ms": 12.3,
    "compression_ratio": 5.9
  }
}
```

### 8. Metrics

**GET** `/metrics`

Get inference metrics.

```bash
curl http://localhost:8000/metrics
```

---

## Python Client Usage

### Installation

```python
from codking.api.client import CodKingClient
```

### Basic Usage

```python
# Create client
client = CodKingClient("http://localhost:8000")

# Check health
health = client.health()
print(f"Status: {health['status']}")

# Load model (optional)
client.load_model("checkpoints/best.ckpt", device="cuda")
```

### Threat Detection

```python
# Single threat detection
result = client.detect_threat(
    log_content="Suspicious SSH login attempt from 192.168.1.100",
    source="auth_log",
    timestamp="2025-10-07T12:34:56Z"
)

print(f"Threat: {result.is_threat}")
print(f"Score: {result.threat_score:.3f}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Latency: {result.metadata['latency_ms']:.2f}ms")
```

### Batch Threat Detection

```python
# Batch detection
logs = [
    "Normal user login",
    "Multiple failed SSH attempts",
    "System backup completed",
    "SQL injection attempt detected"
]

batch_result = client.detect_threats_batch(logs)

print(f"Total processed: {batch_result.total_processed}")
print(f"Threats found: {len(batch_result.get_threats())}")
print(f"Throughput: {batch_result.throughput:.1f} samples/sec")

# Get only threats
for result in batch_result.get_threats():
    print(f"  - Threat score: {result.threat_score:.3f}")
```

### Text Classification

```python
# Single classification
result = client.classify("Text to classify")
print(f"Class: {result.predicted_class}")
print(f"Confidence: {result.confidence:.3f}")

# Batch classification
texts = ["Text 1", "Text 2", "Text 3"]
results = client.classify_batch(texts)

for i, result in enumerate(results):
    print(f"Text {i+1}: Class {result.predicted_class} (conf: {result.confidence:.3f})")
```

### File Analysis

```python
# Analyze log file
result = client.analyze_log_file("/path/to/logfile.log")
print(f"File: {result['filename']}")
print(f"Threat: {result['is_threat']}")
print(f"Score: {result['threat_score']:.3f}")
```

### Context Manager

```python
# Use with context manager
with CodKingClient("http://localhost:8000") as client:
    result = client.detect_threat("Some log entry")
    print(result.is_threat)
```

---

## Performance Optimization

### Batching

Always use batch endpoints for multiple inputs:

```python
# ❌ Slow: Individual requests
for log in logs:
    client.detect_threat(log)  # N network calls

# ✅ Fast: Batch request
client.detect_threats_batch(logs)  # 1 network call
```

**Performance gain:**
- 10× faster for 10 logs
- 100× faster for 100 logs

### Model Caching

The server caches the model in memory:
- First request: ~100ms (model loading)
- Subsequent requests: ~5-10ms (inference only)

### GPU Acceleration

```bash
# Use GPU for 5-10× speedup
python src/codking/api/inference_server.py \
  --checkpoint checkpoints/best.ckpt \
  --device cuda
```

### Concurrent Requests

The server handles concurrent requests automatically:

```python
import asyncio
import aiohttp

async def detect_threat_async(session, log):
    async with session.post(
        "http://localhost:8000/detect_threat",
        json={"log_content": log}
    ) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [detect_threat_async(session, log) for log in logs]
        results = await asyncio.gather(*tasks)
    return results

# Process 100 logs concurrently
results = asyncio.run(main())
```

---

## Integration Examples

### Hybrid OSINT Screening

```python
from codking.api.client import CodKingClient

# CodKing for screening (99% of data)
codking = CodKingClient("http://localhost:8000")

# SOTA API for deep analysis (1% of data)
from anthropic import Anthropic
claude = Anthropic(api_key="...")

def hybrid_osint_screening(events):
    # Step 1: Screen all events with CodKing
    batch_result = codking.detect_threats_batch(events)

    # Step 2: Deep analysis of suspicious events with Claude
    suspicious = []
    for i, result in enumerate(batch_result.results):
        if result.threat_score > 0.7:  # Threshold
            event = events[i]

            # Deep analysis with Claude
            response = claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": f"Analyze: {event}"}]
            )

            suspicious.append({
                'event': event,
                'codking_score': result.threat_score,
                'claude_analysis': response.content
            })

    return suspicious

# Process 1M events
# - CodKing: 1M events @ $0.001 per 1K = $1.00
# - Claude: 10K events @ $3 per 1M = $0.03
# Total: $1.03 (vs $3000 for all-Claude)
# Cost reduction: 99.97%
```

### Log Monitoring Pipeline

```python
import time
from codking.api.client import CodKingClient

client = CodKingClient("http://localhost:8000")

def monitor_logs(log_file_path, check_interval=60):
    """Monitor log file for threats."""

    print(f"Monitoring {log_file_path}")

    with open(log_file_path, 'r') as f:
        # Go to end of file
        f.seek(0, 2)

        while True:
            line = f.readline()

            if line:
                # New log entry
                result = client.detect_threat(line.strip())

                if result.is_threat:
                    print(f"🚨 THREAT DETECTED (score: {result.threat_score:.3f})")
                    print(f"   {line[:100]}")

                    # Trigger alert
                    send_alert(line, result.threat_score)
            else:
                # No new data, wait
                time.sleep(check_interval)

def send_alert(log_entry, threat_score):
    """Send alert to security team."""
    # Your alert logic here
    pass

# Start monitoring
monitor_logs("/var/log/auth.log")
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: codking-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: codking-api
  template:
    metadata:
      labels:
        app: codking-api
    spec:
      containers:
      - name: codking-api
        image: codking-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: CHECKPOINT_PATH
          value: "/models/best.ckpt"
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
---
apiVersion: v1
kind: Service
metadata:
  name: codking-api-service
spec:
  selector:
    app: codking-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Monitoring & Observability

### Metrics Endpoint

```python
import time

client = CodKingClient("http://localhost:8000")

while True:
    metrics = client.get_metrics()

    print(f"Requests: {metrics['total_requests']}")
    print(f"Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
    print(f"Uptime: {metrics['uptime_seconds'] / 3600:.1f}h")

    time.sleep(60)
```

### Prometheus Integration

Add to your FastAPI app:

```python
from prometheus_fastapi_instrumentator import Instrumentator

# In inference_server.py
Instrumentator().instrument(app).expose(app)
```

Then scrape metrics at `/metrics` with Prometheus.

---

## Troubleshooting

### Server Won't Start

**Check port availability:**
```bash
lsof -i :8000
```

**Use different port:**
```bash
python src/codking/api/inference_server.py --port 8001
```

### OOM on GPU

**Reduce batch size or use CPU:**
```bash
python src/codking/api/inference_server.py --device cpu
```

### Slow Inference

**Check device:**
```python
health = client.health()
print(health['device'])  # Should be 'cuda:0' for GPU
```

**Enable GPU:**
```bash
python src/codking/api/inference_server.py --device cuda
```

### Model Not Found

**Check checkpoint path:**
```bash
ls -lh checkpoints/best.ckpt
```

**Load model explicitly:**
```python
client.load_model("path/to/checkpoint.ckpt")
```

---

## API Reference

Full API documentation available at `http://localhost:8000/docs` (Swagger UI) after starting the server.

---

## Security Considerations

### Authentication

Add API key authentication:

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Add to endpoints
@app.post("/detect_threat", dependencies=[Depends(get_api_key)])
async def detect_threat(...):
    ...
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/detect_threat")
@limiter.limit("100/minute")
async def detect_threat(...):
    ...
```

### HTTPS

Use nginx or caddy as reverse proxy with SSL:

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

**Complete API deployment ready! 🚀**

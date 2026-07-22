"""
FastAPI Inference Server for CodKing.

Provides REST API endpoints for:
- Threat detection
- Log anomaly detection
- OSINT screening
- General text classification
- Batch processing

Features:
- Async request handling
- Model caching
- Automatic batching
- Health checks
- Metrics tracking
"""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import torch
import numpy as np
from pathlib import Path
import asyncio
import time
from datetime import datetime
import logging
import uvicorn

from ..training.lightning_module import CodKingLightningModule
from ..models.integration.pipeline import CodKingPipeline, CybersecurityPipeline


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response Models
class TextInput(BaseModel):
    """Single text input for inference."""
    text: str = Field(..., description="Input text to process", max_length=100000)
    return_scores: bool = Field(default=True, description="Return confidence scores")
    return_metadata: bool = Field(default=True, description="Return processing metadata")

    @validator('text')
    def validate_text(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Text cannot be empty")
        return v


class BatchTextInput(BaseModel):
    """Batch text inputs for inference."""
    texts: List[str] = Field(..., description="List of texts to process", max_items=100)
    return_scores: bool = Field(default=True, description="Return confidence scores")
    return_metadata: bool = Field(default=True, description="Return processing metadata")

    @validator('texts')
    def validate_texts(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Texts list cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum 100 texts per batch")
        return v


class LogInput(BaseModel):
    """Log data input for threat detection."""
    log_content: str = Field(..., description="Log content as text")
    source: Optional[str] = Field(None, description="Log source (e.g., 'firewall', 'web_server')")
    timestamp: Optional[str] = Field(None, description="Log timestamp")


class ThreatDetectionResponse(BaseModel):
    """Response for threat detection."""
    is_threat: bool = Field(..., description="Whether a threat was detected")
    threat_score: float = Field(..., description="Threat probability (0-1)")
    confidence: float = Field(..., description="Model confidence")
    prediction_class: str = Field(..., description="Predicted class label")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")


class BatchThreatDetectionResponse(BaseModel):
    """Response for batch threat detection."""
    results: List[ThreatDetectionResponse]
    total_processed: int
    processing_time_ms: float
    throughput_samples_per_sec: float


class ClassificationResponse(BaseModel):
    """Response for general classification."""
    predicted_class: int
    class_label: Optional[str] = None
    confidence: float
    scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float
    total_requests: int
    avg_latency_ms: float


# Model Manager
class ModelManager:
    """
    Manages model loading, caching, and inference.
    Singleton pattern for efficient resource usage.
    """
    _instance = None
    _model = None
    _device = None
    _start_time = None
    _request_count = 0
    _total_latency = 0.0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        use_cybersecurity: bool = True
    ):
        """Load model from checkpoint."""
        if self._model is not None:
            logger.info("Model already loaded")
            return

        logger.info(f"Loading model from {checkpoint_path}")

        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = torch.device(device)

        # Load checkpoint
        try:
            self._model = CodKingLightningModule.load_from_checkpoint(
                checkpoint_path,
                map_location=self._device
            )
            self._model.eval()
            self._model.freeze()

            logger.info(f"Model loaded successfully on {device}")
            logger.info(f"Parameters: {self._model.num_params:,}")

            self._start_time = time.time()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    @torch.no_grad()
    def predict(
        self,
        texts: Union[str, List[str]],
        return_scores: bool = True,
        return_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on text(s).

        Args:
            texts: Single text or list of texts
            return_scores: Return class scores
            return_metadata: Return processing metadata

        Returns:
            Dictionary with predictions and metadata
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # Convert to list if single text
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        batch_size = len(texts)

        # Convert texts to bytes
        input_ids_list = []
        for text in texts:
            byte_seq = text.encode('utf-8')
            byte_array = np.frombuffer(byte_seq, dtype=np.uint8)

            # Pad or truncate to max_length
            max_length = 8192
            if len(byte_array) > max_length:
                byte_array = byte_array[:max_length]
            else:
                padding = np.zeros(max_length - len(byte_array), dtype=np.uint8)
                byte_array = np.concatenate([byte_array, padding])

            input_ids_list.append(torch.from_numpy(byte_array).long())

        # Stack into batch
        input_ids = torch.stack(input_ids_list).to(self._device)

        # Forward pass
        output = self._model(
            input_ids=input_ids,
            use_act_inference=True,
            return_intermediate=False
        )

        # Extract predictions
        logits = output['output']  # [batch, num_classes]
        probs = torch.softmax(logits, dim=-1)

        if self._model.hparams.num_classes == 2:
            # Binary classification (threat detection)
            threat_scores = probs[:, 1].cpu().numpy()
            predictions = (threat_scores > 0.5).astype(int)
        else:
            # Multi-class classification
            predictions = logits.argmax(dim=-1).cpu().numpy()
            threat_scores = None

        # Processing time
        latency_ms = (time.time() - start_time) * 1000

        # Update metrics
        self._request_count += 1
        self._total_latency += latency_ms

        # Prepare results
        results = {
            'predictions': predictions.tolist() if not single_input else int(predictions[0]),
            'scores': probs.cpu().numpy().tolist() if return_scores else None,
            'threat_scores': threat_scores.tolist() if threat_scores is not None else None
        }

        if return_metadata:
            results['metadata'] = {
                'latency_ms': latency_ms,
                'compression_ratio': output['compression_ratio'],
                'num_chunks': output['num_chunks'],
                'num_cycles': output['num_cycles'],
                'batch_size': batch_size,
                'device': str(self._device)
            }

        return results

    def get_health(self) -> Dict[str, Any]:
        """Get server health metrics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        avg_latency = self._total_latency / self._request_count if self._request_count > 0 else 0

        return {
            'status': 'healthy' if self._model is not None else 'unhealthy',
            'model_loaded': self._model is not None,
            'device': str(self._device) if self._device else 'unknown',
            'uptime_seconds': uptime,
            'total_requests': self._request_count,
            'avg_latency_ms': avg_latency
        }


# Initialize model manager
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    # -- Startup --------------------------------------------------------
    # Load default model checkpoint if available
    default_checkpoint = Path("checkpoints/best.ckpt")
    if default_checkpoint.exists():
        try:
            model_manager.load_model(str(default_checkpoint))
            logger.info("Default model loaded on startup")
        except Exception as e:
            logger.warning(f"Could not load default model: {e}")
    else:
        logger.info("No default checkpoint found. Use /load_model endpoint to load a model.")

    # vital-core integration (graceful degradation)
    vital_client = None
    vital_enabled = os.getenv("VITAL_ENABLED", "true").lower() in ("1", "true", "yes")
    if vital_enabled:
        try:
            from ..integrations.vital_sdk import VitalClient, VitalConfig
            vital_config = VitalConfig()
            vital_client = VitalClient(config=vital_config)
            await vital_client.connect()
            await vital_client.register()
            logger.info("vital-core SDK connected")
        except Exception as exc:
            logger.warning("vital-core unavailable: %s", exc)
            vital_client = None

    app.state.vital_client = vital_client

    yield

    # -- Shutdown -------------------------------------------------------
    if vital_client is not None:
        try:
            await vital_client.disconnect()
        except Exception:
            pass


# Initialize FastAPI app
app = FastAPI(
    title="CodKing Inference API",
    description="High-efficiency inference API for threat detection and text classification",
    version="1.0.0",
    lifespan=lifespan,
)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    health = model_manager.get_health()
    return HealthResponse(**health)


# Load model endpoint
@app.post("/load_model")
async def load_model(
    checkpoint_path: str,
    device: Optional[str] = None,
    use_cybersecurity: bool = True
):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        use_cybersecurity: Use cybersecurity-specific model
    """
    try:
        model_manager.load_model(checkpoint_path, device, use_cybersecurity)
        return JSONResponse({
            "status": "success",
            "message": f"Model loaded from {checkpoint_path}",
            "device": str(model_manager._device)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


# Threat detection endpoint
@app.post("/detect_threat", response_model=ThreatDetectionResponse)
async def detect_threat(input: LogInput, request: Request):
    """
    Detect threats in log data.

    Args:
        input: Log data input

    Returns:
        Threat detection result
    """
    try:
        result = model_manager.predict(
            texts=input.log_content,
            return_scores=True,
            return_metadata=True
        )

        prediction = result['predictions']
        threat_score = result['threat_scores'][0] if result['threat_scores'] else result['scores'][0][1]

        response = ThreatDetectionResponse(
            is_threat=bool(prediction),
            threat_score=float(threat_score),
            confidence=float(max(result['scores'][0])),
            prediction_class="threat" if prediction else "normal",
            metadata=result['metadata']
        )

        # Publish event to vital-core
        vital_client = getattr(request.app.state, "vital_client", None)
        if vital_client is not None:
            try:
                await vital_client.publish_event(
                    category="security",
                    action="analyze",
                    event_type="threat.detected",
                    payload={
                        "is_threat": response.is_threat,
                        "threat_score": response.threat_score,
                        "confidence": response.confidence,
                        "prediction_class": response.prediction_class,
                        "source": input.source,
                    },
                )
            except Exception:
                pass  # Never let event publishing crash the endpoint

        return response

    except Exception as e:
        logger.error(f"Threat detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# Batch threat detection endpoint
@app.post("/detect_threats_batch", response_model=BatchThreatDetectionResponse)
async def detect_threats_batch(inputs: List[LogInput], request: Request):
    """
    Batch threat detection.

    Args:
        inputs: List of log inputs (max 100)

    Returns:
        Batch detection results
    """
    if len(inputs) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 logs per batch")

    try:
        start_time = time.time()

        texts = [log.log_content for log in inputs]
        result = model_manager.predict(
            texts=texts,
            return_scores=True,
            return_metadata=True
        )

        predictions = result['predictions']
        threat_scores = result['threat_scores'] if result['threat_scores'] else [scores[1] for scores in result['scores']]
        confidences = [max(scores) for scores in result['scores']]

        responses = []
        for i, pred in enumerate(predictions):
            responses.append(ThreatDetectionResponse(
                is_threat=bool(pred),
                threat_score=float(threat_scores[i]),
                confidence=float(confidences[i]),
                prediction_class="threat" if pred else "normal",
                metadata=None
            ))

        processing_time = (time.time() - start_time) * 1000
        throughput = len(inputs) / (processing_time / 1000)

        batch_response = BatchThreatDetectionResponse(
            results=responses,
            total_processed=len(inputs),
            processing_time_ms=processing_time,
            throughput_samples_per_sec=throughput
        )

        # Publish event to vital-core
        vital_client = getattr(request.app.state, "vital_client", None)
        if vital_client is not None:
            try:
                threats_found = sum(1 for r in responses if r.is_threat)
                await vital_client.publish_event(
                    category="security",
                    action="batch_analyze",
                    event_type="threat.batch_analyzed",
                    payload={
                        "total_processed": len(inputs),
                        "threats_found": threats_found,
                        "processing_time_ms": processing_time,
                        "throughput_samples_per_sec": throughput,
                    },
                )
            except Exception:
                pass  # Never let event publishing crash the endpoint

        return batch_response

    except Exception as e:
        logger.error(f"Batch threat detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


# Text classification endpoint
@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(input: TextInput, request: Request):
    """
    Classify text.

    Args:
        input: Text input

    Returns:
        Classification result
    """
    try:
        result = model_manager.predict(
            texts=input.text,
            return_scores=input.return_scores,
            return_metadata=input.return_metadata
        )

        prediction = result['predictions']
        scores = result['scores'][0] if result['scores'] else None
        confidence = float(max(scores)) if scores else None

        response = ClassificationResponse(
            predicted_class=int(prediction),
            class_label=f"class_{prediction}",
            confidence=confidence,
            scores=scores,
            metadata=result.get('metadata')
        )

        # Publish event to vital-core
        vital_client = getattr(request.app.state, "vital_client", None)
        if vital_client is not None:
            try:
                await vital_client.publish_event(
                    category="ai",
                    action="classify",
                    event_type="classification.completed",
                    payload={
                        "predicted_class": response.predicted_class,
                        "class_label": response.class_label,
                        "confidence": response.confidence,
                    },
                )
            except Exception:
                pass  # Never let event publishing crash the endpoint

        return response

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# Batch classification endpoint
@app.post("/classify_batch")
async def classify_batch(input: BatchTextInput, request: Request):
    """
    Batch text classification.

    Args:
        input: Batch text input

    Returns:
        Batch classification results
    """
    try:
        result = model_manager.predict(
            texts=input.texts,
            return_scores=input.return_scores,
            return_metadata=input.return_metadata
        )

        predictions = result['predictions']
        scores = result['scores'] if result['scores'] else None

        responses = []
        for i, pred in enumerate(predictions):
            pred_scores = scores[i] if scores else None
            confidence = float(max(pred_scores)) if pred_scores else None

            responses.append({
                'predicted_class': int(pred),
                'class_label': f"class_{pred}",
                'confidence': confidence,
                'scores': pred_scores
            })

        json_response = {
            'results': responses,
            'total_processed': len(input.texts),
            'metadata': result.get('metadata')
        }

        # Publish event to vital-core
        vital_client = getattr(request.app.state, "vital_client", None)
        if vital_client is not None:
            try:
                await vital_client.publish_event(
                    category="ai",
                    action="batch_classify",
                    event_type="classification.completed",
                    payload={
                        "total_processed": len(input.texts),
                    },
                )
            except Exception:
                pass  # Never let event publishing crash the endpoint

        return JSONResponse(json_response)

    except Exception as e:
        logger.error(f"Batch classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


# File upload endpoint
@app.post("/analyze_log_file")
async def analyze_log_file(request: Request, file: UploadFile = File(...)):
    """
    Analyze uploaded log file.

    Args:
        file: Log file upload

    Returns:
        Analysis results
    """
    try:
        # Read file content
        content = await file.read()
        text = content.decode('utf-8', errors='ignore')

        # Limit size
        if len(text) > 100000:
            raise HTTPException(status_code=400, detail="File too large (max 100KB)")

        result = model_manager.predict(
            texts=text,
            return_scores=True,
            return_metadata=True
        )

        prediction = result['predictions']
        threat_score = result['threat_scores'][0] if result['threat_scores'] else result['scores'][0][1]

        json_response = {
            'filename': file.filename,
            'is_threat': bool(prediction),
            'threat_score': float(threat_score),
            'confidence': float(max(result['scores'][0])),
            'metadata': result['metadata']
        }

        # Publish event to vital-core
        vital_client = getattr(request.app.state, "vital_client", None)
        if vital_client is not None:
            try:
                await vital_client.publish_event(
                    category="security",
                    action="analyze_file",
                    event_type="log_file.analyzed",
                    payload={
                        "filename": file.filename,
                        "is_threat": bool(prediction),
                        "threat_score": float(threat_score),
                        "confidence": float(max(result['scores'][0])),
                        "file_size_bytes": len(content),
                    },
                )
            except Exception:
                pass  # Never let event publishing crash the endpoint

        return JSONResponse(json_response)

    except Exception as e:
        logger.error(f"File analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"File analysis failed: {str(e)}")


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get inference metrics."""
    health = model_manager.get_health()
    return JSONResponse({
        'total_requests': health['total_requests'],
        'avg_latency_ms': health['avg_latency_ms'],
        'uptime_seconds': health['uptime_seconds'],
        'device': health['device'],
        'model_loaded': health['model_loaded']
    })


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return JSONResponse({
        'name': 'CodKing Inference API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/health',
            'load_model': '/load_model',
            'detect_threat': '/detect_threat',
            'detect_threats_batch': '/detect_threats_batch',
            'classify': '/classify',
            'classify_batch': '/classify_batch',
            'analyze_log_file': '/analyze_log_file',
            'metrics': '/metrics',
            'docs': '/docs'
        }
    })


# Run server
def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None
):
    """
    Run inference server.

    Args:
        host: Host to bind to
        port: Port to bind to
        checkpoint_path: Path to model checkpoint
        device: Device to use ('cuda', 'cpu', or None for auto)
    """
    # Load model if checkpoint provided
    if checkpoint_path:
        logger.info(f"Loading model from {checkpoint_path}")
        model_manager.load_model(checkpoint_path, device)

    # Run server
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run CodKing inference server')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu/None for auto)')

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

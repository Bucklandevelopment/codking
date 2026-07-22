"""
Unit tests for API components.

Tests:
- inference_server.py: FastAPI endpoints, model loading, inference
- client.py: Python client library
- model_manager.py: Model loading and inference management
"""

import pytest
import torch
from fastapi.testclient import TestClient
from pathlib import Path
import json

from codking.api.inference_server import app, model_manager
from codking.api.client import CodKingClient
from codking.api.model_manager import ModelManager


class TestModelManager:
    """Test model manager."""

    def test_model_manager_initialization(self, hnet_config_small, hrm_config_small, temp_checkpoint_dir):
        """Test model manager initialization."""
        # Create a dummy checkpoint
        from codking.models.integration.pipeline import CodKingPipeline

        model = CodKingPipeline(
            task_type='classification',
            num_classes=2,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        )

        checkpoint_path = temp_checkpoint_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'task_type': 'classification',
                'num_classes': 2,
                'hnet_config': hnet_config_small,
                'hrm_config': hrm_config_small
            }
        }, checkpoint_path)

        # Initialize model manager
        manager = ModelManager(
            checkpoint_path=str(checkpoint_path),
            device='cpu'
        )

        # Check model loaded
        assert manager.model is not None
        assert manager.device == torch.device('cpu')

    def test_model_manager_predict(self, hnet_config_small, hrm_config_small, temp_checkpoint_dir):
        """Test model manager prediction."""
        from codking.models.integration.pipeline import CodKingPipeline

        model = CodKingPipeline(
            task_type='classification',
            num_classes=2,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        )

        checkpoint_path = temp_checkpoint_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'task_type': 'classification',
                'num_classes': 2,
                'hnet_config': hnet_config_small,
                'hrm_config': hrm_config_small
            }
        }, checkpoint_path)

        manager = ModelManager(
            checkpoint_path=str(checkpoint_path),
            device='cpu'
        )

        # Test prediction
        texts = ["Test log entry", "Another test entry"]
        result = manager.predict(texts, return_scores=True)

        # Check result
        assert 'predictions' in result
        assert 'scores' in result
        assert len(result['predictions']) == 2
        assert len(result['scores']) == 2

    def test_model_manager_batch_predict(self, hnet_config_small, hrm_config_small, temp_checkpoint_dir):
        """Test batch prediction."""
        from codking.models.integration.pipeline import CodKingPipeline

        model = CodKingPipeline(
            task_type='classification',
            num_classes=2,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        )

        checkpoint_path = temp_checkpoint_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'task_type': 'classification',
                'num_classes': 2,
                'hnet_config': hnet_config_small,
                'hrm_config': hrm_config_small
            }
        }, checkpoint_path)

        manager = ModelManager(
            checkpoint_path=str(checkpoint_path),
            device='cpu',
            batch_size=8
        )

        # Large batch
        texts = [f"Test entry {i}" for i in range(20)]
        result = manager.predict(texts, return_scores=True)

        # Should process all
        assert len(result['predictions']) == 20


class TestInferenceServer:
    """Test FastAPI inference server."""

    @pytest.fixture
    def client(self, hnet_config_small, hrm_config_small, temp_checkpoint_dir, monkeypatch):
        """Create test client with mocked model."""
        from codking.models.integration.pipeline import CodKingPipeline

        # Create and save model
        model = CodKingPipeline(
            task_type='classification',
            num_classes=2,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        )

        checkpoint_path = temp_checkpoint_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'task_type': 'classification',
                'num_classes': 2,
                'hnet_config': hnet_config_small,
                'hrm_config': hrm_config_small
            }
        }, checkpoint_path)

        # Mock model manager initialization
        monkeypatch.setenv('CODKING_CHECKPOINT_PATH', str(checkpoint_path))

        # Reinitialize model manager
        global model_manager
        model_manager.load_model(str(checkpoint_path))

        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'model_loaded' in data

    def test_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/info")

        assert response.status_code == 200
        data = response.json()
        assert 'model_type' in data
        assert 'parameters' in data
        assert 'device' in data

    def test_detect_threat_endpoint(self, client):
        """Test threat detection endpoint."""
        response = client.post(
            "/detect_threat",
            json={
                "log_content": "CRITICAL: Multiple failed SSH attempts from 203.0.113.42"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert 'is_threat' in data
        assert 'threat_score' in data
        assert 'confidence' in data
        assert isinstance(data['is_threat'], bool)
        assert 0 <= data['threat_score'] <= 1

    def test_detect_threats_batch_endpoint(self, client):
        """Test batch threat detection."""
        response = client.post(
            "/detect_threats_batch",
            json={
                "log_contents": [
                    "Normal user login",
                    "CRITICAL: Attack detected",
                    "INFO: System backup complete"
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert 'results' in data
        assert len(data['results']) == 3

        # Each result should have required fields
        for result in data['results']:
            assert 'is_threat' in result
            assert 'threat_score' in result

    def test_classify_endpoint(self, client):
        """Test classification endpoint."""
        response = client.post(
            "/classify",
            json={
                "text": "Sample log entry for classification"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert 'class_id' in data
        assert 'class_probabilities' in data
        assert isinstance(data['class_id'], int)
        assert isinstance(data['class_probabilities'], list)

    def test_classify_batch_endpoint(self, client):
        """Test batch classification."""
        response = client.post(
            "/classify_batch",
            json={
                "texts": [
                    "First entry",
                    "Second entry",
                    "Third entry"
                ]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert 'results' in data
        assert len(data['results']) == 3

    def test_analyze_endpoint(self, client):
        """Test analysis endpoint."""
        response = client.post(
            "/analyze",
            json={
                "log_content": "Test log for analysis",
                "return_metadata": True
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'metadata' in data

        # Check metadata
        metadata = data['metadata']
        assert 'compression_ratio' in metadata
        assert 'num_cycles' in metadata
        assert 'inference_time_ms' in metadata

    def test_invalid_input_handling(self, client):
        """Test error handling for invalid inputs."""
        # Empty log content
        response = client.post(
            "/detect_threat",
            json={"log_content": ""}
        )
        assert response.status_code == 422  # Validation error

        # Missing required field
        response = client.post(
            "/classify",
            json={}
        )
        assert response.status_code == 422

    def test_batch_size_limits(self, client):
        """Test batch size limits."""
        # Try to send too many samples
        large_batch = [f"Entry {i}" for i in range(1001)]  # Assuming limit is 1000

        response = client.post(
            "/classify_batch",
            json={"texts": large_batch}
        )

        # Should either accept with batching or reject
        assert response.status_code in [200, 400, 413]


class TestPythonClient:
    """Test Python client library."""

    @pytest.fixture
    def mock_server(self, hnet_config_small, hrm_config_small, temp_checkpoint_dir):
        """Create mock server for client testing."""
        from codking.models.integration.pipeline import CodKingPipeline

        model = CodKingPipeline(
            task_type='classification',
            num_classes=2,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        )

        checkpoint_path = temp_checkpoint_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'task_type': 'classification',
                'num_classes': 2,
                'hnet_config': hnet_config_small,
                'hrm_config': hrm_config_small
            }
        }, checkpoint_path)

        # Start test server
        import os
        os.environ['CODKING_CHECKPOINT_PATH'] = str(checkpoint_path)

        return TestClient(app)

    def test_client_initialization(self):
        """Test client initialization."""
        client = CodKingClient(base_url="http://localhost:8000")

        assert client.base_url == "http://localhost:8000"
        assert client.session is not None

    def test_client_health_check(self, mock_server):
        """Test client health check."""
        # Mock the client to use TestClient
        client = CodKingClient(base_url="http://testserver")
        client._test_client = mock_server

        # Override _get to use test client
        original_get = client._get

        def mock_get(endpoint):
            response = client._test_client.get(endpoint)
            return response.json()

        client._get = mock_get

        health = client.health()
        assert health['status'] == 'healthy'

    def test_client_detect_threat(self, mock_server):
        """Test client threat detection."""
        client = CodKingClient(base_url="http://testserver")
        client._test_client = mock_server

        def mock_post(endpoint, data):
            response = client._test_client.post(endpoint, json=data)
            return response.json()

        client._post = mock_post

        result = client.detect_threat("CRITICAL: Attack detected")

        assert hasattr(result, 'is_threat')
        assert hasattr(result, 'threat_score')
        assert hasattr(result, 'confidence')

    def test_client_detect_threats_batch(self, mock_server):
        """Test client batch threat detection."""
        client = CodKingClient(base_url="http://testserver")
        client._test_client = mock_server

        def mock_post(endpoint, data):
            response = client._test_client.post(endpoint, json=data)
            return response.json()

        client._post = mock_post

        results = client.detect_threats_batch([
            "Normal log",
            "CRITICAL: Attack",
            "INFO: Status"
        ])

        assert hasattr(results, 'results')
        assert len(results.results) == 3

    def test_client_classify(self, mock_server):
        """Test client classification."""
        client = CodKingClient(base_url="http://testserver")
        client._test_client = mock_server

        def mock_post(endpoint, data):
            response = client._test_client.post(endpoint, json=data)
            return response.json()

        client._post = mock_post

        result = client.classify("Test entry")

        assert hasattr(result, 'class_id')
        assert hasattr(result, 'class_probabilities')

    def test_client_error_handling(self):
        """Test client error handling."""
        client = CodKingClient(base_url="http://invalid-server:9999")

        # Should raise connection error
        with pytest.raises(Exception):
            client.health()

    def test_client_timeout(self):
        """Test client timeout handling."""
        client = CodKingClient(
            base_url="http://localhost:8000",
            timeout=0.001  # Very short timeout
        )

        # Should timeout (or raise connection error if server not running)
        with pytest.raises(Exception):
            client.detect_threat("Test")


class TestIntegrationAPI:
    """Integration tests for complete API workflow."""

    @pytest.mark.slow
    def test_end_to_end_inference(self, hnet_config_small, hrm_config_small, temp_checkpoint_dir):
        """Test complete end-to-end inference workflow."""
        from codking.models.integration.pipeline import CodKingPipeline

        # 1. Train/save model
        model = CodKingPipeline(
            task_type='classification',
            num_classes=2,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        )

        checkpoint_path = temp_checkpoint_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'task_type': 'classification',
                'num_classes': 2,
                'hnet_config': hnet_config_small,
                'hrm_config': hrm_config_small
            }
        }, checkpoint_path)

        # 2. Load with model manager
        manager = ModelManager(
            checkpoint_path=str(checkpoint_path),
            device='cpu'
        )

        # 3. Run inference
        texts = [
            "Normal user login from 192.168.1.10",
            "CRITICAL: Multiple failed SSH attempts from 203.0.113.42"
        ]

        result = manager.predict(texts, return_scores=True, return_metadata=True)

        # 4. Verify results
        assert 'predictions' in result
        assert 'scores' in result
        assert 'metadata' in result

        assert len(result['predictions']) == 2
        assert 'compression_ratio' in result['metadata']
        assert 'inference_time_ms' in result['metadata']

    def test_concurrent_requests(self, hnet_config_small, hrm_config_small, temp_checkpoint_dir):
        """Test handling of concurrent requests."""
        from codking.models.integration.pipeline import CodKingPipeline
        import threading

        model = CodKingPipeline(
            task_type='classification',
            num_classes=2,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        )

        checkpoint_path = temp_checkpoint_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'task_type': 'classification',
                'num_classes': 2,
                'hnet_config': hnet_config_small,
                'hrm_config': hrm_config_small
            }
        }, checkpoint_path)

        manager = ModelManager(
            checkpoint_path=str(checkpoint_path),
            device='cpu'
        )

        results = []

        def make_request():
            result = manager.predict(["Test log entry"])
            results.append(result)

        # Launch concurrent requests
        threads = [threading.Thread(target=make_request) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should complete successfully
        assert len(results) == 5


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

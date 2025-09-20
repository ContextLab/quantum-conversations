"""
Comprehensive unit tests for ModelManager using real models.

Tests use actual models from HuggingFace to verify correctness.
Small models are used to ensure tests run quickly.
"""

import pytest
import torch
import gc
from quantum_conversations.model_manager import ModelManager, ModelConfig


class TestModelManager:
    """Test cases for ModelManager with real models."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up models after each test."""
        yield
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @pytest.fixture
    def manager(self):
        """Create a model manager instance."""
        return ModelManager()

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert manager.loaded_models == {}
        assert manager.default_device in ["cpu", "cuda", "mps"]

    def test_load_small_model(self, manager):
        """Test loading a small real model."""
        # Use the smallest Pythia model for testing
        model_name = "EleutherAI/pythia-70m"

        model, tokenizer = manager.load_model(
            model_name=model_name,
            device="cpu"  # Force CPU for consistent testing
        )

        # Verify model loaded
        assert model is not None
        assert tokenizer is not None

        # Verify model is on correct device
        assert str(next(model.parameters()).device) == "cpu"

        # Verify tokenizer works
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        assert len(tokens) > 0

        # Verify model can do forward pass
        input_ids = torch.tensor([tokens])
        with torch.no_grad():
            outputs = model(input_ids)
            assert outputs.logits is not None
            assert outputs.logits.shape[0] == 1
            assert outputs.logits.shape[1] == len(tokens)

    def test_model_caching(self, manager):
        """Test that models are cached properly."""
        model_name = "EleutherAI/pythia-70m"

        # Load model first time
        model1, tokenizer1 = manager.load_model(model_name, device="cpu")

        # Load model second time (should use cache)
        model2, tokenizer2 = manager.load_model(model_name, device="cpu")

        # Verify same objects returned
        assert model1 is model2
        assert tokenizer1 is tokenizer2

        # Verify model is in cache
        assert model_name in manager.loaded_models

    def test_unload_model(self, manager):
        """Test unloading models from memory."""
        model_name = "EleutherAI/pythia-70m"

        # Load model
        manager.load_model(model_name, device="cpu")
        assert model_name in manager.loaded_models

        # Unload model
        success = manager.unload_model(model_name)
        assert success
        assert model_name not in manager.loaded_models

        # Try unloading non-existent model
        success = manager.unload_model("nonexistent")
        assert not success

    def test_get_logits(self, manager):
        """Test getting raw logits from model."""
        model_name = "EleutherAI/pythia-70m"
        model, tokenizer = manager.load_model(model_name, device="cpu")

        # Create input
        text = "The quick brown"
        input_ids = tokenizer.encode(text, return_tensors="pt")

        # Get logits
        logits = ModelManager.get_logits(model, input_ids)

        # Verify shape
        assert logits.shape[0] == 1  # Batch size
        assert logits.shape[1] == input_ids.shape[1]  # Sequence length
        assert logits.shape[2] == tokenizer.vocab_size  # Vocab size

    def test_get_next_token_probabilities(self, manager):
        """Test getting next token probabilities."""
        model_name = "EleutherAI/pythia-70m"
        model, tokenizer = manager.load_model(model_name, device="cpu")

        # Create input
        text = "The weather is"
        input_ids = tokenizer.encode(text, return_tensors="pt")

        # Get probabilities
        probs = ModelManager.get_next_token_probabilities(
            model=model,
            input_ids=input_ids,
            temperature=1.0
        )

        # Verify probabilities
        assert probs.shape[-1] == tokenizer.vocab_size
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_temperature_effect(self, manager):
        """Test that temperature affects probability distribution."""
        model_name = "EleutherAI/pythia-70m"
        model, tokenizer = manager.load_model(model_name, device="cpu")

        text = "Hello"
        input_ids = tokenizer.encode(text, return_tensors="pt")

        # Get probabilities with different temperatures
        probs_low = ModelManager.get_next_token_probabilities(
            model, input_ids, temperature=0.1
        )
        probs_high = ModelManager.get_next_token_probabilities(
            model, input_ids, temperature=2.0
        )

        # Low temperature should have higher max probability (more focused)
        assert probs_low.max() > probs_high.max()

        # High temperature should have higher entropy (more spread out)
        entropy_low = -(probs_low * torch.log(probs_low + 1e-10)).sum()
        entropy_high = -(probs_high * torch.log(probs_high + 1e-10)).sum()
        assert entropy_high > entropy_low

    def test_top_k_filtering(self, manager):
        """Test top-k filtering."""
        model_name = "EleutherAI/pythia-70m"
        model, tokenizer = manager.load_model(model_name, device="cpu")

        text = "The"
        input_ids = tokenizer.encode(text, return_tensors="pt")

        # Get probabilities with top-k
        probs = ModelManager.get_next_token_probabilities(
            model, input_ids, temperature=1.0, top_k=10
        )

        # Count non-zero probabilities
        non_zero_count = (probs > 0).sum().item()
        assert non_zero_count <= 10

    def test_top_p_filtering(self, manager):
        """Test nucleus (top-p) filtering."""
        model_name = "EleutherAI/pythia-70m"
        model, tokenizer = manager.load_model(model_name, device="cpu")

        text = "The"
        input_ids = tokenizer.encode(text, return_tensors="pt")

        # Get probabilities with top-p
        probs = ModelManager.get_next_token_probabilities(
            model, input_ids, temperature=1.0, top_p=0.9
        )

        # Verify cumulative probability of non-zero tokens
        sorted_probs, _ = torch.sort(probs, descending=True)
        non_zero_probs = sorted_probs[sorted_probs > 0]
        cumsum = non_zero_probs.cumsum(dim=0)

        # The cumulative sum should be close to 1 (might be slightly less due to filtering)
        assert cumsum[-1] >= 0.85  # Allow some tolerance

    def test_model_info(self, manager):
        """Test getting model information."""
        model_name = "EleutherAI/pythia-70m"

        # Before loading
        info = manager.get_model_info(model_name)
        assert info["name"] == model_name
        assert not info["loaded"]

        # After loading
        manager.load_model(model_name, device="cpu")
        info = manager.get_model_info(model_name)
        assert info["loaded"]
        assert "vocab_size" in info
        assert "num_parameters" in info
        assert info["device"] == "cpu"

    def test_verify_model_availability(self):
        """Test checking model availability."""
        # Test with valid model
        assert ModelManager.verify_model_availability("EleutherAI/pythia-70m")
        assert ModelManager.verify_model_availability("gpt2")

        # Test with invalid model
        assert not ModelManager.verify_model_availability("nonexistent/model-xyz")

    def test_is_open_weights_model(self):
        """Test checking if model is open-weights."""
        # Open-weights models
        assert ModelManager.is_open_weights_model("EleutherAI/pythia-70m")
        assert ModelManager.is_open_weights_model("gpt2")
        assert ModelManager.is_open_weights_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        # Closed/API-only models (should return False)
        assert not ModelManager.is_open_weights_model("openai/gpt-4")
        assert not ModelManager.is_open_weights_model("anthropic/claude")

    def test_multiple_models(self, manager):
        """Test loading multiple models simultaneously."""
        models = [
            "EleutherAI/pythia-70m",
            "distilgpt2"  # Another small model
        ]

        loaded = []
        for model_name in models:
            model, tokenizer = manager.load_model(model_name, device="cpu")
            loaded.append((model, tokenizer))

        # Verify all models loaded
        assert len(manager.list_cached_models()) == 2

        # Verify each model works
        for (model, tokenizer), model_name in zip(loaded, models):
            input_ids = tokenizer.encode("Test", return_tensors="pt")
            with torch.no_grad():
                outputs = model(input_ids)
                assert outputs.logits is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_loading(self, manager):
        """Test loading model on GPU if available."""
        model_name = "EleutherAI/pythia-70m"
        model, tokenizer = manager.load_model(model_name, device="cuda")

        # Verify on GPU
        assert "cuda" in str(next(model.parameters()).device)

        # Test forward pass on GPU
        input_ids = tokenizer.encode("Test", return_tensors="pt").cuda()
        with torch.no_grad():
            outputs = model(input_ids)
            assert outputs.logits.device.type == "cuda"


class TestModelConfig:
    """Test ModelConfig helper class."""

    def test_get_test_config(self):
        """Test getting test configuration."""
        config = ModelConfig.get_test_config()

        assert config["model_name"] == "EleutherAI/pythia-70m"
        assert "torch_dtype" in config
        assert "low_cpu_mem_usage" in config

    def test_get_production_config(self):
        """Test getting production configuration."""
        config = ModelConfig.get_production_config("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        assert config["model_name"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert "torch_dtype" in config
        assert config["low_cpu_mem_usage"] is True

    def test_model_lists(self):
        """Test predefined model lists."""
        assert len(ModelConfig.SMALL_MODELS) > 0
        assert len(ModelConfig.MEDIUM_MODELS) > 0
        assert "EleutherAI/pythia-70m" in ModelConfig.SMALL_MODELS
        assert ModelConfig.DEFAULT_MODEL == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert ModelConfig.TEST_MODEL == "EleutherAI/pythia-70m"
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from mate_engine_ollama._client import GenerateResult, OllamaClient


VERSION_RESPONSE = {"version": "0.23.4"}

TAGS_RESPONSE = {
    "models": [
        {
            "name": "llama3.2:3b",
            "model": "llama3.2:3b",
            "size": 2019393189,
            "digest": "a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72",
            "details": {"parameter_size": "3.2B", "quantization_level": "Q4_K_M"},
        }
    ]
}

GENERATE_RESPONSE = {
    "model": "llama3.2:3b",
    "response": "Paris.",
    "done": True,
    "prompt_eval_count": 26,
    "prompt_eval_duration": 130079000,
    "eval_count": 10,
    "eval_duration": 423271000,
    "load_duration": 5012345,
    "total_duration": 560000000,
}


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    mock.json.return_value = data
    mock.raise_for_status = MagicMock()
    return mock


class TestOllamaClientUnit:
    def test_version(self):
        client = OllamaClient()
        with patch("httpx.Client") as mock_cls:
            mock_http = mock_cls.return_value.__enter__.return_value
            mock_http.get.return_value = _mock_response(VERSION_RESPONSE)
            assert client.version() == "0.23.4"

    def test_list_models(self):
        client = OllamaClient()
        with patch("httpx.Client") as mock_cls:
            mock_http = mock_cls.return_value.__enter__.return_value
            mock_http.get.return_value = _mock_response(TAGS_RESPONSE)
            models = client.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "llama3.2:3b"

    def test_generate_result_metrics(self):
        client = OllamaClient()
        with patch("httpx.Client") as mock_cls:
            mock_http = mock_cls.return_value.__enter__.return_value
            mock_http.post.return_value = _mock_response(GENERATE_RESPONSE)
            result = client.generate("llama3.2:3b", "What is the capital of France?")

        assert result.model == "llama3.2:3b"
        assert result.prompt_tokens == 26
        assert result.generated_tokens == 10
        assert result.eval_duration_ns == 423271000
        assert result.load_duration_ns == 5012345

    def test_tokens_per_second(self):
        result = GenerateResult(
            model="test",
            prompt_tokens=26,
            generated_tokens=100,
            total_duration_ns=1_000_000_000,
            load_duration_ns=0,
            prompt_eval_duration_ns=100_000_000,
            eval_duration_ns=1_000_000_000,
        )
        assert result.tokens_per_second == pytest.approx(100.0)

    def test_tokens_per_second_zero_duration(self):
        result = GenerateResult(
            model="test",
            prompt_tokens=0,
            generated_tokens=0,
            total_duration_ns=0,
            load_duration_ns=0,
            prompt_eval_duration_ns=0,
            eval_duration_ns=0,
        )
        assert result.tokens_per_second == 0.0

    def test_is_alive_returns_false_on_connection_error(self):
        client = OllamaClient()
        with patch("httpx.Client") as mock_cls:
            mock_http = mock_cls.return_value.__enter__.return_value
            mock_http.get.side_effect = httpx.ConnectError("refused")
            assert client.is_alive() is False

    def test_is_alive_returns_false_on_timeout(self):
        client = OllamaClient()
        with patch("httpx.Client") as mock_cls:
            mock_http = mock_cls.return_value.__enter__.return_value
            mock_http.get.side_effect = httpx.TimeoutException("timeout")
            assert client.is_alive() is False


class TestModelInfo:
    def test_digest_used_as_file_hash(self):
        from mate_engine_ollama._models import model_info_from_ollama
        info = model_info_from_ollama("llama3.2:3b", TAGS_RESPONSE["models"])
        assert info.source == "ollama"
        assert info.source_ref == "llama3.2:3b"
        assert info.file_hash == "sha256:a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72"
        assert info.file_hash_available is True

    def test_unknown_model_no_hash(self):
        from mate_engine_ollama._models import model_info_from_ollama
        info = model_info_from_ollama("unknown:latest", TAGS_RESPONSE["models"])
        assert info.file_hash is None
        assert info.file_hash_available is False


@pytest.mark.integration
class TestOllamaIntegration:
    """Requires a running Ollama server at localhost:11434."""

    def test_is_alive(self):
        client = OllamaClient()
        assert client.is_alive() is True

    def test_version_is_string(self):
        client = OllamaClient()
        v = client.version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_list_models_is_list(self):
        client = OllamaClient()
        models = client.list_models()
        assert isinstance(models, list)

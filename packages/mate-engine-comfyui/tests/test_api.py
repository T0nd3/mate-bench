from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mate_engine_comfyui._api import ComfyUiClient
from mate_engine_comfyui._workflow import txt2img_workflow


# ── txt2img_workflow ──────────────────────────────────────────────────────────

class TestTxt2ImgWorkflow:
    def test_contains_required_nodes(self):
        wf = txt2img_workflow("model.safetensors", "a cat", "", 20, 512, 512, 7.0, "euler_ancestral")
        assert "3" in wf  # KSampler
        assert "4" in wf  # CheckpointLoaderSimple
        assert "5" in wf  # EmptyLatentImage
        assert "9" in wf  # SaveImage

    def test_model_name_set(self):
        wf = txt2img_workflow("my_model.safetensors", "a cat", "", 20, 512, 512, 7.0, "euler_ancestral")
        assert wf["4"]["inputs"]["ckpt_name"] == "my_model.safetensors"

    def test_resolution_set(self):
        wf = txt2img_workflow("m", "p", "", 20, 768, 512, 7.0, "euler")
        assert wf["5"]["inputs"]["width"] == 768
        assert wf["5"]["inputs"]["height"] == 512

    def test_steps_set(self):
        wf = txt2img_workflow("m", "p", "", 30, 512, 512, 7.0, "euler")
        assert wf["3"]["inputs"]["steps"] == 30

    def test_seed_deterministic(self):
        wf = txt2img_workflow("m", "p", "", 20, 512, 512, 7.0, "euler", seed=1234)
        assert wf["3"]["inputs"]["seed"] == 1234

    def test_positive_prompt_set(self):
        wf = txt2img_workflow("m", "a red apple", "blurry", 20, 512, 512, 7.0, "euler")
        assert wf["6"]["inputs"]["text"] == "a red apple"
        assert wf["7"]["inputs"]["text"] == "blurry"


# ── ComfyUiClient ─────────────────────────────────────────────────────────────

def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_data
    r.raise_for_status = MagicMock()
    return r


class TestComfyUiClient:
    def test_queue_prompt_returns_id(self):
        client = ComfyUiClient()
        with patch("mate_engine_comfyui._api.requests.post") as mock_post:
            mock_post.return_value = _mock_response({"prompt_id": "abc-123"})
            pid = client.queue_prompt({"3": {}})
        assert pid == "abc-123"

    def test_wait_for_completion_success(self):
        client = ComfyUiClient()
        completed_entry = {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {"9": {"images": [{"filename": "mate_bench_00001_.png"}]}},
        }
        with patch("mate_engine_comfyui._api.requests.get") as mock_get:
            mock_get.return_value = _mock_response({"abc-123": completed_entry})
            entry = client.wait_for_completion("abc-123")
        assert entry["status"]["completed"] is True

    def test_run_workflow_returns_image_result(self):
        client = ComfyUiClient()
        prompt_resp = _mock_response({"prompt_id": "xyz-456"})
        history_resp = _mock_response({
            "xyz-456": {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {"9": {"images": [{"filename": "mate_bench_00001_.png"}]}},
            }
        })
        with patch("mate_engine_comfyui._api.requests.post", return_value=prompt_resp):
            with patch("mate_engine_comfyui._api.requests.get", return_value=history_resp):
                result = client.run_workflow({"3": {}})
        assert result.generation_time_s >= 0.0
        assert "mate_bench_00001_.png" in result.filenames

    def test_wait_raises_on_error_status(self):
        client = ComfyUiClient()
        with patch("mate_engine_comfyui._api.requests.get") as mock_get:
            mock_get.return_value = _mock_response({
                "fail-id": {"status": {"completed": False, "status_str": "error"}}
            })
            with pytest.raises(RuntimeError, match="error"):
                client.wait_for_completion("fail-id")

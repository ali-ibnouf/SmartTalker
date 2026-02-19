"""Tests for Video and Upscale engines."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.utils.exceptions import VideoError, UpscaleError

@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock heavy DL dependencies to avoid ImportErrors during logic tests."""
    mocks = {
        "torch": MagicMock(),
        "diffusers": MagicMock(),
        "transformers": MagicMock(),
        "realesrgan": MagicMock(),
        "basicsr": MagicMock(),
        "basicsr.archs.rrdbnet_arch": MagicMock(),
    }
    with patch.dict("sys.modules", mocks):
        yield

# =============================================================================
# Video Engine Tests
# =============================================================================

class TestVideoEngine:
    """Tests for VideoEngine."""

    def test_init(self, config):
        """Engine initializes without loading model."""
        from src.pipeline.video import VideoEngine
        engine = VideoEngine(config)
        assert not engine.is_loaded

    def test_load_missing_model_raises(self, config):
        """Loading missing model raises VideoError."""
        from src.pipeline.video import VideoEngine
        engine = VideoEngine(config)
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(VideoError, match="not found"):
                engine.load()

    @pytest.mark.asyncio
    async def test_generate_without_load_raises(self, config):
        """Generating without load() raises VideoError."""
        from src.pipeline.video import VideoEngine
        engine = VideoEngine(config)
        with pytest.raises(VideoError, match="not loaded"):
            await engine.generate("audio.wav", "image.png")

    @pytest.mark.asyncio
    async def test_run_inference_success(self, config):
        """Inference runs successfully."""
        from src.pipeline.video import VideoEngine
        engine = VideoEngine(config)
        engine._loaded = True
        engine._model = {"model_dir": "fake_dir"}
        
        # Mock inputs existence
        with patch("pathlib.Path.exists", return_value=True):
            # Mock subprocess
            process_mock = AsyncMock()
            process_mock.returncode = 0
            process_mock.communicate.return_value = (b"", b"")
            
            with patch("asyncio.create_subprocess_exec", return_value=process_mock):
                with patch("src.utils.audio.get_duration", return_value=5.0):
                    duration = await engine._run_inference("a.wav", "i.png", "o.mp4")
                    assert duration == 5.0

    @pytest.mark.asyncio
    async def test_run_inference_failure(self, config):
        """Inference failure raises VideoError."""
        from src.pipeline.video import VideoEngine
        engine = VideoEngine(config)
        engine._loaded = True
        engine._model = {"model_dir": "fake_dir"}

        with patch("pathlib.Path.exists", return_value=True):
            process_mock = AsyncMock()
            process_mock.returncode = 1
            process_mock.communicate.return_value = (b"", b"Error details")

            with patch("asyncio.create_subprocess_exec", return_value=process_mock):
                with pytest.raises(VideoError, match="failed"):
                    await engine._run_inference("a.wav", "i.png", "o.mp4")


# =============================================================================
# Upscale Engine Tests
# =============================================================================

class TestUpscaleEngine:
    """Tests for UpscaleEngine."""

    def test_init(self, config):
        """Engine initializes without loading model."""
        from src.pipeline.upscale import UpscaleEngine
        engine = UpscaleEngine(config)
        assert not engine.is_loaded

    @pytest.mark.asyncio
    async def test_enhance_no_upscale_needed(self, config):
        """Skips upscale if resolution is already sufficient."""
        from src.pipeline.upscale import UpscaleEngine
        engine = UpscaleEngine(config)
        engine._loaded = True
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("src.utils.video.get_video_info", return_value={"width": 1920, "height": 1080, "fps": 25}):
                result = await engine.enhance("video.mp4", target_resolution="1080p")
                assert result.scale_factor == 1.0
                assert result.latency_ms == 0

    @pytest.mark.asyncio
    async def test_process_video_frames_calls_subprocess(self, config):
        """Upscaling calls the correct subprocess command."""
        from src.pipeline.upscale import UpscaleEngine
        engine = UpscaleEngine(config)
        
        # Mock subprocess
        process_mock = AsyncMock()
        process_mock.returncode = 0
        process_mock.communicate.return_value = (b"", b"")
        
        with patch("asyncio.create_subprocess_exec", return_value=process_mock) as mock_exec:
            await engine._process_video_frames("in.mp4", "out.mp4", 1920, 1080, 25.0)
            mock_exec.assert_called()
            args = mock_exec.call_args[0]
            assert "realesrgan" in args

    @pytest.mark.asyncio
    async def test_ffmpeg_fallback(self, config):
        """Fallback to ffmpeg works."""
        from src.pipeline.upscale import UpscaleEngine
        engine = UpscaleEngine(config)
        
        # Mock subprocess
        process_mock = AsyncMock()
        process_mock.returncode = 0
        process_mock.communicate.return_value = (b"", b"")
        
        with patch("asyncio.create_subprocess_exec", return_value=process_mock) as mock_exec:
            await engine._ffmpeg_upscale("in.mp4", "out.mp4", 1920, 1080, 25.0)
            args = mock_exec.call_args[0]
            assert "ffmpeg" in args

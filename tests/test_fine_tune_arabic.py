"""Unit tests for scripts/fine_tune_arabic.py.

All torch / torchaudio modules are faked at the sys.modules level so the
script can be imported without GPU dependencies.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Lightweight fake tensor ─────────────────────────────────────────────────


class FakeTensor:
    """Minimal tensor stand-in that supports the numeric ops used by the SUT."""

    def __init__(self, data: list | float, shape: tuple[int, ...] | None = None):
        if isinstance(data, (int, float)):
            # scalar
            self._data = [data]
            self.shape = shape if shape is not None else ()
        else:
            self._data = list(data)
            self.shape = shape if shape is not None else (len(self._data),)

    # ---- numeric helpers used by evaluate / collate_fn ----
    def item(self) -> float:
        return float(self._data[0])

    def dim(self) -> int:
        return len(self.shape)

    def to(self, *_a, **_kw) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def mean(self, dim: int = 0, keepdim: bool = False) -> "FakeTensor":
        """Collapse first axis (used for stereo → mono)."""
        if len(self.shape) == 2 and dim == 0:
            new_len = self.shape[1]
            new_shape = (1, new_len) if keepdim else (new_len,)
            return FakeTensor([0.0] * new_len, shape=new_shape)
        return self

    def squeeze(self, dim: int = 0) -> "FakeTensor":
        new_shape = tuple(s for i, s in enumerate(self.shape) if not (i == dim and s == 1))
        if not new_shape:
            new_shape = (1,)
        return FakeTensor(self._data[:1], shape=new_shape)

    def __len__(self) -> int:
        return self.shape[0]


def _fake_pad(tensor: FakeTensor, pad_spec: tuple) -> FakeTensor:
    """F.pad(w, (0, extra))  →  extend the last dim."""
    extra = pad_spec[1] if len(pad_spec) >= 2 else 0
    new_len = tensor.shape[-1] + extra
    new_shape = (*tensor.shape[:-1], new_len)
    return FakeTensor([0.0] * new_len, shape=new_shape)


def _fake_stack(tensors: list) -> FakeTensor:
    """torch.stack([t, ...])  →  batch dim prepended."""
    inner = tensors[0].shape
    return FakeTensor([0.0], shape=(len(tensors), *inner))


# ── sys.modules-level torch mock ────────────────────────────────────────────

# Build mock hierarchy
_torch = MagicMock()
_torch_nn = MagicMock()
_torch_nn_functional = MagicMock()
_torch_utils = MagicMock()
_torch_utils_data = MagicMock()
_torch_amp = MagicMock()
_torch_optim = MagicMock()
_torchaudio = MagicMock()
_torchaudio_functional = MagicMock()

# Wire real classes so isinstance / issubclass work
_torch_nn.Module = type("Module", (), {})
_torch_utils_data.Dataset = type("Dataset", (), {})
_torch_utils_data.DataLoader = MagicMock()
_torch_utils_data.random_split = MagicMock()

# Make torch.Tensor point to FakeTensor for isinstance checks
_torch.Tensor = FakeTensor

# Real behaviour for stack / pad
_torch.stack = _fake_stack
_torch_nn_functional.pad = _fake_pad

# no_grad / autocast → no-op context managers
_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
_torch_amp.autocast.return_value.__enter__ = MagicMock(return_value=None)
_torch_amp.autocast.return_value.__exit__ = MagicMock(return_value=False)

# Attach submodules
_torch.nn = _torch_nn
_torch.nn.functional = _torch_nn_functional
_torch.nn.Module = _torch_nn.Module
_torch.utils = _torch_utils
_torch.utils.data = _torch_utils_data
_torch.amp = _torch_amp
_torch.optim = _torch_optim
_torchaudio.functional = _torchaudio_functional

# Insert into sys.modules BEFORE importing the script
sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch.nn", _torch_nn)
sys.modules.setdefault("torch.nn.functional", _torch_nn_functional)
sys.modules.setdefault("torch.utils", _torch_utils)
sys.modules.setdefault("torch.utils.data", _torch_utils_data)
sys.modules.setdefault("torch.amp", _torch_amp)
sys.modules.setdefault("torch.optim", _torch_optim)
sys.modules.setdefault("torchaudio", _torchaudio)
sys.modules.setdefault("torchaudio.functional", _torchaudio_functional)

# Now import the module under test
fta = importlib.import_module("scripts.fine_tune_arabic")

# ── Tests ───────────────────────────────────────────────────────────────────


class TestParseArgs:
    """Tests for parse_args()."""

    def test_required_data_dir(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog"])
        with pytest.raises(SystemExit):
            fta.parse_args()

    def test_defaults(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--data-dir", "/tmp/d"])
        args = fta.parse_args()
        assert args.data_dir == Path("/tmp/d")
        assert args.epochs == 10
        assert args.batch_size == 4
        assert args.lr == pytest.approx(1e-4)
        assert args.grad_accum == 4
        assert args.save_every == 500
        assert args.device == "cuda:0"
        assert args.resume is None
        assert args.eval_only is False

    def test_all_flags(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--data-dir", "/data",
                "--output-dir", "/out",
                "--epochs", "20",
                "--batch-size", "8",
                "--lr", "3e-5",
                "--grad-accum", "2",
                "--save-every", "100",
                "--device", "cpu",
                "--resume", "/ckpt.pt",
                "--eval-only",
            ],
        )
        args = fta.parse_args()
        assert args.data_dir == Path("/data")
        assert args.output_dir == Path("/out")
        assert args.epochs == 20
        assert args.batch_size == 8
        assert args.lr == pytest.approx(3e-5)
        assert args.grad_accum == 2
        assert args.save_every == 100
        assert args.device == "cpu"
        assert args.resume == Path("/ckpt.pt")
        assert args.eval_only is True


class TestLoadManifest:
    """Tests for load_manifest()."""

    @staticmethod
    def _write_manifest(data_dir: Path, rows: list[dict]) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        manifest = data_dir / "manifest.csv"
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            writer = None
            for row in rows:
                if writer is None:
                    writer = __import__("csv").DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)

    def test_valid_manifest(self, tmp_path):
        data_dir = tmp_path / "data"
        self._write_manifest(data_dir, [
            {"audio_path": "a.wav", "text": "مرحبا", "speaker_id": "sp1"},
            {"audio_path": "b.wav", "text": "أهلا", "speaker_id": "sp2"},
        ])
        with (
            patch.object(fta, "validate_audio", return_value=True),
            patch.object(fta, "get_duration", return_value=5.0),
            patch.object(fta, "normalize_arabic_text", side_effect=lambda t: t),
        ):
            entries = fta.load_manifest(data_dir)
        assert len(entries) == 2
        assert entries[0]["text"] == "مرحبا"

    def test_missing_manifest(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            fta.load_manifest(tmp_path / "nope")

    def test_skips_invalid_audio(self, tmp_path):
        data_dir = tmp_path / "data"
        self._write_manifest(data_dir, [
            {"audio_path": "bad.wav", "text": "نص", "speaker_id": "sp1"},
        ])
        with (
            patch.object(fta, "validate_audio", side_effect=Exception("corrupt")),
        ):
            entries = fta.load_manifest(data_dir)
        assert len(entries) == 0

    def test_skips_out_of_range_duration(self, tmp_path):
        data_dir = tmp_path / "data"
        self._write_manifest(data_dir, [
            {"audio_path": "short.wav", "text": "نص", "speaker_id": "sp1"},
        ])
        with (
            patch.object(fta, "validate_audio", return_value=True),
            patch.object(fta, "get_duration", return_value=1.0),
        ):
            entries = fta.load_manifest(data_dir)
        assert len(entries) == 0


class TestArabicTTSDataset:
    """Tests for ArabicTTSDataset."""

    def test_len(self):
        ds = fta.ArabicTTSDataset([{"a": 1}, {"a": 2}, {"a": 3}])
        assert len(ds) == 3

    def test_getitem_mono(self):
        entry = {"audio_path": Path("/fake.wav"), "text": "hi", "speaker_id": "s1"}
        ds = fta.ArabicTTSDataset([entry], sample_rate=22050)

        mono_wave = FakeTensor([0.0] * 100, shape=(1, 100))
        _torchaudio.load.return_value = (mono_wave, 22050)
        result = ds[0]

        assert result["text"] == "hi"
        # squeeze(0) removes leading 1 → shape (100,)
        assert result["waveform"].shape == (100,)

    def test_getitem_resamples(self):
        entry = {"audio_path": Path("/fake.wav"), "text": "hi", "speaker_id": "s1"}
        ds = fta.ArabicTTSDataset([entry], sample_rate=22050)

        wave = FakeTensor([0.0] * 80, shape=(1, 80))
        _torchaudio.load.return_value = (wave, 16000)

        resampled = FakeTensor([0.0] * 110, shape=(1, 110))
        _torchaudio_functional.resample.return_value = resampled

        result = ds[0]

        _torchaudio_functional.resample.assert_called_once_with(wave, 16000, 22050)
        assert result["waveform"].shape == (110,)


class TestCollateFn:
    """Tests for collate_fn()."""

    def test_pads_to_max_length(self):
        batch = [
            {"waveform": FakeTensor([0.0] * 50, shape=(50,)), "text": "a", "speaker_id": "s1"},
            {"waveform": FakeTensor([0.0] * 80, shape=(80,)), "text": "b", "speaker_id": "s2"},
        ]
        out = fta.collate_fn(batch)
        # stack of 2 tensors each padded to 80
        assert out["waveform"].shape[0] == 2
        assert out["waveform"].shape[1] == 80

    def test_preserves_metadata(self):
        batch = [
            {"waveform": FakeTensor([0.0] * 10, shape=(10,)), "text": "مرحبا", "speaker_id": "s1"},
            {"waveform": FakeTensor([0.0] * 10, shape=(10,)), "text": "أهلا", "speaker_id": "s2"},
        ]
        out = fta.collate_fn(batch)
        assert out["text"] == ["مرحبا", "أهلا"]
        assert out["speaker_id"] == ["s1", "s2"]


class TestEvaluate:
    """Tests for evaluate()."""

    def test_evaluate_returns_avg_loss(self):
        model = MagicMock()
        model.eval = MagicMock()
        model.train = MagicMock()

        loss1 = FakeTensor(2.0, shape=())
        loss2 = FakeTensor(4.0, shape=())
        # model(waveform, text, speaker_id) returns a scalar tensor
        model.side_effect = [loss1, loss2]

        batch1 = {
            "waveform": FakeTensor([0.0], shape=(1, 100)),
            "text": ["a"],
            "speaker_id": ["s1"],
        }
        batch2 = {
            "waveform": FakeTensor([0.0], shape=(1, 100)),
            "text": ["b"],
            "speaker_id": ["s2"],
        }

        val_loader = [batch1, batch2]
        device = MagicMock()

        avg = fta.evaluate(model, val_loader, device)
        assert avg == pytest.approx(3.0)
        model.eval.assert_called_once()
        model.train.assert_called_once()


class TestSynthesizeSamples:
    """Tests for synthesize_samples()."""

    def test_generates_samples(self, tmp_path):
        cosyvoice = MagicMock()
        wav_tensor = FakeTensor([0.0], shape=(1, 16000))
        cosyvoice.inference_sft.return_value = iter([{"tts_speech": wav_tensor}])

        entries = [{"audio_path": Path("/a.wav"), "text": "مرحبا", "speaker_id": "sp1"}]
        ds = fta.ArabicTTSDataset(entries)

        with patch.object(fta, "get_settings") as mock_settings:
            mock_settings.return_value.tts_sample_rate = 22050
            fta.synthesize_samples(cosyvoice, ds, tmp_path, n=1)

        _torchaudio.save.assert_called_once()

    def test_handles_generation_error(self, tmp_path):
        cosyvoice = MagicMock()
        cosyvoice.inference_sft.side_effect = RuntimeError("boom")

        entries = [{"audio_path": Path("/a.wav"), "text": "نص", "speaker_id": "sp1"}]
        ds = fta.ArabicTTSDataset(entries)

        _torchaudio.save.reset_mock()
        with patch.object(fta, "get_settings") as mock_settings:
            mock_settings.return_value.tts_sample_rate = 22050
            fta.synthesize_samples(cosyvoice, ds, tmp_path, n=1)

        _torchaudio.save.assert_not_called()


class TestMain:
    """Tests for main()."""

    def test_eval_only_without_resume_exits(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["prog", "--data-dir", "/d", "--eval-only"]
        )
        with pytest.raises(SystemExit) as exc_info:
            fta.main()
        assert exc_info.value.code == 1

    def test_dispatches_to_train(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["prog", "--data-dir", "foo"])
        with patch.object(fta, "train") as mock_train:
            fta.main()
        mock_train.assert_called_once()

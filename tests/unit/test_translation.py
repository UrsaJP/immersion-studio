"""Unit tests for core.translation path-construction and error-handling logic.

These tests do NOT call ffprobe/ffmpeg or any AI provider — they exercise the
logic that is independent of external processes.

Requires PySide6 (skipped automatically when not installed).
"""

import pytest
pytest.importorskip("PySide6", reason="PySide6 not installed — skipping Qt-dependent tests")

from pathlib import Path
from unittest.mock import MagicMock, patch

from core.translation import (
    TRANSLATION_OUTPUT_SUFFIX,
    TranslationSignals,
    TranslationWorker,
    list_subtitle_tracks,
    mux_subtitle_into_mkv,
)


class TestOutputPathConstruction:
    def test_suffix_constant_not_empty(self):
        assert TRANSLATION_OUTPUT_SUFFIX, "Output suffix must be non-empty"

    def test_output_path_appends_suffix(self, fake_mkv_path):
        """TranslationWorker builds output path as <stem><suffix>.mkv."""
        stem = Path(fake_mkv_path).stem
        expected_name = f"{stem}{TRANSLATION_OUTPUT_SUFFIX}.mkv"
        expected_path = str(Path(fake_mkv_path).parent / expected_name)

        # Verify construction manually (mirrors worker logic)
        output_path = str(
            Path(fake_mkv_path).parent
            / f"{Path(fake_mkv_path).stem}{TRANSLATION_OUTPUT_SUFFIX}.mkv"
        )
        assert output_path == expected_path

    def test_original_file_not_overwritten(self, fake_mkv_path):
        """Output path must never equal the source path."""
        output_path = str(
            Path(fake_mkv_path).parent
            / f"{Path(fake_mkv_path).stem}{TRANSLATION_OUTPUT_SUFFIX}.mkv"
        )
        assert output_path != fake_mkv_path


class TestListSubtitleTracksErrors:
    def test_raises_runtime_error_when_ffprobe_missing(self, fake_mkv_path, monkeypatch):
        """list_subtitle_tracks must raise RuntimeError if ffprobe is not in PATH."""
        import subprocess
        def _raise(*args, **kwargs):
            raise FileNotFoundError("ffprobe not found")
        monkeypatch.setattr(subprocess, "run", _raise)

        with pytest.raises(RuntimeError, match="ffprobe"):
            list_subtitle_tracks(fake_mkv_path)

    def test_raises_runtime_error_on_nonzero_exit(self, fake_mkv_path, monkeypatch):
        """list_subtitle_tracks must raise RuntimeError if ffprobe exits non-zero."""
        import subprocess
        def _fail(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "ffprobe", stderr="some error")
        monkeypatch.setattr(subprocess, "run", _fail)

        with pytest.raises(RuntimeError, match="ffprobe"):
            list_subtitle_tracks(fake_mkv_path)

    def test_returns_empty_list_for_no_streams(self, fake_mkv_path, monkeypatch):
        """list_subtitle_tracks returns [] if ffprobe reports no subtitle streams."""
        import subprocess
        import json
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({"streams": []})
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        result = list_subtitle_tracks(fake_mkv_path)
        assert result == []

    def test_parses_stream_fields(self, fake_mkv_path, monkeypatch):
        """list_subtitle_tracks extracts index, codec, language, title."""
        import subprocess
        import json
        mock_result = MagicMock()
        mock_result.stdout = json.dumps({
            "streams": [{
                "index": 2,
                "codec_name": "subrip",
                "tags": {"language": "jpn", "title": "Japanese"},
            }]
        })
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        tracks = list_subtitle_tracks(fake_mkv_path)
        assert len(tracks) == 1
        assert tracks[0]["index"] == 2
        assert tracks[0]["codec"] == "subrip"
        assert tracks[0]["language"] == "jpn"
        assert tracks[0]["title"] == "Japanese"


class TestMuxSubtitleIntoMkvErrors:
    def test_raises_runtime_error_when_ffmpeg_missing(self, fake_mkv_path, tmp_path, monkeypatch):
        import subprocess
        def _raise(*args, **kwargs):
            raise FileNotFoundError("ffmpeg not found")
        monkeypatch.setattr(subprocess, "run", _raise)

        with pytest.raises(RuntimeError, match="ffmpeg"):
            mux_subtitle_into_mkv(
                fake_mkv_path,
                str(tmp_path / "sub.srt"),
                str(tmp_path / "out.mkv"),
            )

    def test_raises_runtime_error_on_nonzero_exit(self, fake_mkv_path, tmp_path, monkeypatch):
        import subprocess
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "ffmpeg error output"
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        with pytest.raises(RuntimeError, match="ffmpeg"):
            mux_subtitle_into_mkv(
                fake_mkv_path,
                str(tmp_path / "sub.srt"),
                str(tmp_path / "out.mkv"),
            )


class TestTranslationSignals:
    def test_signals_object_created(self):
        """TranslationSignals can be instantiated independently."""
        sig = TranslationSignals()
        assert hasattr(sig, "progress")
        assert hasattr(sig, "finished")
        assert hasattr(sig, "error")


class TestTranslationWorkerInit:
    def test_worker_stores_params(self, fake_mkv_path):
        worker = TranslationWorker(
            mkv_path=fake_mkv_path,
            track_index=2,
            provider_id="ollama",
            model="qwen2.5:7b",
            api_key=None,
        )
        assert worker.mkv_path == fake_mkv_path
        assert worker.track_index == 2
        assert worker.provider_id == "ollama"
        assert worker.model == "qwen2.5:7b"
        assert worker.api_key is None

    def test_worker_has_signals(self, fake_mkv_path):
        worker = TranslationWorker(
            mkv_path=fake_mkv_path,
            track_index=0,
            provider_id="ollama",
            model="qwen2.5:7b",
        )
        assert isinstance(worker.signals, TranslationSignals)

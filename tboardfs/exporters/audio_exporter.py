"""Audio data exporter for TensorBoard events."""

from pathlib import Path
from typing import Any
import io
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util
from loguru import logger

from .base_exporter import BaseExporter

# Import pydub for audio format conversion
try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None


class AudioExporter(BaseExporter):
    """Export audio data from TensorBoard events."""

    def __init__(self, output_path: Path, digits: int = 6, audio_format: str = "wav"):
        """Initialize audio exporter.

        Args:
            output_path: Base output directory
            digits: Number of digits for step padding
            audio_format: Output audio format (wav, mp3, ogg)
        """
        super().__init__(output_path, digits)
        self.audio_format = audio_format

    def save_data(self, event: event_pb2.Event, value: Any, **kwargs: Any) -> None:
        """Save audio data from a TensorBoard event.

        Args:
            event: TensorBoard event containing the audio data
            value: Summary value containing audio information
        """
        audio_format = kwargs.get("audio_format", self.audio_format)

        tag = value.tag
        safe_tag = self._sanitize_tag(tag)
        tag_dir = self.output_path / "audio" / safe_tag
        self._ensure_directory_exists(tag_dir)

        padded_step = self._format_step(event.step)
        audio_file = tag_dir / f"{padded_step}.{audio_format}"

        try:
            # Extract raw audio data
            raw_audio_data = self._extract_audio_data(value)

            if raw_audio_data is None:
                logger.warning(f"No audio data extracted for tag {tag}")
                return

            # Convert audio format if needed and pydub is available
            if PYDUB_AVAILABLE and audio_format != "wav":
                self._convert_and_save_audio(
                    raw_audio_data, audio_file, audio_format, tag
                )
            else:
                # Save raw audio data (no conversion)
                if not PYDUB_AVAILABLE and audio_format != "wav":
                    logger.warning(
                        f"pydub not available - saving {tag} as raw audio with .{audio_format} extension"
                    )

                self._save_raw_audio(raw_audio_data, audio_file)

        except Exception as e:
            logger.error(f"Failed to save audio for tag {tag}: {e}")
            # Remove the empty file if it was created
            if audio_file.exists():
                audio_file.unlink()

    def _extract_audio_data(self, value: Any) -> bytes | None:
        """Extract raw audio data from summary value."""
        if value.HasField("audio"):
            # Legacy audio format
            return value.audio.encoded_audio_string
        elif value.HasField("tensor") and value.tensor.dtype == 7:  # DT_STRING
            # Modern tensor-based audio format
            try:
                arr = tensor_util.make_ndarray(value.tensor)
                if arr.size > 0:
                    # For string tensors, extract the bytes directly
                    audio_data = arr.item() if arr.ndim == 0 else arr[0]
                    if isinstance(audio_data, bytes):
                        return audio_data
                    else:
                        # If it's a string, encode it to bytes
                        return str(audio_data).encode("utf-8")
                else:
                    logger.warning(f"Empty audio tensor for tag {value.tag}")
            except Exception as e:
                logger.warning(
                    f"Could not decode audio tensor for tag '{value.tag}': {e}"
                )

        return None

    def _convert_and_save_audio(
        self, raw_audio_data: bytes, audio_file: Path, audio_format: str, tag: str
    ) -> None:
        """Convert and save audio using pydub."""
        try:
            # Load the raw audio data (typically WAV format) into AudioSegment
            audio_segment = AudioSegment.from_file(io.BytesIO(raw_audio_data))

            # Convert to requested format
            output_buffer = io.BytesIO()
            audio_segment.export(output_buffer, format=audio_format)
            converted_data = output_buffer.getvalue()

            # Save the converted audio
            with audio_file.open("wb") as f:
                f.write(converted_data)

            logger.debug(f"Converted audio for tag {tag} from WAV to {audio_format}")

        except Exception as conversion_error:
            logger.warning(f"Audio conversion failed for tag {tag}: {conversion_error}")
            logger.info(f"Falling back to saving raw audio data for tag {tag}")
            # Fall back to saving raw data
            self._save_raw_audio(raw_audio_data, audio_file)

    def _save_raw_audio(self, raw_audio_data: bytes, audio_file: Path) -> None:
        """Save raw audio data to file."""
        with audio_file.open("wb") as f:
            f.write(raw_audio_data)

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "tensorboard==2.21.0", "imageio-ffmpeg==0.6.0",
#   "python-ffmpeg==2.0.12", "pillow==12.1.1",
# ]
# ///
"""Rewrap caller-encoded media with independent, isolated event writers.

The bounded envelopes and file hashes allow exact TFRecord reassembly.
The extended matrix reuses original fixture payloads without storing extra media.
"""

import argparse
from fnmatch import fnmatchcase
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from io import BytesIO
from itertools import pairwise
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from time import time
import wave

from google.protobuf.message import Message


CORPUS = Path(__file__).resolve().parents[1] / "tests/data/producer-cases"
ANIMATIONS = ("animated-gif", "animated-apng", "animated-webp")
STILLS = ("png", "jpeg", "webp", "bmp", "tiff", "avif")
VIDEOS = ("mp4", "mov", "avi", "webm", "mkv", "ogg")
AUDIO = (
    ("wav", "audio/wav"),
    ("mp3", "audio/mpeg"),
    ("flac", "audio/flac"),
    ("ogg", "audio/ogg"),
    ("webm", "audio/webm"),
)
EXTENDED_SOURCES = (
    (
        "media/media",
        ("gif", "channels", "unlabelled", "pcm8", "pcm16", "pcm24", "pcm32"),
    ),
    ("rasters/rasters", ("png-*", "jpeg-*")),
    ("rasters/moving-scenes", ("*",)),
    ("audio-variants/audio-variants", ("*",)),
    ("av-clips/av-clips", ("*",)),
    ("geotiff/geotiff", ("*",)),
    ("heic/heic", ("*",)),
    ("wave64/wave64", ("*",)),
    ("native-media/native-media", ("*",)),
    (
        "media-containers/media-containers",
        (
            "qoi",
            "ico-bmp",
            "ico-png",
            "hdr",
            "svg",
            "caf",
            "ogg",
            "webm",
            "program-v1",
            "program-v2",
            "transport",
            "windows-media",
            "elementary",
        ),
    ),
)


class _MediaCorpus:
    """Stage shared codec inputs and dispatch native writers in isolated environments."""

    work: Path

    def __init__(self, work: Path) -> None:
        self.work = work

    def _stage_image(self, tag: str, blob: bytes, dimensions: tuple[int, int]) -> None:
        """Decode the original red still or timed red/blue animation before reuse."""
        with import_module("PIL.Image").open(BytesIO(blob)) as pixels:
            if pixels.size != dimensions or pixels.size != (16, 8):
                raise ValueError(f"{tag}: source dimensions do not match")
            colors, durations = [], []
            for i in range(getattr(pixels, "n_frames", 1)):
                pixels.seek(i)
                colors.append(pixels.convert("RGB").getpixel((8, 4)))
                durations.append(pixels.info.get("duration"))
        if tag in ANIMATIONS:
            if colors != [(255, 0, 0), (0, 0, 255)] or durations != [120, 240]:
                raise ValueError(f"{tag}: expected timed red and blue frames")
        elif len(colors) != 1 or any(
            abs(a - b) > 8 for a, b in zip(colors[0], (255, 0, 0), strict=True)
        ):
            raise ValueError(f"{tag}: expected a red still")
        (self.work / tag).write_bytes(blob)

    def stage_sources(self, source: Path) -> None:
        """Select exact input files: additional runs can coexist in their directories."""
        loader = import_module(
            "tensorboard.backend.event_processing.event_file_loader"
        ).LegacyEventFileLoader
        for run in ("media", "rasters"):
            events = source / run / f"events.out.tfevents.{run}"
            values = (
                v for event in loader(str(events)).Load() for v in event.summary.value
            )
            images = STILLS if run == "media" else ANIMATIONS
            for value in values:
                if value.tag in images:
                    image = value.image
                    self._stage_image(
                        value.tag,
                        image.encoded_image_string,
                        (image.width, image.height),
                    )
                elif run == "media" and value.tag in VIDEOS:
                    (self.work / f"video-{value.tag}").write_bytes(
                        value.tensor.tensor_content
                    )
                elif run == "media" and value.tag == "riff-midi":
                    (self.work / value.tag).write_bytes(
                        value.audio.encoded_audio_string
                    )

    def stage_extended(self, source: Path) -> bytes:
        """Stage a single Summary from original events, without modifying media fields."""
        loader = import_module(
            "tensorboard.backend.event_processing.event_file_loader"
        ).LegacyEventFileLoader
        summary = import_module("tensorboard.compat.proto.summary_pb2").Summary()
        for key, patterns in EXTENDED_SOURCES:
            run, name = key.split("/")
            path = source / run / f"events.out.tfevents.{name}"
            candidates = [
                v for event in loader(str(path)).Load() for v in event.summary.value
            ]
            for pattern in patterns:
                if not any(fnmatchcase(v.tag, pattern) for v in candidates):
                    raise ValueError(f"{path}: no values for {pattern}")
            for value in candidates:
                if any(fnmatchcase(value.tag, pattern) for pattern in patterns):
                    staged = summary.value.add()
                    staged.CopyFrom(value)
                    staged.tag = f"{run}-{value.tag}"
        payload = bytes(summary.SerializeToString())
        if len(payload) > 1024**2:
            raise ValueError("extended Summary exceeds 1 MiB")
        (self.work / "extended-summary.pb").write_bytes(payload)
        return payload

    def stage_tones(self) -> None:
        """Decode the interior 0.2 s of every lossy clip and verify 88 cycles of A4."""
        rate = 16000
        samples = [
            round(12000 * math.sin(math.tau * 440 * i / rate)) for i in range(rate // 4)
        ]
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        with wave.open(str(self.work / "tone.wav"), "wb") as stream:
            stream.setparams((1, 2, rate, len(samples), "NONE", "not compressed"))
            stream.writeframes(pcm)
        pipeline = import_module("ffmpeg").FFmpeg
        ffmpeg = import_module("imageio_ffmpeg").get_ffmpeg_exe()
        for ext, codec in (
            ("mp3", "libmp3lame"),
            ("flac", "flac"),
            ("ogg", "libvorbis"),
            ("webm", "libopus"),
        ):
            path = self.work / f"tone.{ext}"
            pipeline(executable=ffmpeg).option("loglevel", "error").input(
                self.work / "tone.wav"
            ).output(path, acodec=codec).execute()
            decoded = (
                pipeline(executable=ffmpeg)
                .option("loglevel", "error")
                .input(path)
                .output("pipe:1", ar=rate, ac=1, f="s16le")
                .execute()
            )
            samples = [sample for (sample,) in struct.iter_unpack("<h", decoded)]
            window = samples[400:3600]
            crossings = sum(a <= 0 < b for a, b in pairwise(window))
            if len(window) != 3200 or abs(crossings - 88) > 1 or max(window) < 8000:
                raise ValueError(f"{codec}: did not recover the 440-Hz tone")

    def _build_summary(self, schema: str) -> Message:
        """Construct every field in the isolated writer's own protobuf schema."""
        summary = import_module(f"{schema}.summary_pb2").Summary()
        staged = self.work / "extended-summary.pb"
        if staged.is_file():
            summary.ParseFromString(staged.read_bytes())
            return summary
        types = import_module(f"{schema}.types_pb2")
        for tag in (*STILLS, *ANIMATIONS):
            data = (self.work / tag).read_bytes()
            img = summary.value.add(tag=f"legacy-{tag}").image
            img.width, img.height, img.colorspace, img.encoded_image_string = (
                16,
                8,
                3,
                data,
            )
            value = summary.value.add(tag=f"plugin-{tag}")
            value.metadata.plugin_data.plugin_name = "images"
            value.tensor.dtype = types.DT_STRING
            value.tensor.tensor_shape.dim.add(size=3)
            value.tensor.string_val.extend((b"16", b"8", data))
        for ext, mime in AUDIO:
            data = (self.work / f"tone.{ext}").read_bytes()
            audio = summary.value.add(tag=f"legacy-tone-{ext}").audio
            audio.content_type = mime
            audio.sample_rate, audio.num_channels, audio.length_frames = 16000, 1, 4000
            audio.encoded_audio_string = data
            value = summary.value.add(tag=f"plugin-tone-{ext}")
            value.metadata.plugin_data.plugin_name = "audio"
            value.tensor.dtype = types.DT_STRING
            value.tensor.tensor_shape.dim.add(size=1)
            value.tensor.tensor_shape.dim.add(size=2)
            value.tensor.string_val.extend((data, b"Quarter-second A4"))
        midi = summary.value.add(tag="riff-midi").audio
        midi.content_type = "audio/mid"
        midi.encoded_audio_string = (self.work / "riff-midi").read_bytes()
        for ext in VIDEOS:
            data = (self.work / f"video-{ext}").read_bytes()
            for layout in ("scalar", "vector", "bytes"):
                value = summary.value.add(tag=f"{layout}-{ext}")
                value.metadata.plugin_data.plugin_name = (
                    "videos" if layout == "vector" else "video"
                )
                tensor = value.tensor
                if layout == "bytes":
                    tensor.dtype = types.DT_UINT8
                    tensor.tensor_shape.dim.add(size=len(data))
                    tensor.tensor_content = data
                else:
                    tensor.dtype = types.DT_STRING
                    tensor.tensor_shape.SetInParent()
                    if layout == "vector":
                        tensor.tensor_shape.dim.add(size=1)
                    tensor.string_val.append(data)
        return summary

    @staticmethod
    def _write_native(writer: str, dest: Path, event: Message, summary: Message) -> str:
        """Call the public event/raw-summary API. Lightning delegates to its experiment."""
        if writer == "tf1":
            tf = import_module("tensorflow")
            with tf.Graph().as_default():
                native = tf.compat.v1.summary.FileWriter(str(dest))
                try:
                    write = native.add_event
                    write(event)
                finally:
                    native.close()
        elif writer == "tf2":
            tf = import_module("tensorflow")
            native = tf.summary.create_file_writer(str(dest))
            with native.as_default():
                write = tf.summary.experimental.write_raw_pb
                if not write(summary.SerializeToString(), step=7):
                    raise ValueError("TF2 did not write the caller-supplied Summary")
            native.close()
        elif writer == "tb221":
            native = import_module(
                "tensorboard.summary.writer.event_file_writer"
            ).EventFileWriter(str(dest))
            try:
                write = native.add_event
                write(event)
            finally:
                native.close()
        elif writer == "lightning":
            logger = import_module("lightning.pytorch.loggers").TensorBoardLogger(
                save_dir=str(dest), name="", version=""
            )
            experiment = logger.experiment
            try:
                write = experiment.file_writer.add_summary
                write(summary, global_step=7, walltime=1.0)
            finally:
                experiment.close()
            owner = type(logger)
            return (
                f"{owner.__module__}.{owner.__qualname__} -> "
                f"{write.__module__}.{write.__qualname__}"
            )
        else:
            module = (
                "tensorboardX.writer"
                if writer == "tbx"
                else "torch.utils.tensorboard.writer"
            )
            native = import_module(module).FileWriter(str(dest))
            try:
                write = native.add_summary
                write(summary, global_step=7, walltime=1.0)
            finally:
                native.close()
        return f"{write.__module__}.{write.__qualname__}"

    def emit(self, dest: Path, writer: str) -> None:
        """Load only modules installed for this backend, and serialize a native event."""
        if writer in {"tf1", "tf2"}:
            schema, event_schema = "tensorflow.core.framework", "tensorflow.core.util"
        else:
            schema = (
                "tensorboardX.proto" if writer == "tbx" else "tensorboard.compat.proto"
            )
            event_schema = schema
        dest.mkdir()
        events = import_module(f"{event_schema}.event_pb2")
        summary = self._build_summary(schema)
        event = events.Event(step=7, wall_time=1, summary=summary)
        path = dest / f"events.out.tfevents.encoded-{writer}"
        extended = (self.work / "extended-summary.pb").is_file()
        if writer in {"tb115", "tb200"}:
            with path.open("xb") as handle:
                native = import_module(
                    "tensorboard.summary.writer.record_writer"
                ).RecordWriter(handle)
                header = events.Event(file_version="brain.Event:2")
                if extended:
                    header.wall_time = time()
                for record in (header, event):
                    native.write(record.SerializeToString())
                api = f"{native.write.__module__}.{native.write.__qualname__}"
        else:
            api = self._write_native(writer, dest, event, summary)
            (source,) = dest.glob("events.out.tfevents.*")
            source.rename(path)
        if extended:
            distributions = {
                "tb115": ("tensorboard",),
                "tb200": ("tensorboard",),
                "tf1": ("tensorflow-cpu",),
                "tf2": ("tensorflow-cpu",),
                "tb221": ("tensorboard",),
                "tbx": ("tensorboardX",),
                "torch": ("torch", "tensorboard"),
                "lightning": ("lightning", "torch", "tensorboard"),
            }
            (dest / "provenance.json").write_text(
                json.dumps(
                    {
                        "api": api,
                        "versions": {
                            name: version(name) for name in distributions[writer]
                        },
                    }
                )
            )

    def _run_writer(self, writer: str, dependencies: list[str], dest: Path) -> Path:
        """Run this module's adapter in a pinned environment, with torch only from its CPU index."""
        python = "3.11" if writer in {"tb115", "tb200"} else "3.12"
        metadata = [
            "#!/usr/bin/env -S uv run --script",
            "# /// script",
            f'# requires-python = "=={python}.*"',
            f"# dependencies = {json.dumps(dependencies)}",
        ]
        if writer in {"torch", "lightning"}:
            metadata.extend(
                (
                    "# [tool.uv.sources]",
                    '# torch = { index = "pytorch-cpu" }',
                    "# [[tool.uv.index]]",
                    '# name = "pytorch-cpu"',
                    '# url = "https://download.pytorch.org/whl/cpu"',
                    "# explicit = true",
                )
            )
        adapter = self.work / f"adapter-{writer}.py"
        adapter.write_text(
            "\n".join(
                (
                    *metadata,
                    "# ///",
                    "import sys",
                    "from pathlib import Path",
                    "sys.path.insert(0, sys.argv[1])",
                    "from generate_encoded_media_cases import _MediaCorpus",
                    "_MediaCorpus(Path(sys.argv[2])).emit(Path(sys.argv[3]), sys.argv[4])",
                    "",
                )
            )
        )
        uv = shutil.which("uv")
        if uv is None:
            raise FileNotFoundError("uv is required for isolated native writers")
        command = [
            uv,
            "run",
            "--python",
            python,
            "--script",
            str(adapter),
            str(Path(__file__).resolve().parent),
            str(self.work),
            str(dest),
            writer,
        ]
        subprocess.run(  # noqa: S603
            command,
            check=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "TF_NUM_INTEROP_THREADS": "1",
                "TF_NUM_INTRAOP_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "",
            },
        )
        (event_file,) = dest.glob("events.out.tfevents.*")
        return event_file

    def generate(self, output: Path, *, extended: bytes | None = None) -> None:
        """Save native envelopes around one independently reconstructible Summary."""
        historical = ["numpy==1.23.5", "protobuf==3.20.3", "setuptools==80.9.0"]
        tf = ["tensorflow-cpu==2.21.0", "tensorboard==2.21.0"]
        packages = {
            "tb115": ["tensorboard==1.15.0", *historical],
            "tb200": ["tensorboard==2.0.0", *historical],
            "tf1": tf,
            "tf2": tf,
            "tb221": ["tensorboard==2.21.0"],
            "tbx": ["tensorboardX==2.6.5"],
            "torch": ["torch==2.12.0+cpu", "tensorboard==2.21.0"],
            "lightning": [
                "lightning==2.6.6",
                "torch==2.12.0+cpu",
                "tensorboard==2.21.0",
            ],
        }
        if extended is not None:
            shared = extended
        else:
            loader = import_module(
                "tensorboard.backend.event_processing.event_file_loader"
            ).LegacyEventFileLoader
            canonical = self._run_writer(
                "tb115", packages.pop("tb115"), output / "encoded-tb115"
            )
            (reference,) = (
                e for e in loader(str(canonical)).Load() if e.HasField("summary")
            )
            shared = bytes(reference.summary.SerializeToString())
        envelopes: dict[str, dict[str, object]] = {}
        for writer, dependencies in packages.items():
            path = self._run_writer(
                writer, dependencies, self.work / f"encoded-{writer}"
            )
            actual = path.read_bytes()
            before, after = actual.split(shared)
            if len(before) + len(after) > 4096:
                raise ValueError(f"{writer}: native envelope exceeds 4 KiB")
            envelope: dict[str, object] = {
                "prefix": before.hex(),
                "suffix": after.hex(),
                "sha256": sha256(actual).hexdigest(),
            }
            if extended is not None:
                envelope.update(
                    json.loads(path.with_name("provenance.json").read_text())
                )
            envelopes[writer] = envelope
        if extended is not None:
            (output / "encoded-extended.json").write_text(
                json.dumps(
                    {
                        "summary_sha256": sha256(shared).hexdigest(),
                        "writers": envelopes,
                    },
                    indent=2,
                )
                + "\n"
            )
            return
        (output / "encoded-envelopes.json").write_text(
            json.dumps(envelopes, indent=2) + "\n"
        )


def main() -> None:
    """Generate the selected matrix in a new or empty directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="write only the cross-writer envelopes for the extended media sources",
    )
    parser.add_argument(
        "-s",
        "--source",
        type=Path,
        default=CORPUS,
        help="producer-cases directory containing the original media runs",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("output directory must be new or empty")
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        corpus = _MediaCorpus(work)
        if args.extended:
            corpus.generate(
                args.output.resolve(), extended=corpus.stage_extended(args.source)
            )
            return
        corpus.stage_sources(args.source)
        corpus.stage_tones()
        hashes = {
            ext: sha256((work / f"tone.{ext}").read_bytes()).hexdigest()
            for ext, _mime in AUDIO
        }
        (args.output / "encoded-tone-sha256.json").write_text(
            json.dumps(hashes, indent=2) + "\n"
        )
        corpus.generate(args.output.resolve())


if __name__ == "__main__":
    main()

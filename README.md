# tboardfs

<p>
  <a href="https://github.com/dvolgyes/tboardfs/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/dvolgyes/tboardfs/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://coveralls.io/github/dvolgyes/tboardfs?branch=main"><img alt="Coverage Status" src="https://coveralls.io/repos/github/dvolgyes/tboardfs/badge.svg?branch=main" /></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://pypi.org/project/tboardfs/"><img alt="Version: 0.3.0" src="https://img.shields.io/badge/version-0.3.0-orange.svg" /></a>
  <a href="https://pypi.org/project/tboardfs/"><img alt="Status: Alpha" src="https://img.shields.io/badge/status-alpha-yellow.svg" /></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg" /></a>
  <a href="https://www.python.org/"><img alt="Python: >=3.12" src="https://img.shields.io/badge/python-%3E=3.12-blue.svg" /></a>
</p>

Read TensorBoard event logs as ordinary files.

tboardfs shows TensorBoard event files (`events.out.tfevents*`) as a read-only directory tree. Browse the typed view to
read scalars as tables, images as images, and tensors as NumPy arrays. Use the separate `raw` view to read the original
TFRecord frames, including records with no typed representation. Neither view needs TensorFlow or TensorBoard.

```console
$ tboardfs runs mnt
$ ls mnt
experiment-1  experiment-2
$ ls mnt/experiment-1/scalars/train
accuracy.json  accuracy.npz  accuracy.tsv  loss.json  loss.npz  loss.tsv
$ head -3 mnt/experiment-1/scalars/train/loss.tsv
epoch	step	wall_time	relative_time	value
0.0	1	1700000001.0	0.0	1.5
1.0	2	1700000002.0	1.0	0.75
```

tboardfs reads the logs of TensorFlow 1 and 2, Keras, PyTorch (`torch.utils.tensorboard`), PyTorch Lightning,
tensorboardX, and other writers of the same event format.

There are two ways to use it:

- `tboardfs` mounts a log directory with FUSE. The mount follows a training run while it writes new events.
- `tboardfs-file` lists and copies the same virtual files without FUSE, for example in scripts, in containers, or on
  machines where you cannot mount.

## Installation

tboardfs needs Python 3.12 or later, and these system libraries:

- Mounting requires FUSE, permission to use `/dev/fuse` on Linux, and a working `fusermount` or `fusermount3`. Use
  `tboardfs-file` where mount privileges are unavailable.
- libmagic, which finds the file type of images, audio and other stored files.

On Debian or Ubuntu, for example:

```bash
sudo apt install fuse3 libmagic1
```

Install tboardfs from PyPI:

```bash
pip install tboardfs
```

Or run it without an installation:

```bash
uvx tboardfs SOURCE MOUNTPOINT
uvx --from tboardfs tboardfs-file list SOURCE
```

## Filesystem Mode

```bash
tboardfs [OPTIONS] SOURCE MOUNTPOINT
```

SOURCE is a log directory or a run directory. MOUNTPOINT is an existing directory. The mountpoint must not be inside
SOURCE, and SOURCE must not be inside the mountpoint.

```bash
mkdir -p mnt
tboardfs runs/experiment-1 mnt
ls mnt
cat mnt/scalars/loss.json
fusermount -u mnt    # unmount; with libfuse 3, use fusermount3 -u
```

tboardfs runs in the background by default. With `--foreground`, it stays in the terminal until you stop it with Ctrl+C.

| Option | Default | Meaning |
| -- | -- | -- |
| `--scalar-format LIST` | `json,tsv,npz` | Comma-separated table formats for scalars, histograms and distributions: `json`, `tsv`, `csv`, `npz`. |
| `--step-digits N` | `6` | Zero-padding width of step file names, from 1 to 20. |
| `--refresh-age-seconds S` | `150` | Minimum interval between local searches for new files and directories. Use `0` for immediate discovery or `inf` to freeze each directory after first access. |
| `--cache-size-mb N` | `1024` | Memory for rendered files. `0` turns the cache off. |
| `--root-owner` / `--user-owner` | `--root-owner` | Owner of all files and directories: root, or the user who mounts. |
| `--foreground` / `--background` | `--background` | Stay in the terminal, or detach. |
| `--logfile FILE` | none | Write the log to FILE. |
| `--loglevel LEVEL` | `INFO` | `trace`, `debug`, `info`, `success`, `warning`, `error`, or `critical`, in any case. |

In the background, the log goes only to `--logfile`. Without `--logfile`, you do not see the warnings about damaged
data; use `--logfile` or `--foreground` when you need them.

### Live Training Runs

Listing a directory discovers its children without indexing every descendant. Opening a path checks the known event
files in the active run for changes and deletions, even within the discovery interval. New event files, subdirectories
and sidecars become visible on local discovery; an idle run does not need a background scan. `inf` disables subsequent
checks in a directory after its first access.

Directory timestamps reflect discovered content, not files in unvisited subdirectories.

Writers should append complete frames and replace the file when rewriting history. tboardfs resumes at the end of the
last complete frame and checks that frame before reusing its index. A changed file identity, truncation or invalid
checkpoint causes a fresh index. Identity uses device and inode plus birth time where available; Unix `ctime` is not
creation time. This scheme does not detect all changes to earlier records in a growing file.

Each open file keeps the content it had when it was opened, so a read never mixes two versions. Open the file again to
see new data. If a raw frame, image or graph changes before its first read, tboardfs validates the indexed source rather
than returning unrelated replacement bytes.

### Memory and Caching

Rendered files are kept in memory, so `ls -l` followed by `cat` renders a file once. When the cache goes over
`--cache-size-mb`, the least recently used files are dropped until half the limit is left. New events in a run make only
the files of that run render again. The limit applies to rendered bytes, not to indexes, directory entries, or bytes
pinned by open handles.

### Permissions and Memory Maps

The filesystem is read-only. Files have mode `0444` and directories `0555`. They belong to root (uid and gid 0), or to
the user who mounts with `--user-owner`.

Mounted files use direct I/O. Plain reads and private memory maps work, for example `np.load(path, mmap_mode="c")`.
Shared memory maps (`mmap` with `MAP_SHARED`, `np.load(path, mmap_mode="r")`) fail with `ENODEV`.

## Command Mode

```bash
tboardfs-file [--logfile FILE] [--loglevel LEVEL] COMMAND [OPTIONS] SOURCE ...
```

SOURCE is one event file, a run directory, or a log directory.

- For one event file, the tabs are at the root, for example `/scalars` and `/images`. The files beside the event file
  (projector, profile, `hparams.yaml`) are not included.
- For a directory, the tree is the same as a mount of that directory, with those files.

| Command | Purpose |
| -- | -- |
| `list SOURCE [PREFIX]` | Print the virtual file paths below PREFIX, one per line. |
| `get SOURCE VPATH -o OUTPUT` | Write one virtual file to OUTPUT. `-o -` writes it to stdout. |
| `copy-all SOURCE OUTDIR` | Copy the selected view into OUTDIR. |

All three commands accept `--scalar-format` and `--step-digits`, with the same meaning as in filesystem mode. `list` and
`copy-all` default to typed files. Use `--view raw` for complete frames or `--view both` for both views. `list` with an
explicit raw prefix and `get` with an explicit raw path can select a frame directly.

```bash
tboardfs-file list events.out.tfevents.123
tboardfs-file list runs /experiment-1/meshes
tboardfs-file get events.out.tfevents.123 /scalars/loss.json -o -
tboardfs-file get events.out.tfevents.123 images/sample/000001.png -o sample.png
tboardfs-file get events.out.tfevents.123 scalars/loss.tsv -o exported/
tboardfs-file copy-all runs exported-runs --skip-existing
tboardfs-file copy-all --view raw events.out.tfevents.123 frames
cat frames/raw/events.out.tfevents.123/*.tfrecord > reconstructed.events
```

- A virtual path can start with or without `/`.
- When `-o` is an existing directory, or a missing path that ends in `/`, `get` writes the file under its own name in
  that directory.
- `get` does not overwrite a file unless you give `--force`.
- `copy-all` copies the files in sorted virtual-path order. When it finds a file that already exists, it stops, and
  tells which files it copied and which path stopped it. `-f/--force` overwrites existing files. `-s/--skip-existing`
  keeps existing files, writes a warning for each, and copies the rest. You cannot use both.
- Both commands refuse to write through destination symlinks. They publish each complete file atomically, not the entire
  export as one transaction.

Only file data and file lists go to stdout, so you can pipe them. Messages, warnings and errors go to stderr.

## Directory Layout

A log directory has one directory for each run. A run is a directory that holds event files. The tabs of a run are
directories in the run directory:

```text
MOUNTPOINT/
├── experiment-1/                  a run
│   ├── scalars/train/loss.json    the tag train/loss
│   ├── images/sample/000010.png   the tag sample at step 10
│   ├── ...                        the other tabs
│   └── eval/                      a run inside the run experiment-1
│       └── scalars/...
└── experiment-2/
    └── ...
```

When SOURCE is a run, its 16 typed tabs and its `raw` directory appear at the mount root. A typed tab without data is
empty. Multiple event files in a run contribute to the same typed tabs but retain separate directories under `raw`.

### Tags and Steps

- A tag is a path below its tab: the tag `train/loss` is `scalars/train/loss.json`.
- Each empty component is `~empty`, including a wholly empty tag. Thus `a//b` becomes `a/~empty/b`, distinct from `a/b`.
- Literal `%` and `~` are `%25` and `%7E`. The reserved components `.` and `..`, backslashes, control characters and the
  reserved name-map filename use percent escapes.
- Data with one value for each step has one file per step. The file name is the step, zero-padded to `--step-digits`
  digits, such as `000010.png`.
- An event without a step field is at step 0, as in TensorBoard.

### Name Conflicts

Ordinary names keep their spelling. This naming scheme replaces the earlier lossy scheme; scripts that addressed unusual
tags or conflicting names must use the new paths. Decode percent escapes only after splitting the virtual path into
components.

- When a tag directory has the name of a file, such as the tag `loss.json/x` beside `scalars/loss.json`, the directory
  becomes `loss.json~2/`, and the file keeps its name.
- When a run inside another run has the name of a tab, such as the run `exp/scalars`, it becomes `exp/scalars~2/`.
  `exp/scalars/` keeps only the scalars of `exp`.
- A literal directory name `scalars~2` becomes `scalars%7E2`, distinct from the generated `scalars~2`.
- Distinct sidecars at the same virtual path receive deterministic suffixes rather than hiding one file.
- A component longer than the host limit is shortened to at most 255 UTF-8 bytes. In an affected directory,
  `.tboardfs-names.json` maps shortened spellings to their original names.

### Symbolic Links

Treat SOURCE and the files it links to as trusted input for the account running tboardfs. Links to event files, run
directories, and sidecars may leave SOURCE or cross onto a remote filesystem. Traversal skips ancestor loops and links
back into the active mountpoint. Do not give an untrusted writer control of a source directory when the mounting account
can read sensitive files. Sibling links to one directory use the first name in sorted order. The same source can appear
under different branches because tboardfs does not search idle branches to eliminate aliases.

## Exported Data

| Tab | Files | Content |
| -- | -- | -- |
| `scalars` | `<tag>.json` `.tsv` `.csv` `.npz` | One table per tag, all steps |
| `histograms` | `<tag>/<step>.json` `.tsv` `.csv` `.npz` | The buckets of one histogram |
| `distributions` | `<tag>/<step>.json` `.tsv` `.csv` `.npz` | The 9 percentiles of the Distributions tab |
| `pr_curves` | `<tag>/<step>.json` `.tsv` `.npy` | One row per threshold |
| `images` | `<tag>/<step>.png` (or `.jpg`, ...) | The stored image |
| `audio` | `<tag>/<step>.wav` (or other type) | The stored audio clip |
| `videos` | `<tag>/<step>.gif` (or other type) | Animated GIFs and videos |
| `text` | `<tag>/<step>.txt` (and `.json`) | Text, or a Markdown table |
| `meshes` | `<tag>/<step>.json` `.npz` `.obj` `.ply` | Vertices, faces and colors |
| `tensors` | `<tag>/<step>.npy` `.json` | Tensor values |
| `hparams` | `hparams.json`, `hparams.yaml` | Hyperparameters and metrics |
| `custom_scalars` | `layout.json` | The Custom Scalars layout |
| `graphs` | `graph.pb`, `<tag>/<step>.*` | Model graphs and run metadata |
| `projector` | files of the run directory | Embedding projector files |
| `profile` | files of the run directory | Profiler files |
| `plugins` | `<plugin>/<tag>/<step>.json` | Data of other plugins |
| `raw` | `<event-file>/<20-digit-byte-offset>.tfrecord` | One complete frame: header, payload and footer, in source order |

`--scalar-format` selects the table formats of scalars, histograms and distributions. The default is `json,tsv,npz`;
`csv` is also available.

### Scalars

Each scalar tag is one table, sorted by integer step without loss of precision beyond $2^{53}$. Equal-step rows keep
their producer order.

| Column | Meaning |
| -- | -- |
| `epoch` | The `epoch` scalar at that step (see below) |
| `step` | The global step |
| `wall_time` | Seconds since the Unix epoch |
| `relative_time` | Seconds since the first summary of the event file |
| `value` | The scalar value |

- The `epoch` column is the value of the scalar tag `epoch` at the same step, as PyTorch Lightning writes it. A step
  without an `epoch` value has the last epoch written before it. Without an earlier `epoch` value, the cell is missing
  (`null` in JSON, `nan` in TSV and CSV).
- In the `.npz` file, each column is an array with the same name. `value` keeps the stored type: float32 for TensorFlow
  1 and PyTorch scalars, or the dtype of a tensor scalar, such as float64 or int64.
- A supported one-value numeric tensor without a more specific plugin can be a scalar.

### Histograms and Distributions

A histogram table has one row per bucket, with the columns `step`, `epoch`, `wall_time`, `relative_time`, `bucket_left`,
`bucket_right` and `count`. TensorFlow 1 and TensorFlow 2 histograms give the same table. As in TensorBoard, a
TensorFlow 1 histogram runs from its minimum to its maximum, without the empty buckets at its two ends. A histogram of
no values has no step file. Two observations with the same tag, step and timestamp retain separate bucket groups and
separate sets of percentiles in the per-step tables.

A distribution table has the values that the TensorBoard Distributions tab shows, with the columns `step`, `epoch`,
`wall_time`, `relative_time`, `percentile` and `value`. The 9 percentiles are 0, 6.68, 15.87, 30.85, 50, 69.15, 84.13,
93.32 and 100: the minimum, the maximum, and the positions of the mean and ±1, 2 and 3 standard deviations of a normal
distribution. Each value interpolates linearly inside its histogram bucket.

### PR Curves

The `.json` and `.tsv` tables have one row per threshold, with the columns `threshold`, `true_positives`,
`false_positives`, `true_negatives`, `false_negatives`, `precision` and `recall`. The thresholds are evenly spaced in
[0, 1], as in TensorBoard. The `.npy` file is the stored `[6, N]` tensor. A tensor of another shape has the columns
`value_0`, `value_1`, and so on.

### Images, Audio and Videos

Media files contain the original bytes; tboardfs does not transcode them.

- A multi-sample image or audio summary has one file per sample, such as `000001_0.png` and `000001_1.png`.
- Audio tensors can hold a vector of clips or `[clip, label]` rows; the labels remain in the raw event.
- Both single-frame and animated GIF images go under `videos/`.
- TensorBoardX and PyTorch `add_video` write GIF image summaries rather than native video containers.
- A custom `video` plugin can instead store a native video container.

Repeated values with the same tag retain their own payloads even when a writer omits repeated plugin metadata.

### Text

A text summary with one string is a `.txt` file with that text. A summary with many strings is a Markdown table in the
`.txt` file, as TensorBoard shows it: a vector is one column, and a tensor with more than 2 dimensions shows its first
2-D slice (tboardfs logs a warning). A `.json` file beside the table keeps the shape and the exact strings.

### Meshes

The path of a mesh is its tag without the `_VERTEX`, `_FACE` or `_COLOR` suffix, so `train/cloud_VERTEX` is under
`meshes/train/cloud`. A tag without that suffix uses the mesh name, which the `.json` file keeps. TensorBoard groups
meshes by the mesh name only, and thus mixes `train/cloud` and `val/cloud`; tboardfs keeps them apart.

- `.npz` holds the arrays `vertices`, `faces` and `colors` (when present), with their stored types and shapes.
- `.json` holds the mesh name, the content types, the shape, and the JSON config of the summary.
- `.obj` is a Wavefront OBJ file. A batch of more than one mesh has one object (`o mesh_<i>`) per element. Vertex colors
  follow the coordinates as `r g b` values in [0, 1], which MeshLab and Blender read. A mesh without faces is a point
  cloud.
- `.ply` is a binary PLY file that keeps the coordinates exactly: float32 coordinates as `float`, other numbers as
  `double`, and colors as `uchar` `red green blue`. A point cloud has no `face` element. A batch is one PLY mesh, with
  the faces of a later element offset by the vertices before it, and one `comment mesh_<i> vertices <start> <count>`
  line per element. Faces that are not vertex indices, and colors that are not whole numbers from 0 to 255, are not in
  the PLY file; the `.npz` file keeps them.

### Tensors

A supported tensor has a `.npy` preview and a `.json` preview with its `shape`, `dtype` and `values`.

- Float16/32/64, complex64/128, boolean, and signed and unsigned integers retain their storage precision.
- Quantized integer tensors export their stored integers, not dequantized values.
- Bfloat16 widens exactly to float32.
- Use the raw frame for float8, float4, 2- and 4-bit integers, resources, variants and obsolete reference types; none
  has a NumPy preview.
- String tensors are NumPy text arrays (dtype `U`), so `np.load` reads them without `allow_pickle`.
- A numeric tensor without a plugin name is also a tensor here.
- A single repeated-field value can fill the declared shape up to 1 KiB of expanded data. Shapes requiring more
  expansion do not get an array preview.
- Large binary tensor fields can also have a content-typed file; `.bin` means unknown content type. These previews
  contain field values, not the enclosing record. Use `raw` for byte-exact recovery, including unsupported types and
  plugin metadata.

### HParams

`hparams.json` merges the experiment, the sessions and the metrics of the run.

- The metrics are the declared metric tags, or all scalar tags when the run declares no experiment, as in TensorBoard.
- Each metric has its last value, or `null` without data. Metrics of a sub-run group (for example the Keras `validation`
  run) have no value here; read them from the scalars of that run.
- `sessions` lists every session of the run; `session_start` and `session_end` are the first ones, as in TensorBoard.

The `hparams.yaml` file that PyTorch Lightning writes in the run directory is also in `hparams`.

### Graphs

- `graph.pb` is the `GraphDef` that TensorBoard shows: the newest `graph_def` of the event files, which TensorFlow 1 and
  PyTorch `add_graph` write, or else the graph of the newest TensorFlow 1 meta graph. A run with several `graph_def`
  graphs also has `graph_<i>.pb` for each one, in write order.
- `meta_graph.pb` is the newest TensorFlow 1 `MetaGraphDef`, with `meta_graph_<i>.pb` in the same way.
- A TensorFlow 2 graph summary is under `graphs/<tag>/`:
  - `<step>.json` is the Keras model config that the Keras TensorBoard callback writes.
  - `<step>.run_metadata.pb` is the `RunMetadata` of `tf.summary.trace_export`.
  - `<step>.conceptual.pb` or `<step>.pb` beside it is the `GraphDef` that the Graphs tab draws for it.
- A TensorFlow 1 tagged `RunMetadata` (`FileWriter.add_run_metadata`, PyTorch `add_graph`) is
  `graphs/<tag>/<step>.run_metadata.pb`.
- Beside each `RunMetadata`, `<step>.run_metadata.pbtxt` is the profile text that the Graphs tab shows.

The `.pb` files are binary protobuf messages, which TensorFlow and Netron can read. TensorBoard 2.21 cannot draw a
functional model inside a Sequential model from a Keras 3 config, because it reads the one input of the nested model as
three inputs; tboardfs gives the graph with that one input.

### Projector and Profile

These files come from the run directory, not from the event files, so they are only available when SOURCE is a
directory.

- `projector` has `projector_config.pbtxt` and the other files of a `projector` directory. It also has the files that
  the config names (`tensor_path`, `metadata_path`, sprite `image_path`, `bookmarks_path`), such as the
  `<step>/<tag>/tensors.tsv` files of PyTorch and tensorboardX `add_embedding`, at the paths that the config uses. Only
  relative paths to files of the run are followed, and only the first MiB of the config is read.
- `profile` has the files below a `profile` directory of the run, such as `plugins/profile/...`.

tboardfs copies profiler `.xplane.pb` files without decoding them; it does not extract checkpoint tensors or decode
Debugger V2 dumps.

### Other Plugins

Summaries of a plugin without its own tab are JSON files under `plugins/<plugin>/<tag>/<step>.json`.

## File Formats

### JSON

All `.json` files are strict JSON (RFC 8259), UTF-8, and indented with 2 spaces.

- A table is a list of row objects, with the column names as keys.
- Infinities are the strings `"Infinity"` and `"-Infinity"`.
- In tables, NaN is a missing value and is `null`. In other JSON files, NaN is the string `"NaN"`, as in protobuf JSON.
- A complex number is its `[real, imag]` pair.
- Bytes that are valid UTF-8 are text; other bytes are `{"base64": "..."}`.
- A number reads back as the same value, but its spelling can differ from Python `repr`, for example `1e-5` for `1e-05`.

### TSV and CSV

The first line has the column names. Missing values and NaN are `nan`; infinities are `inf` and `-inf`. A cell that
contains the separator or a quote is quoted.

```python
import pandas as pd

df = pd.read_csv("mnt/run/scalars/loss.tsv", sep="\t")
```

### NumPy

An `.npz` table has one array per column, with the stored types: float32 values stay float32.

```python
import numpy as np

table = np.load("mnt/run/scalars/loss.npz")
steps, values = table["step"], table["value"]
```

## Resumed Runs

A resumed run hides the rows it replaces, as TensorBoard does. When a writer marks a restart at step S (a
`SessionLog.START` event, for example from PyTorch `SummaryWriter(purge_step=S)`), the rows written before it at step S
or later are not shown in the typed tables. Graphs and complete raw frames stay. Restart filtering applies to file
version 2 or later.

## Damaged and Unsupported Data

Frame verification is independent of typed conversion. tboardfs checks both masked CRC32C fields and rejects payload
lengths above 1 GiB before reading the payload. CRC32C detects accidental damage; it does not authenticate the producer.

- **Single event file:** `list`, `get` and `copy-all` fail on a corrupt frame or an incomplete tail. No single-file
  salvage option is available.
- **Mount or directory source:** a bad frame stops that file at the last verified frame. Other source files remain
  readable. An incomplete live tail is retried after the file changes.
- A verified frame whose payload cannot be decoded still appears under `raw`. Concatenating the frames of a stable,
  fully verified source in filename order reproduces the entire event file.
- A value that tboardfs cannot convert has no file, and tboardfs logs a warning once. Examples are an hparams or mesh
  record whose plugin data does not decode, a custom scalar layout that does not decode, and a malformed graph,
  `RunMetadata` or meta graph. The other values of the same event stay.
- When only a conversion fails, the file with the stored bytes stays, for example `<step>.run_metadata.pb` without its
  `.pbtxt`.

## Generate Test Data

The test suite reads `tests/data/producer-cases/` and `test-logs/` without installing the writers.

To regenerate a fixture, run its executable PEP 723 script with a new or empty output directory; `uv` isolates the
generator's dependencies.

```bash
scripts/generate_media_cases.py --output samples/media
scripts/generate_geometry_cases.py --output samples/geometry
```

| Generator in `scripts/` | What the fixtures and tests distinguish |
| -- | -- |
| `generate_tb115_cases.py` | TensorBoard 1.15.0: scalar siblings, log messages, checkpoint/stop events, obsolete histogram bytes, a debugger numeric summary and a Beholder sidecar |
| `generate_tf1_cases.py` | TF1-format image and histogram fields from `tf.compat.v1.summary.FileWriter`, omitted step, multiple images and coincident histograms |
| `generate_tf2_cases.py` | Native TF2 batched PNG/WAV, repeated metadata omitted, tensor histograms, integer matrices and adjacent large steps |
| `generate_tensorboard_cases.py` | Native `EventFileWriter` source metadata, unknown-only event, opaque plugin metadata, colliding tags and equal-step rows |
| `generate_tensorboardx_cases.py` | Two writer sessions, a restart marker, epoch joins and out-of-order steps |
| `generate_geometry_cases.py` | TensorBoardX meshes and point clouds, raw histogram buckets and derived percentiles, weighted and precomputed PR curves |
| `generate_media_cases.py` | Legacy media fields and plugin tensors: image channels, static/animated GIF, audio sample widths, and native video containers |
| `generate_raster_cases.py` | PNG color types and sample depths, baseline and progressive JPEG, and two-frame GIF/APNG/WebP with known colors and timings |
| `generate_dtype_cases.py` | Both encodings of all 20 supported numeric/boolean dtype IDs, binary strings, and raw recovery of unsupported dtype IDs |
| `generate_torch_cases.py` | CPU PyTorch scalar precision and actual `add_tensor` encodings, including records with no faithful NumPy preview |
| `generate_plugin_cases.py` | Unicode text tables, HParams domains, margin/multiline custom charts, projector TSV/sprite/bookmarks, CPU profile and unknown plugin data |
| `generate_auxiliary_cases.py` | Real checkpoint-backed projector config and six Debugger V2 streams beside an ordinary scalar; neither is mistaken for summary data |
| `generate_writer_inputs.py` | TensorBoardX grouped scalar subruns, image grids and boxes, ONNX and OpenVINO inputs converted to `GraphDef` |
| `generate_framing_cases.py` | Length and payload CRC failures, incomplete tails, and an over-limit header; these cases are generated during tests |

### Compatibility Scope

- **Version marker:** `brain.Event:2` identifies the event format, not the TensorBoard or TensorFlow release.
- **Historical comparison:** TensorBoard 1.15.0 and 2.0.0 produced identical bytes for the historical fixture.
- **TF1 API:** `tf.compat.v1` on TensorFlow 2.21.0 wrote this fixture, not a TensorFlow 1.x runtime.
- **Modern writers:** the pinned format cases use TensorBoard 2.21.0, TensorBoardX 2.6.5 and CPU PyTorch 2.12.0.

Compare the upstream plugin inventories for
[1.15.0](https://github.com/tensorflow/tensorboard/blob/1.15.0/tensorboard/default.py),
[2.0.0](https://github.com/tensorflow/tensorboard/blob/2.0.0/tensorboard/default.py) and
[2.21.0](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/default.py).

A dashboard need not define a summary type: distributions derive from histograms, and Debugger V2 uses separate files.

| Media in a verified event | Filesystem result |
| -- | -- |
| PNG (gray, indexed, gray + alpha, RGB, RGBA), baseline/progressive JPEG, BMP, WebP | `images/`, with the detected extension |
| Animated PNG and WebP | `images/<tag>/<step>.png` or `.webp`, including all frames |
| Static and animated GIF, including an `images` plugin tensor | `videos/<tag>/<step>.gif`, with the original frames |
| WAV (8-, 16-, 24-, 32-bit PCM), FLAC, MP3, Ogg/Vorbis, WebM/Opus | `audio/`, with the detected extension |
| MP4, QuickTime, AVI, WebM, Matroska, Ogg/Theora in a custom `video` summary | `videos/`, with `.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`, `.ogg` respectively |
| TIFF or AVIF image, RIFF MIDI audio | `.bin` under the corresponding tab; no invented codec or lost bytes |

The [TF2 image writer](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/plugins/image/summary_v2.py)
emits PNG and the
[TF2 audio writer](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/plugins/audio/summary_v2.py) emits
WAV.

TensorBoard's
[image MIME detector](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/util/img_mime_type_detector.py)
recognizes BMP, GIF, JPEG and PNG; the other containers test filesystem export, not dashboard playback.

The dtype matrix follows the
[2.21.0 wire enum](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/compat/proto/types.proto), not just
the [scalar writer](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/plugins/scalar/summary_v2.py),
which casts to float32.

The tests compare values and dtypes for both `tensor_content` and repeated value fields, and recover opaque records from
`raw/`.

[PyTorch 2.12.0 `add_tensor`](https://github.com/pytorch/pytorch/blob/v2.12.0/torch/utils/tensorboard/summary.py) writes
float32 bit patterns into `half_val` for float16 and bfloat16, and writes uint8 values into `uint32_val` instead of
`int_val`.

The affected records remain byte-exact in `raw/`; tboardfs does not guess a NumPy preview for these noncanonical
encodings.

Writer headers, profile timestamps and encoder output can vary when regenerating fixtures.

## License

MIT

## Star History

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="https://api.star-history.com/svg?repos=dvolgyes/tboardfs&type=Date&theme=dark"
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="https://api.star-history.com/svg?repos=dvolgyes/tboardfs&type=Date"
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=dvolgyes/tboardfs&type=Date"
  />
</picture>

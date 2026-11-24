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

tboardfs shows TensorBoard event files (`events.out.tfevents*`) as a read-only directory tree. The default mount exposes
typed files, so you can read scalars as tables, images as images, and tensors as NumPy arrays. Enable the separate `raw`
view to read original TFRecord frames, including records with no typed representation. Neither view needs TensorFlow or
TensorBoard.

```console
$ tboardfs runs mnt

$ ls mnt
experiment-1  experiment-2

$ ls mnt/experiment-1/scalars/train
accuracy.json  accuracy.tsv  loss.json  loss.tsv

$ head -3 mnt/experiment-1/scalars/train/loss.tsv
epoch   step    wall_time   relative_time   value
0.0 1   1700000001.0    0.0 1.5
1.0 2   1700000002.0    1.0 0.75

```

tboardfs reads the logs of TensorFlow 1 and 2, Keras, PyTorch (`torch.utils.tensorboard`), PyTorch Lightning,
tensorboardX, and other writers of the same event format.

There are two ways to use it:

- `tboardfs` mounts a log directory with FUSE. The mount follows a training run while it writes new events.
- `tboardfs-file` lists and copies the same virtual files without FUSE, for example in scripts, in containers, or on
  machines where you cannot mount.

## Installation

tboardfs needs Python 3.12 or later.

Mounting requires FUSE, permission to use `/dev/fuse` on Linux, and a working `fusermount` or `fusermount3`. Use
`tboardfs-file` where mount privileges are unavailable.

On Debian or Ubuntu, install FUSE:

```bash
sudo apt install fuse3
```

The Python package uses puremagic's bundled signature database. It does not require the system `libmagic` library.

Run tboardfs directly with uv:

```bash
uvx tboardfs SOURCE MOUNTPOINT
```

On x86-64 Linux with GIL-enabled CPython, libfuse 2.9 and mfusepy 3.1.1, a second optional extension lets FUSE workers
reuse Python thread state and copy encoded stat data directly into libfuse's reply buffer.

This does not change source validation or the size reported for an open handle.

Other builds, including libfuse 3 mounts, use the ordinary mfusepy callbacks.

To build a pure Python wheel, use `TBOARDFS_PURE_PYTHON=1 uv build --wheel`.

Set `TBOARDFS_PURE_PYTHON=1` for a strict editable install on a machine without a C compiler.

If an installed extension cannot load, tboardfs reports the import error rather than hiding the installation problem.

Reinstall from a pure Python wheel to use the portable implementation.

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

tboardfs runs in the background by default.

With `--foreground`, it stays in the terminal until you stop it with Ctrl+C.

To include raw frames alongside typed files, mount with `tboardfs --raw SOURCE MOUNTPOINT` or
`tboardfs -r SOURCE MOUNTPOINT`.

A background thread finds the event file with the latest modification time under SOURCE and starts indexing that file.

Mount startup does not wait for this index to finish.

The mount indexes other event files when you access their run's contents.

| Option | Default | Meaning |
| -- | -- | -- |
| `-r`, `--raw` | off | Include the raw TFRecord view alongside typed files. |
| `--scalar-format LIST` | `json,tsv` | Select the formats for scalar tables only. |
| `--histogram-format LIST` | `json,tsv,npz` | Select the formats for both histogram bucket tables and distribution percentile tables. |
| `--pr-curve-npy` | off | Add NPY files for PR curves alongside JSON and TSV. |
| `--step-digits N` | `6` | Zero-padding width of step file names, from 1 to 20. |
| `--refresh-age-seconds S` | `150` | Minimum interval between local searches for new files and directories (`0` to search on each access, `inf` to freeze each directory after its first scan). |
| `--source-check-seconds S` | `120` | Minimum interval between status checks of known event files in an accessed run. Accepts finite, nonnegative seconds; `0` checks on every access. |
| `--cache-size-mb N` | `1024` | Memory for rendered files. `0` turns the cache off. |
| `-j N`, `--index-workers N` | `4` | Maximum workers per cold file index. |
| `--disk-index` | off | Reuse verified per-file indexes across mounts. |
| `--disk-index-size-mb N` | `2048` | Disk budget in MiB for saved index payloads; does not enable the disk index by itself. |
| `--root-owner` / `--user-owner` | `--root-owner` | Owner of all files and directories: root, or the user who mounts. |
| `--foreground` / `--background` | `--background` | Stay in the terminal, or detach. |
| `--logfile FILE` | none | Write the log to FILE. |
| `--loglevel LEVEL` | `INFO` | `trace`, `debug`, `info`, `success`, `warning`, `error`, or `critical`, in any case. |

Each of `--scalar-format` and `--histogram-format` replaces its default with a comma-separated list of `json`, `tsv`,
`csv` and `npz`.

In the background, the log goes only to `--logfile`. Without `--logfile`, you do not see the warnings about damaged
data; use `--logfile` or `--foreground` when you need them.

**Storage:** The default is `$XDG_CACHE_HOME/tboardfs/indexes-v1`, or `~/.cache/tboardfs/indexes-v1` when no absolute
XDG cache directory is configured.

The cache can contain training values and metadata: choose a private location outside the source and mount with no
symlink components.

Entries expire 30 days after writing; subsequent writes reclaim expired entries and evict the oldest writes to fit
`--disk-index-size-mb`.

Deleting the cache while unmounted loses only the speedup, not the event data.

## Command Mode

```bash
tboardfs-file [--logfile FILE] [--loglevel LEVEL] [--framing-only] COMMAND [OPTIONS] SOURCE ...
```

SOURCE is one event file, a run directory, or a log directory.

- For one event file, the tabs are at the root, for example `/scalars` and `/images`.
- Projector, profile, Open3D and other sidecar files require a run directory as SOURCE.
- For a directory, the file paths use the same run layout as a mount, with typed and raw selection controlled by the
  command.

| Command | Purpose |
| -- | -- |
| `list SOURCE [PREFIX]` | Print the virtual file paths below PREFIX, one per line. |
| `get SOURCE VPATH -o OUTPUT` | Write one virtual file to OUTPUT. `-o -` writes it to stdout. |
| `copy-all SOURCE OUTDIR` | Copy the selected view into OUTDIR. |

`list`, `get` and `copy-all` accept `--scalar-format`, `--histogram-format`, `--pr-curve-npy` and `--step-digits` with
the same defaults and behavior as in filesystem mode.

`list` and `copy-all` default to typed files.

Use `--view raw` for complete frames or `--view both` for both views.

`list` with an explicit raw prefix and `get` with an explicit raw path can select a frame directly.

For a single event file, `tboardfs-file --framing-only list SOURCE` lists raw frames without decoding summaries or
calling the payload classifier.

`--framing-only` also works with `get` and `copy-all` and implies the raw view.

It checks both CRCs of every frame before producing output, even when you request only the first frame.

Use the default mode when you also want warnings about malformed protobufs or typed conversions.

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

A run is a directory that directly contains one or more event files. Each run's typed tabs appear in the run directory.

```text
MOUNTPOINT/
├── experiment-1/                  a run
│   ├── llms.txt                   guide to this run's exports
│   ├── scalars/train/loss.json    the tag train/loss
│   ├── images/sample/000010.png   the tag sample at step 10
│   ├── ...                        the other tabs
│   └── eval/                      a run inside the run experiment-1
│       └── scalars/...
└── experiment-2/
    └── ...
```

When SOURCE is itself a run, its typed tabs appear at the mount root.

The mount omits empty tabs and other empty directories, including source branches with no exported content.

Each directory that directly contains an event file exposes an `llms.txt` guide.

Multiple event files in a run contribute to the same typed tabs but retain separate directories under `raw` when that
view is enabled.

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

## Exported Data

| Tab | Files | Content |
| -- | -- | -- |
| `scalars` | `<tag>.json`, `.tsv` (`.csv` and `.npz` opt-in) | One table per tag, all steps |
| `histograms` | `<tag>/<step>.json`, `.tsv`, `.npz` (`.csv` opt-in) | The buckets of one histogram |
| `distributions` | `<tag>/<step>.json`, `.tsv`, `.npz` (`.csv` opt-in) | The 9 percentiles of the Distributions tab |
| `pr_curves` | `<tag>/<step>.json`, `.tsv` (`.npy` opt-in) | One row per threshold |
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
| `plugins/Open3D` | `<tag>/<step>.manifest.json`, `~assets/*.msgpack` | Native geometry ranges, references to earlier steps and original sidecar bytes |
| `raw` (opt-in on mounts) | `<event-file>/<20-digit-byte-offset>.tfrecord` | One complete frame: header, payload and footer, in source order |

### Migrating format options

`--scalar-format` previously selected formats for scalars, histograms and distributions.

To keep an explicit format list for all three views, pass the same list to both `--scalar-format` and
`--histogram-format`.

For example, replace `--scalar-format csv` with `--scalar-format csv --histogram-format csv`.

If you previously relied on defaults, use `--scalar-format json,tsv,npz` to retain scalar NPZ files.

The default histogram and distribution formats have not changed.

Use `--pr-curve-npy` to retain PR-curve NPY files.

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

For small tables with identical metadata on every row, the text writer formats those fields once.

For common NPZ shapes, the mount reuses a NumPy-generated archive layout, not data from an earlier step.

### PR Curves

The `.json` and `.tsv` tables have one row per threshold, with the columns `threshold`, `true_positives`,
`false_positives`, `true_negatives`, `false_negatives`, `precision` and `recall`.

The thresholds are evenly spaced in [0, 1], as in TensorBoard.

A tensor of another shape has the columns `value_0`, `value_1`, and so on.

With `--pr-curve-npy`, the `.npy` file contains the stored tensor, normally shaped `[6, N]`.

### Images, Audio and Videos

Media files contain the original bytes; tboardfs does not transcode them.

- A multi-sample image or audio summary has one file per sample, such as `000001_0.png` and `000001_1.png`.
- Audio tensors can hold a vector of clips or `[clip, label]` rows; the labels remain in the raw event.
- Both single-frame and animated GIF images go under `videos/`.
- TensorBoardX and PyTorch `add_video` write GIF image summaries rather than native video containers.
- A custom `video` plugin can instead store a native video container.
- An audio summary stays under `audio/` even if the bytes identify a different format.
- A TF2 image tensor must be a vector containing two dimension strings followed by the declared number of images. If its
  layout is inconsistent, `plugins/images/` contains the packed JSON and the optional raw view retains the original
  event.

Filename extensions identify likely content; they do not certify that it will decode. Unknown payloads keep `.bin`. If a
generated tensor view and an embedded file would have the same path, the generated view keeps its name. For example,
`000007.npy` describes the outer byte tensor and `000007.embedded.npy` contains the original NPY file.

To store a GeoTIFF, put the **encoded TIFF file bytes** in `Summary.Image.encoded_image_string` (legacy) or after the
width and height strings in an `images` plugin tensor (TF2).

Passing a pixel array to `tf.summary.image` instead produces PNG without the TIFF tags.

Both classic TIFF and BigTIFF export as `.tiff`; a reader such as Rasterio uses the internal offsets and georeferencing.

External world files, `.aux.xml` files and `.ovr` files are not part of the encoded TIFF.

Repeated values with the same tag retain their own payloads even when a writer omits repeated plugin metadata.

### Text

A text summary with one string has a `.txt` preview.

For multiple strings, the `.txt` preview is a Markdown table: a vector is one column, and a tensor with more than two
dimensions shows its first 2-D slice with a warning.

The adjacent `.json` preview keeps the full shape and all strings, but both previews replace invalid UTF-8 with `�`.

The text and JSON previews preserve trailing NUL characters.

Use the raw event frame to recover the exact string bytes.

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
- String tensors use a pickle-free fixed-width Unicode dtype (`U`) in `.npy` files.
- Use the JSON preview for decoded strings with trailing NULs because NumPy cannot distinguish them from padding.
- A numeric tensor without a plugin name is also a tensor here.
- A single repeated-field value can fill the declared shape up to 1 KiB of expanded data. Shapes requiring more
  expansion do not get an array preview.
- Each string member and each `uint8` content field can expose a recognized file, even below 1 KiB.
- Short labels and unrecognized short fields do not get extra files, even beside a larger member.
- At 1 KiB or above, a field can also have an opaque `.bin` or plain `.txt` export.
- A well-formed packed `float32` tensor of 1 KiB or more keeps its raw `.bin` export even if the bytes match a known
  file format.
- These files contain the original field bytes, not the enclosing record.
- Use `raw` to recover the full frame, including unsupported types and plugin metadata.

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

- `projector` contains the config, the files of a `projector` directory and the relative files named by `tensor_path`,
  `metadata_path`, sprite `image_path` and `bookmarks_path`.
- Only relative paths to files of the run are followed, and only the first MiB of the config is read.
- `profile` contains the files below a `profile` directory of the run, such as `plugins/profile/...`.

The config bytes are unchanged, but the exported filenames use the filesystem's path escaping.

For example, TensorBoardX writes the tag `sky/positions` as `sky%2fpositions`, which exports as `sky%252fpositions`;
restore that directory name before opening the copied config in TensorBoard.

tboardfs copies profiler `.xplane.pb` files without decoding them; it does not extract checkpoint tensors or decode
Debugger V2 dumps.

### Other Plugins

Summaries of a plugin without its own tab are JSON files under `plugins/<plugin>/<tag>/<step>.json`.

The [Open3D plugin](https://www.open3d.org/docs/release/tutorial/visualization/tensorboard_plugin.html) stores geometry
in `plugins/Open3D/*.msgpack` beside the event file, not inside a TensorBoard mesh tensor.

For a version 14 index, open `plugins/Open3D/<tag>/<step>.manifest.json` to find the sidecar, the byte ranges for each
batch item, any auxiliary ranges and the steps that supply reused properties.

The `masked_crc32c` fields are the checksums recorded by the writer; listing a manifest does not verify the sidecar.

The `asset` field is a path relative to the virtual run root, or `null` if no discovered file covers the recorded
ranges.

The filesystem exposes regular `.msgpack` files under `plugins/Open3D/~assets/` without following symlinks below the run
root.

It does not convert an Open3D scene to OBJ or PLY or require an Open3D installation.

```bash
tboardfs-file get tests/data/producer-cases/open3d /plugins/Open3D/triangle_mesh/000002.manifest.json -o -
tboardfs-file get tests/data/producer-cases/open3d /plugins/Open3D/~assets/triangle_mesh.msgpack -o mesh.msgpack
```

## File Formats

### JSON

All `.json` files are strict JSON (RFC 8259), UTF-8, and indented with 2 spaces.

- A table is a list of row objects, with the column names as keys.
- Infinities are the strings `"Infinity"` and `"-Infinity"`.
- In tables, NaN is a missing value and is `null`. In other JSON files, NaN is the string `"NaN"`, as in protobuf JSON.
- A complex number is its `[real, imag]` pair.
- In record JSON, valid UTF-8 bytes become text and other bytes become `{"base64": "..."}`.
- Text and string-array previews instead decode invalid UTF-8 with replacement.
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

Include `npz` in the scalar format list when mounting the logs for this example:

```bash
tboardfs --scalar-format json,tsv,npz runs mnt
```

```python
import numpy as np

table = np.load("mnt/run/scalars/loss.npz")
steps, values = table["step"], table["value"]
```

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

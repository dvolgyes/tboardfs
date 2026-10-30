# tboardfs

<p>
  <a href="https://github.com/dvolgyes/tboardfs/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/dvolgyes/tboardfs/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://coveralls.io/github/dvolgyes/tboardfs?branch=main"><img alt="Coverage Status" src="https://coveralls.io/repos/github/dvolgyes/tboardfs/badge.svg?branch=main" /></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://pypi.org/project/tboardfs/"><img alt="Version: 0.2.3" src="https://img.shields.io/badge/version-0.2.3-orange.svg" /></a>
  <a href="https://pypi.org/project/tboardfs/"><img alt="Status: Alpha" src="https://img.shields.io/badge/status-alpha-yellow.svg" /></a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green.svg" /></a>
  <a href="https://www.python.org/"><img alt="Python: >=3.12" src="https://img.shields.io/badge/python-%3E=3.12-blue.svg" /></a>
</p>

Mount and inspect TensorBoard event logs as ordinary files.

`tboardfs` has two access modes:

- Filesystem mode mounts a log directory with FUSE.
- Command mode inspects or copies the virtual content of one event file.

Both modes use the same virtual paths for TensorBoard tabs such as `scalars`, `images`, `tensors`, `meshes`,
`pr_curves`, `hparams`, and `custom_scalars`.

## Filesystem Mode

Mount a TensorBoard log directory:

```bash
tboardfs SOURCE MOUNTPOINT
```

For non-developers, the simplest way to run it without cloning this repository or managing a local virtual environment
is:

```bash
uvx tboardfs SOURCE MOUNTPOINT
```

Example:

```bash
tboardfs runs/experiment-1 mnt/tboardfs
ls mnt/tboardfs
cat mnt/tboardfs/train/scalars/loss.json
```

Mount mode is the primary workflow for browsing full log directories. It exposes run directories and supported sidecars
such as projector/profile files.

Each open file keeps the version it had when it was opened, so reads stay whole while a training run writes new events.
A new `open` shows the new data. Mounted files use direct I/O: shared memory maps (`mmap` with `MAP_SHARED`,
`np.load(..., mmap_mode="r")`) fail with `ENODEV`; private maps (`mmap_mode="c"`) and plain reads work.

Rendered files are kept in memory, so `ls -l` followed by `cat` renders a file once. `--cache-size-mb` sets the memory
for this cache (default 1024; 0 turns it off). When the cache goes over the limit, the least recently used files are
dropped until half the limit is left. New events in a run make only the files of that run render again.

A resumed run hides the rows it replaces, as TensorBoard does. When a writer marks a restart at step S (a
`SessionLog.START` event, for example from PyTorch `SummaryWriter(purge_step=S)`), rows written before it with step S or
higher are not shown. Graphs stay. This applies to event files with file version 2 or later, which all current writers
produce.

## Command Mode

Use `tboardfs-file` when mounting is unavailable or when you need scripted access to one TensorBoard event file.

List every virtual file:

```bash
tboardfs-file list events.out.tfevents.123
```

List files below a virtual path:

```bash
tboardfs-file list events.out.tfevents.123 /meshes
```

Extract one virtual file to stdout:

```bash
tboardfs-file get events.out.tfevents.123 /scalars/loss.json -o -
```

Extract one virtual file to disk:

```bash
tboardfs-file get events.out.tfevents.123 /images/sample/000001.png -o sample.png
```

The virtual path may be written with or without the leading `/`. If `-o` points to an existing directory, or to a
missing path ending in `/`, `get` writes the virtual file basename inside that directory.

```bash
tboardfs-file get events.out.tfevents.123 scalars/loss.json -o exported/
```

Copy the full virtual tree:

```bash
tboardfs-file copy-all events.out.tfevents.123 exported-event
```

Command mode exposes tab directories directly at the root, for example `/scalars`, `/images`, and `/meshes`. It accepts
a single event file and does not include sibling sidecars.

`get` refuses to overwrite existing files unless `--force` is provided. `copy-all` copies files in deterministic
virtual-path order. If it reaches an existing target without `--force`, it stops, reports how many files were already
copied, lists those virtual paths, prints the conflicting output path, and suggests `--force`. After a successful copy,
`copy-all` reports the number of copied files.

Use `--skip-existing` to leave existing output files untouched and continue copying the rest of the tree:

```bash
tboardfs-file copy-all events.out.tfevents.123 exported-event --skip-existing
```

Each skipped virtual path is reported as a warning on stderr. `--force` and `--skip-existing` are mutually exclusive.

## Supported Objects

The virtual tree covers these TensorBoard objects:

- Scalars: full-series `json`, `tsv`, and `npz` exports under `/scalars`.
- Custom scalars: layout JSON under `/custom_scalars`.
- Images: encoded image files under `/images`.
- Audio: encoded audio files under `/audio`.
- Videos: encoded video/GIF outputs under `/videos`, including raw subpaths when present.
- Histograms: per-step `json`, `tsv`, and `npz` exports under `/histograms`.
- Distributions: per-step distribution tables under `/distributions`.
- Text summaries: UTF-8 text files under `/text`. A summary with one string is that text. A summary with many strings is
  a Markdown table, as TensorBoard shows it: a vector is one column, and a tensor with more than 2 dimensions shows its
  first 2-D slice after a warning. A `.json` file next to the table keeps the shape and the exact strings.
- Meshes: per-step `json`, `npz`, and Wavefront `obj` exports under `/meshes`. The OBJ file has one object
  (`o mesh_<i>`) per batch element when the batch has more than one mesh. Vertex colors follow the coordinates as
  `r g b` values in [0, 1], which MeshLab and Blender read. A mesh without faces is an OBJ point cloud.
- PR curves: per-step `json`, `tsv`, and `npy` exports under `/pr_curves`. The tables have one row per threshold, with
  the columns `threshold`, `true_positives`, `false_positives`, `true_negatives`, `false_negatives`, `precision`, and
  `recall`. The thresholds are evenly spaced in [0, 1], as in TensorBoard. The `.npy` file keeps the raw `[6, N]`
  tensor. A tensor of another shape keeps generic `value_<i>` columns.
- HParams: merged experiment/session/metric JSON under `/hparams`. The metrics are the declared metric tags, or all
  scalar tags when the run declares no experiment, as in TensorBoard. Each metric has its last value, or `null` without
  data. Metrics in a sub-run group (for example Keras `validation`) have no value here; read them from the scalars of
  that run. `sessions` lists every session of the run; `session_start` and `session_end` are the first ones, as in
  TensorBoard.
- Tensors: tensor arrays as `npy` plus compact JSON or native blob files under `/tensors`. Every numeric TensorFlow
  dtype that NumPy has keeps its dtype, including float16, int8, int16, uint16 and complex. NumPy has no bfloat16, so
  bfloat16 tensors are widened to float32; this is exact. Float8 and 4-bit tensors are not decoded. String tensors are
  NumPy text arrays (dtype `U`), so `np.load` reads them without `allow_pickle`.
- Graphs: graph protobuf files under `/graphs`. `graph.pb` is the newest graph of the run, as in TensorBoard. A run with
  several graphs also has `graph_<i>.pb` for each graph, in write order.
- Projector and profile sidecars: available in filesystem mode under `/projector` and `/profile`; command mode
  intentionally omits sibling sidecar files because it operates on one event file. The embedding files that
  `projector_config.pbtxt` names (`tensor_path`, `metadata_path`, sprite `image_path`, `bookmarks_path`), such as the
  `<step>/<tag>/tensors.tsv` files of PyTorch and tensorboardX `add_embedding`, are under `/projector` at the same
  relative paths. Only relative paths to files of the run are followed, and only the first MiB of the config is read.
- Other plugin JSON payloads: exposed under `/plugins` when no typed tab handles them.

## Output Rules

All `.json` files are strict JSON (RFC 8259). Infinities are written as the strings `"Infinity"` and `"-Infinity"`. In
scalar, histogram and other table files, NaN is a missing value and is written as `null`. In other JSON files, NaN is
written as the string `"NaN"`, as in protobuf JSON. A complex number is written as its `[real, imag]` pair. Bytes that
are valid UTF-8 are written as text. Other bytes are written as `{"base64": "..."}`.

`tboardfs-file list` prints virtual file paths to stdout, one per line. `tboardfs-file get -o -` writes raw bytes to
stdout. Extraction status, warnings, errors, copied-file counts, skipped-path warnings, copied-path conflict reports,
and overwrite hints are written to stderr through Loguru/Click so stdout remains safe for pipelines.

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

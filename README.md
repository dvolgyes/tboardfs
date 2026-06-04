# tboardfs

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python: >=3.12](https://img.shields.io/badge/python-%3E=3.12-blue.svg)](https://www.python.org/)
[![Version: 0.2.2](https://img.shields.io/badge/version-0.2.2-orange.svg)](https://pypi.org/project/tboardfs/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-yellow.svg)](https://pypi.org/project/tboardfs/)

Mount and inspect TensorBoard event logs as ordinary files.

`tboardfs` has two access modes:

- Filesystem mode mounts a log directory with FUSE.
- Command mode inspects or copies the virtual content of one event file.

Both modes use the same virtual paths for TensorBoard tabs such as `scalars`,
`images`, `tensors`, `meshes`, `pr_curves`, `hparams`, and `custom_scalars`.

## Filesystem Mode

Mount a TensorBoard log directory:

```bash
tboardfs SOURCE MOUNTPOINT
```

For non-developers, the simplest way to run it without cloning this repository
or managing a local virtual environment is:

```bash
uvx tboardfs SOURCE MOUNTPOINT
```

Example:

```bash
tboardfs runs/experiment-1 mnt/tboardfs
ls mnt/tboardfs
cat mnt/tboardfs/train/scalars/loss.json
```

Mount mode is the primary workflow for browsing full log directories. It exposes
run directories, FUSE cache-control files, and supported sidecars such as
projector/profile files.

## Command Mode

Use `tboardfs-file` when mounting is unavailable or when you need scripted
access to one TensorBoard event file.

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

The virtual path may be written with or without the leading `/`. If `-o` points
to an existing directory, or to a missing path ending in `/`, `get` writes the
virtual file basename inside that directory.

```bash
tboardfs-file get events.out.tfevents.123 scalars/loss.json -o exported/
```

Copy the full virtual tree:

```bash
tboardfs-file copy-all events.out.tfevents.123 exported-event
```

Command mode exposes tab directories directly at the root, for example
`/scalars`, `/images`, and `/meshes`. It accepts a single event file, omits FUSE
control files, and does not include sibling sidecars.

`get` refuses to overwrite existing files unless `--force` is provided.
`copy-all` copies files in deterministic virtual-path order. If it reaches an
existing target without `--force`, it stops, reports how many files were already
copied, lists those virtual paths, prints the conflicting output path, and
suggests `--force`. After a successful copy, `copy-all` reports the number of
copied files.

Use `--skip-existing` to leave existing output files untouched and continue
copying the rest of the tree:

```bash
tboardfs-file copy-all events.out.tfevents.123 exported-event --skip-existing
```

Each skipped virtual path is reported as a warning on stderr. `--force` and
`--skip-existing` are mutually exclusive.

## Supported Objects

The virtual tree covers these TensorBoard objects:

- Scalars: full-series `json`, `tsv`, and `npz` exports under `/scalars`.
- Custom scalars: layout JSON under `/custom_scalars`.
- Images: encoded image files under `/images`.
- Audio: encoded audio files under `/audio`.
- Videos: encoded video/GIF outputs under `/videos`, including raw subpaths
  when present.
- Histograms: per-step `json`, `tsv`, and `npz` exports under `/histograms`.
- Distributions: per-step distribution tables under `/distributions`.
- Text summaries: UTF-8 text files under `/text`.
- Meshes: per-step `json`, `npz`, and Wavefront `obj` exports under `/meshes`.
- PR curves: per-step `json`, `tsv`, and `npy` exports under `/pr_curves`.
- HParams: merged experiment/session/metric JSON under `/hparams`.
- Tensors: tensor arrays as `npy` plus compact JSON or native blob files under
  `/tensors`.
- Graphs: graph protobuf files under `/graphs`.
- Projector and profile sidecars: available in filesystem mode under
  `/projector` and `/profile`; command mode intentionally omits sibling
  sidecar files because it operates on one event file.
- Other plugin JSON payloads: exposed under `/plugins` when no typed tab handles
  them.

## Output Rules

`tboardfs-file list` prints virtual file paths to stdout, one per line.
`tboardfs-file get -o -` writes raw bytes to stdout. Extraction status,
warnings, errors, copied-file counts, skipped-path warnings, copied-path
conflict reports, and overwrite hints are written to stderr through Loguru/Click
so stdout remains safe for pipelines.

## License

MIT

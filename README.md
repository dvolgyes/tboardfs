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
| `--refresh-age-seconds S` | `150` | Minimum interval between local searches for new files and directories (`0` for immediate discovery, `inf` to freeze each directory after its first scan). |
| `--cache-size-mb N` | `1024` | Memory for rendered files. `0` turns the cache off. |
| `-j N`, `--index-workers N` | `1` | Maximum workers per cold file index (`1` keeps indexing serial). |
| `--disk-index` | off | Reuse verified per-file indexes across mounts. |
| `--disk-index-size-mb N` | `2048` | Disk budget in MiB for saved index payloads; does not enable the disk index by itself. |
| `--root-owner` / `--user-owner` | `--root-owner` | Owner of all files and directories: root, or the user who mounts. |
| `--foreground` / `--background` | `--background` | Stay in the terminal, or detach. |
| `--logfile FILE` | none | Write the log to FILE. |
| `--loglevel LEVEL` | `INFO` | `trace`, `debug`, `info`, `success`, `warning`, `error`, or `critical`, in any case. |

In the background, the log goes only to `--logfile`. Without `--logfile`, you do not see the warnings about damaged
data; use `--logfile` or `--foreground` when you need them.

The initial serial scan uses a 64 KiB read buffer for event files of at least 64 KiB.

Smaller files and append refreshes use Python's default buffering.

Serial and parallel indexing skip the temporary value list for three legacy summary shapes: one scalar, one image, or a
scalar followed by an image.

If any value in the summary declares or inherits metadata, the indexer processes the whole summary through the
metadata-aware path instead.

A plugin declaration applies to its own value and later values with the same tag, not earlier values in the same
summary.

Legacy scalars and metadata-only values on this path skip histogram and media detection without skipping plugin output.

### Parallel cold indexing

For a large event file that takes tens of seconds to open, try `tboardfs -j 4 SOURCE MOUNTPOINT`.

The mount starts workers only for uncached regular event files of at least 8 MiB.

The mount limits the worker count to the available CPUs and divides the snapshot into equal byte ranges.

The planner follows TFRecord length headers from offset zero with a 4 KiB buffer, seeking past payloads.

It submits each nonempty range as soon as it knows its first record start and stops searching once it reaches the final
range or EOF.

Each worker owns records whose starting offsets fall in its range, including the lower bound but excluding the upper
bound.

Workers use a 64 KiB buffer and validate complete records, including any record that crosses their upper bound.

A separate prefix scan estimates the initial clock and metadata for later workers.

The mount merges worker results in file order.

If only the starting epoch was wrong, the mount corrects the affected scalar and histogram rows in memory.

It leaves rows from the first local epoch onward unchanged, even if that epoch equals the worker's estimate.

If a range used an incorrect first timestamp or inherited plugin metadata, the mount reparses that range.

Before exposing the index, the mount checks that the ranges cover the unchanged snapshot without gaps or overlaps.

Workers can write a small legacy `Summary.Image` directly to a packed row when its only siblings are ordinary scalars.

Images that need a verified byte range or share a record with another image use the general encoding path.

The parent builds directory names from packed binary rows and resolves an individual entry when its file is accessed.

For ordinary numeric tensors with a verified `tensor_content` extent of at least 64 KiB, a live parallel mount can list
the `.npy` and `.json` files without rereading the tensor.

The first operation that needs the file's contents or exact size reads and verifies the source bytes.

If the snapshot changes, validation detects corruption or inconsistent coverage, or a worker exits unexpectedly, the
mount discards the partial index and uses the serial reader.

Repeated reads reuse the cache; appends resume from the existing index without starting workers.

On memory-constrained systems, keep the serial default: each active worker needs memory in addition to the index.

### Reuse an index on the next mount

Use `tboardfs --disk-index SOURCE MOUNTPOINT` when you repeatedly mount the same large event files.

Without `--disk-index`, tboardfs neither reads nor writes persistent indexes.

- **Miss:** A new path or size goes straight to the configured parser; add `-j N` to use parallel indexing on large
  files.
- **Candidate:** tboardfs checks a 16-byte XXH3-128 fingerprint of **all** current event-file bytes before loading the
  saved index.
- **Hit:** The bytes, parser configuration and plugin metadata inherited from earlier files must all match.
- **Multiple files:** `-j N` allows up to four concurrent candidate hashes in the active run.
- **Diagnostics:** `--loglevel DEBUG` reports `persistent index stored` or `persistent index hit` with the event-file
  path.

The persistent index uses XXH3-128 to fingerprint the parser configuration and to checksum the saved index bytes.

These hashes detect accidental changes; they do not authenticate the source or the cache.

The first mount pays for parsing, hashing and writing; subsequent hits avoid protobuf parsing and file-type detection
but still verify the source and build the virtual tree.

On a parallel cache miss, tboardfs writes ordinary packed binary rows to the persistent index without first creating a
Python entry object for each row.

The serial and parallel writers use the same disk format and check the rows against the indexed TFRecord frames.

The tree stores the resolved names, including collision suffixes, before a directory is listed.

Listing a directory does not construct its typed file nodes; looking up a file constructs only that node.

On a persistent-index hit, tboardfs validates every binary row against the frame index before exposing any files.

It uses the packed labels, steps and timestamps to place binary files without constructing a full entry for each one.

Listing names or checking a binary file's size does not unpack its frame and byte-range evidence.

Reading the file unpacks that evidence once and reuses it for subsequent reads.

The FUSE mount returns each name with its file type, so a name-only walk such as `find MOUNTPOINT` does not need a
separate `stat` for every file on Linux.

Commands that request sizes or timestamps, such as `find MOUNTPOINT -ls`, still need file attributes.

Only complete, stable parses without skipped records or conversion warnings are saved.

Within a mount, appends still extend the in-memory index; a later mount reparses a changed file rather than trusting a
saved prefix.

An unreadable source is never served from the cache.

Restart the mount process after upgrading libmagic or changing its rule files.

**Storage:** The default is `$XDG_CACHE_HOME/tboardfs/indexes-v1`, or `~/.cache/tboardfs/indexes-v1` when no absolute
XDG cache directory is configured.

The cache can contain training values and metadata: choose a private location outside the source and mount with no
symlink components.

Entries expire 30 days after writing; subsequent writes reclaim expired entries and evict the oldest writes to fit
`--disk-index-size-mb`.

The budget counts index payloads, not the catalog or temporary files, and a single index cannot exceed 512 MiB.

If the cache is unavailable or incompatible, tboardfs uses the normal parser.

Deleting the cache while unmounted loses only the speedup, not the event data.

### Live Training Runs

`ls -l MOUNTPOINT` can show run directories without indexing their event files.

Listing a run or looking up a path inside it indexes the event files directly in that run, not those in sibling runs.

A recursive `find MOUNTPOINT` enters each listed run, so its first traversal still indexes them all.

When only leaf runs have exports and all run folders retain their original names, updating a leaf run replaces its
branch without rebuilding populated siblings.

This works for sibling runs and for runs nested below grouping directories.

Adding or removing a directory, changing a run with child runs, or resolving a folder-name collision uses full
placement.

After a run is indexed, accessing its virtual paths checks known event files even within the discovery interval.

The check uses device, inode, birth time where available, size, modification time and status-change time; it ignores
access time.

Unchanged sources need no reopen for indexing.

On a FUSE mount with `--disk-index`, the status check excludes the private index without a separate path walk.

Opening a virtual file on a live run checks the known event files again, even after a path-based `getattr`.

For regular files in one directory, a live refresh resolves the directory once and checks each file relative to it.

On Linux, when libc provides `openat2` and the kernel accepts it, tboardfs can open an ordinary absolute directory path
in one system call without following symlinks.

Between directory scans, tboardfs reuses the names and expected fingerprints of indexed sibling files.

It opens the parent again and compares each regular file with its indexed fingerprint.

If any check fails or detects a change, tboardfs refreshes the run in source order to preserve inherited metadata.

For paths that need symlink resolution or cannot use `openat2`, tboardfs keeps the guarded component-by-component
lookup.

If that directory is renamed during the check, tboardfs still checks its regular files through the open directory.

The next refresh resolves the directory name again.

Local directory discovery finds new event files, subdirectories and sidecars without scanning idle runs.

`--refresh-age-seconds inf` freezes a directory after its first listing or lookup inside it.

Before tboardfs indexes a run, `stat` reports its parent's published time if it is nonzero.

Otherwise it uses the backing directory's modification time, without opening event files.

If tboardfs cannot safely read that directory's metadata, `stat` reports time zero for that lookup.

Listing the run discovers its contents, so its timestamp can change even if no process wrote to it.

After indexing, `stat` reports the newest discovered content time and checks for changes in live mode.

Append complete frames, and replace the event file when rewriting history.

On growth, tboardfs verifies the last complete frame and resumes at its end, retrying any incomplete tail.

When only the last event file in filename order grows, the mount can update the affected directories and reuse the rest
of the run tree.

It updates dependent exports too: an appended epoch can change existing scalar rows, histograms and distributions.

Some changes require a full build of that run's directory tree, for example a restart purge, a change to an earlier
event file, or a name collision that can move existing files.

Listing a changed directory still returns all its names; an incremental update does not make listing cost independent of
directory size.

A replacement, a decrease in observed file size, a same-size metadata change or an invalid checkpoint requires a fresh
index.

A deleted or unreadable source drops out of the active view without invalidating unrelated runs.

Changing an earlier frame *and* growing the same inode can leave old typed rows visible under the append-only
assumption.

If the filesystem reports identical metadata across a same-size rewrite, a status check alone cannot detect it.

Opening a rendered export pins its bytes; opening a directly indexed binary pins the source descriptor.

Appending, renaming or unlinking that source does not redirect an open handle to a replacement.

An in-place rewrite can make the original binary unavailable: if verification fails, the read returns `EIO` rather than
mixing versions.

### Memory and Caching

The first indexing pass collects scalar columns by tag and records each stored binary's format and byte count.

If a record contains only `simple_value` scalars and legacy images, with no explicit or inherited metadata, the indexer
reads the decoded protobuf fields directly into scalar columns and binary entries.

When a serial mount is not saving a disk index, it defers frame construction for direct-path images without a byte
range.

Listing these images and checking their sizes needs no frame object.

Reading one still verifies the original source identity and whole-frame checksum.

Other records use the general conversion path, including records that mix these values with plugin tensors.

A decoded `scalars` plugin value goes directly into its scalar table; a value that cannot be decoded as a scalar still
uses the normal plugin handling.

Legacy image and audio values without plugin metadata go directly to binary indexing.

Metadata inherited from an earlier value with the same tag still applies.

During each scan, tboardfs remembers the last audio clip larger than 4 KiB and no larger than 1 MiB.

It reuses the detected extension only if the complete clip matches, so a custom libmagic rule can still distinguish
clips with identical headers.

The classifier loads libmagic only for payloads that need its rules, or before mounting with the disk index enabled to
fingerprint those rules.

A valid `M4A ` major brand does not need a MIME lookup; the existing HEIF-family image brand checks take precedence.

For unambiguous fields in frames of at least 4 KiB, the index also records the exact payload range.

Directory listings return names and file types together.

Code that needs only names and types can use `os.scandir()` and `entry.is_dir(follow_symlinks=False)` without a separate
`stat` for every file.

To list file paths without FUSE, use `tboardfs-file list --view both SOURCE`.

Directories reuse their sorted names for listings.

An exact name lookup uses a mapping.

Raw frame lookup searches sorted byte offsets without allocating all raw filenames.

Stat uses the indexed size of stored binaries and graphs; it does not render them.

On a stable source, a mapped binary read checks only the 4 KiB blocks that overlap the requested range.

If source metadata changes, the reader revalidates the indexed frame against its original CRC before trusting the new
metadata.

These checks detect accidental corruption, not deliberate tampering; if metadata is identical, a read does not check
unrequested blocks.

Small frames and ambiguous protobuf layouts use full-frame verification and extraction; derived exports such as CSV and
NumPy files still require rendering.

Rendered exports use a run-local generation and a shared byte budget.

When `--cache-size-mb` is exceeded, the least recently used renders are dropped until at most half the budget remains.

Changing a run discards its obsolete renders without evicting the renders of other runs.

The budget excludes indexes, directory entries, and content pinned by open handles.

### Permissions and Memory Maps

The filesystem is read-only. Files have mode `0444` and directories `0555`. They belong to root (uid and gid 0), or to
the user who mounts with `--user-owner`.

Mounted files use direct I/O. Plain reads and private memory maps work, for example `np.load(path, mmap_mode="c")`.
Shared memory maps (`mmap` with `MAP_SHARED`, `np.load(path, mmap_mode="r")`) fail with `ENODEV`.

## Command Mode

```bash
tboardfs-file [--logfile FILE] [--loglevel LEVEL] [--framing-only] COMMAND [OPTIONS] SOURCE ...
```

SOURCE is one event file, a run directory, or a log directory.

- For one event file, the tabs are at the root, for example `/scalars` and `/images`.
- Projector, profile, Open3D and other sidecar files require a run directory as SOURCE.
- For a directory, the tree is the same as a mount of that directory.

| Command | Purpose |
| -- | -- |
| `list SOURCE [PREFIX]` | Print the virtual file paths below PREFIX, one per line. |
| `get SOURCE VPATH -o OUTPUT` | Write one virtual file to OUTPUT. `-o -` writes it to stdout. |
| `copy-all SOURCE OUTDIR` | Copy the selected view into OUTDIR. |

All three commands accept `--scalar-format` and `--step-digits`, with the same meaning as in filesystem mode. `list` and
`copy-all` default to typed files. Use `--view raw` for complete frames or `--view both` for both views. `list` with an
explicit raw prefix and `get` with an explicit raw path can select a frame directly.

For a single event file, `tboardfs-file --framing-only list SOURCE` lists raw frames without decoding summaries or
calling libmagic.

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
| `plugins/Open3D` | `<tag>/<step>.manifest.json`, `~assets/*.msgpack` | Native geometry ranges, references to earlier steps and original sidecar bytes |
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
- An audio summary stays under `audio/` even if the bytes identify a different format.
- A TF2 image tensor must be a vector containing two dimension strings followed by the declared number of images. If its
  layout is inconsistent, `plugins/images/` contains the packed JSON and `raw/` keeps the original event.

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

Protobuf decodes the payload but does not check either TFRecord checksum.

The reader reuses a validated length only if the next 12 header bytes match exactly, including the stored checksum.

It still checks that the frame fits and recomputes the checksum of each payload.

- **Single event file:** `list`, `get` and `copy-all` fail on a corrupt frame or an incomplete tail. No single-file
  salvage option is available.

- **Mount or directory source:** a bad frame stops that file at the last verified frame. Other source files remain
  readable. An incomplete live tail is retried after the file changes.

- A verified frame whose payload cannot be decoded still appears under `raw`. Concatenating the frames of a stable,
  fully verified source in filename order reproduces the entire event file.

- During typed indexing, tboardfs logs one warning for a value it cannot convert and keeps the other values of the
  event.

  Examples include invalid hparams, mesh, custom scalar layout, graph, `RunMetadata`, and meta graph records.

- When only a conversion fails, the file with the stored bytes stays, for example `<step>.run_metadata.pb` without its
  `.pbtxt`.

## Measuring Large Runs

The scale test uses 20,000 distinct small images and scalar rows by default, plus a 9 MiB PNG in an event with small
siblings.

Run the larger case to check append, partial tails, truncation, replacement, deletion and open handles:

```bash
TBOARDFS_SCALE_COUNT=100000 uv run --locked pytest -q tests/test_scale.py
```

The benchmark generates one mixed TFRecord file of at least 1 GiB by default.

It interleaves 100,000 scalar/image pairs with bulk RGB images, stereo PCM audio and float32 activation maps.

The file also contains selected real producer records for histograms, distributions, PR curves, graphs, run metadata,
text, custom scalars, HParams, meshes, point clouds, Open3D manifests, encoded media, and the full dtype fixture.

Projector embeddings, Open3D assets, profiles and other sidecars are checked in their native layouts; they do not count
toward the event file size.

Generate the file once at a new path and use a fresh process for each trial:

```bash
uv run --locked python -m scripts.benchmark_scale --prepare --data /tmp/scale.tfevents
uv run --locked python -m scripts.benchmark_scale --data /tmp/scale.tfevents
uv run --locked python -m scripts.benchmark_scale --data /tmp/scale.tfevents \
    --profile /tmp/cold-listing.pstats --profile-phase cold_listing
uv run --locked python -m pstats /tmp/cold-listing.pstats
```

The benchmark measures filesystem callbacks, not kernel FUSE throughput or cold-disk latency.

Enter `sort cumulative` and `stats 25` in `pstats` to see where the selected operation spends time.

Use runs **without** `--profile` for timing comparisons: deterministic call tracing adds significant overhead.

The JSON output reports callback timings and peak process RSS; input generation, copying and result checks are not
timed.

Compare fresh serial and parallel indexes on the same file:

```bash
uv run --locked python -m scripts.benchmark_index_workers /tmp/scale.tfevents \
    --workers 1 2 4 8 --repeat 3 --compare-workers 4
```

The comparison times three worker result representations and separates record discovery, result decoding and index
merging.

It does not include FUSE mount startup or virtual directory construction.

To benchmark a slower cold open, prepare a distinct mixed file with 450,000 small scalar/image pairs and 4 GiB of total
event data in a directory on the storage device you want to measure:

```bash
uv run --locked python -m scripts.benchmark_scale --prepare --count 450000 \
    --target-bytes 4294967296 --data ./events.out.tfevents.large
```

Run the full-size integration test with
`TBOARDFS_MIXED_SCALE_BYTES=1073741824 uv run --locked pytest -q tests/test_mixed_scale.py`.

Ordinary tests use a smaller mixed file with the same data families.

Use `--target-bytes` to change the minimum generated size or `--narrow` to reproduce the earlier scalar/image-only
benchmark.

### Cold mounts through Linux FUSE

Keep the established 4 GiB mixed and Lightning-generated logs on the storage device you want to measure, outside the
checkout.

Under the default corpus root `~/workspace/benchmarks/tboardfs/cold-mount`, put only the named run directory in each
`log/` directory and only one complete, unchanged `events.out.tfevents.*` file in each run directory:

```text
cold-mount/
├── mixed/log/run/
└── lightning/log/lightning-4g/
```

Run each workflow separately on a host with Linux FUSE, `/dev/fuse` access and `fusermount3` or `fusermount`:

```bash
OPENBLAS_NUM_THREADS=1 uv run --locked python -m scripts.benchmark_cold_mount mixed > mixed.jsonl
OPENBLAS_NUM_THREADS=1 uv run --locked python -m scripts.benchmark_cold_mount lightning > lightning.jsonl
```

Use `--corpus-root` if the corpus is on another device.

Each sample launches the real foreground `tboardfs` command with a fresh mount and empty persistent index cache in a
private disposable directory beside `log/` on the corpus filesystem.

The benchmark rejects changed inputs, samples with resident source pages before launch, and samples that fail to publish
a persistent index.

Use local files you own with writes already flushed; this does not clear filesystem metadata or device-level caches.

On failure, the benchmark reports the retained trial directory and its `mount.log`; confirm the mount is detached before
removing the directory.

`elapsed_ns` runs from process launch through the first indexed run listing.

The JSON lines also separate mount readiness, root listing and run listing.

The defaults are four rounds with one and four index workers, reversing worker order each round.

For sources of at least 8 MiB, a sample with more than one requested worker must report successful parallel indexing;
`parallel_ranges` is the count reported by the runtime and is zero for serial samples.

Worker counts do not restrict CPU affinity; use `taskset` with one CPU and four CPUs respectively when measuring
one-core versus four-core scaling.

For a revision comparison, keep the machine, corpus and settings fixed, alternate revision order across rounds
(`--repeat 1` per invocation, with `--launcher` pointing to each revision's installed `tboardfs`), and discard the first
round for each revision.

Report the retained samples, median and sample standard deviation for each workflow and worker count.

Never pool the mixed and Lightning results; when alternatives are otherwise tied, favor the Lightning result.

Set `run_dir` to the absolute path of the run inside the mount, not a symlink.

Run each command on a fresh mount so the listing does not warm the full-read sample:

```bash
find "$run_dir" >/dev/null                # Traversal only (all paths)
scripts/benchmark_find_cat.sh "$run_dir"  # Traversal plus reading regular files to EOF
```

The script uses GNU `find` to read every regular file, including raw and derived exports.

It rejects a symlink as the starting directory and does not follow symlinks inside the tree.

It discards the file contents and exits nonzero if traversal or reading fails.

Use a stable directory tree during each measurement.

### Compact corpora for read comparisons

For shorter repeated FUSE measurements, make a separate event-file prefix for each workflow and reuse it for all
revisions:

```bash
mkdir -p "$work/mixed/log/run" "$work/lightning/log/lightning-4g"
scripts/extract_tfrecord_prefix.py --source "$mixed_event" \
    --output "$work/mixed/log/run/$(basename "$mixed_event")"
scripts/extract_tfrecord_prefix.py --source "$lightning_event" \
    --output "$work/lightning/log/lightning-4g/$(basename "$lightning_event")"
```

- The extractor checks both CRC32C fields for each copied TFRecord and stops before the first frame that would exceed
  the 40 MiB cap (`--max-bytes` changes it).
- With a cap at or beyond EOF, it rejects an incomplete or corrupt tail.
- It publishes the completed output atomically without overwriting any existing path.
- The mixed prefix retains captured producer formats and repeated bulk records; the Lightning prefix retains training
  and validation, JPEG and PNG images, the model graph, and hyperparameters.
- A prefix can end within a training iteration and does not represent the full run's scale or later-step distribution.

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
| `generate_geometry_scene.py` | Batched terrain surfaces and face-free range scans; checks vertex and color order, triangle connectivity and boundary edges in NPZ, OBJ and PLY |
| `generate_embedding_cases.py` | Native TensorBoardX vectors alone, with labels, with a sprite, and with both; repeated observations change position and row order between steps |
| `generate_projector_layouts.py` | A historical plugin-local TSV layout with no root config and a modern config that names binary float32 vectors; both preserve row and label order |
| `generate_media_cases.py` | Legacy media fields and plugin tensors: image channels, static/animated GIF, audio sample widths, and native video containers |
| `generate_raster_cases.py` | PNG color types and sample depths, baseline and progressive JPEG, timed GIF/APNG/WebP pairs and moving sprites with partial frames, blending and disposal |
| `generate_embedded_formats.py` | Standard MIDI notes, two-page big-endian 16-bit TIFF, an embedded Fortran-order NPY matrix in string and byte tensors, and plain ZIP versus numeric NPZ |
| `generate_geotiff_cases.py` | Rasterio writes and reopens classic GeoTIFF and forced BigTIFF with the same projected grid and pixels; both go into legacy and TF2 image layouts |
| `generate_heic_cases.py` | Real HEVC and AV1 still images with distinguishable colors, in legacy and TF2 image layouts |
| `generate_wave64_cases.py` | An independently decoded 440 Hz PCM tone in small RF64 and BW64 containers, in legacy and TF2 audio layouts |
| `generate_office_cases.py` | Standalone DOCX field report, XLSX readings and PPTX briefing in `tests/data/file-types/`; these are file detector tests, not TensorBoard events |
| `generate_native_media.py` | JPEG 2000, raw and boxed JPEG XL, OpenEXR, and distinct playable MPEG Layer II and III tones |
| `generate_media_containers.py` | QOI, Radiance HDR, BMP- and PNG-backed icons, SVG, CAF, MPEG program/transport streams, ASF and container controls |
| `generate_scientific_files.py` | Standalone GLB, ASCII and binary PLY, PDF, HDF5 with and without a user block, Parquet and Arrow IPC file/stream controls |
| `generate_science_extra.py` | Standalone MAT v5, SQLite, NetCDF, Safetensors, compressed measurements and triangle meshes |
| `generate_zip_packages.py` | OpenDocument text, spreadsheet and presentation packages, plus small ZIP64 subtype examples |
| `generate_open3d_cases.py` | Open3D point clouds and triangle meshes with batches, changing colors or topology, and references to the preceding step |
| `generate_open3d_advanced_cases.py` | Open3D's TensorFlow writer: colored lines, bounding boxes with labels and confidences, labeled points and a textured mesh with UV coordinates |
| `generate_dtype_cases.py` | Both encodings of all 20 supported numeric/boolean dtype IDs, binary strings, and raw recovery of unsupported dtype IDs |
| `generate_torch_cases.py` | CPU PyTorch scalar precision and actual `add_tensor` encodings, including records with no faithful NumPy preview |
| `generate_plugin_cases.py` | Unicode text tables, HParams domains, margin/multiline custom charts, projector TSV/sprite/bookmarks, CPU profile and unknown plugin data |
| `generate_auxiliary_cases.py` | Real checkpoint-backed projector config and six Debugger V2 streams beside an ordinary scalar; neither is mistaken for summary data |
| `generate_writer_inputs.py` | TensorBoardX grouped scalar subruns, image grids and boxes, ONNX and OpenVINO inputs converted to `GraphDef` |
| `generate_framing_cases.py` | Length and payload CRC failures, incomplete tails, and an over-limit header; these cases are generated during tests |
| `generate_historical_cases.py` | Release-specific TensorBoard 1.15.0 and 2.0.0 summaries, projector assets, debugger statistics and historical profile protos, without TensorFlow |
| `generate_tf115_native.py`, `generate_tf200_native.py` | Actual TensorFlow 1.15.0 graph/session and 2.0.0 eager writers: image pixels, audio samples, histogram percentiles and numeric tensors |
| `generate_tf1_writer_formats.py`, `generate_tf2_plugin_formats.py` | TF1 graph-mode summaries, real session tracing, TF2 plugin PR curves and opaque tensor values on TensorFlow 2.21.0 |
| `generate_tf115_variants.py`, `generate_tf200_variants.py` | Native historical PR and geometry variants, text and binary tensors, dashboard layouts, checkpoints, watched tensors, timelines and model graphs |
| `generate_tf221_extra.py` | Graph and eager writer variants, including live CPU debugger capture, checkpoint-backed vectors, batched geometry and mixed HParams domains |
| `generate_tensorboardx_matrix.py`, `generate_torch_matrix.py`, `generate_lightning_matrix.py` | Writer-specific geometry, grouped scalars, PR curves, optional projector assets, tensors, image grids and boxes |
| `generate_torch_plugins.py`, `generate_tensorboardx_plugins.py` | Direct writer calls for model metrics, projector sprites, stereo or mono tones, and custom scalar layouts; PyTorch also writes animated GIF and a CPU trace |
| `generate_torch_graph.py` | A CPU PyTorch model graph produced by `add_graph`; checks the linear-to-ReLU path, tensor shapes and tagged run metadata |
| `generate_lightning_plugins.py` | Lightning metrics, PyTorch writer summaries including a mesh, cloud, sprite and animated GIF, and a Lightning CPU profiler capture |
| `generate_legacy_tool_assets.py` | A TensorBoard 1.15 Beholder arrays capture and an example TPU operation hierarchy serialized with the official profile schema, not captured on TPU hardware |
| `generate_legacy_text.py` | TF 1.2 stores two tables by `node_name`, but registers only one with the Text dashboard; a TensorBoard 1.15 reader checks the distinction |
| `generate_encoded_media_cases.py` | Caller-encoded media supplied to multiple event writers; the offline tests reconstruct the exact files without storing duplicate payloads |
| `generate_av_clips.py` | One second of motion and synchronized sound in six containers; the generator decodes both streams before recording its reference measurements |
| `generate_audio_variants.py` | A 440 Hz tone encoded as ADTS AAC, AAC/M4A, ALAC/M4A, float WAV and PCM AIFF; tests check the file type and the decoded signal |

### Lightning Segmentation Preview

Generate a CPU-only preview in a new or empty directory:

```bash
scripts/generate_lightning_preview.py --output samples/lightning-preview
mkdir -p mnt
tboardfs samples/lightning-preview mnt
ls mnt/segmentation/images/val
```

The default run logs two epochs, each with 50 training iterations followed by two validation iterations.

Each iteration logs five 96×96 PNGs, a 100-bin probability histogram, and ten scalars under `train/` or `val/`.

The five images show the rendered step number, a seeded rotation, a 3×3 dilation of the input, the unchanged label, and
the XOR of the dilation and label.

The dilation is a visual baseline.

The histogram and classification metrics use scores from the CNN that Lightning trains on the rotated inputs.

Only validation logs F1 and precision–recall curves.

The run also includes the traced CNN graph, TensorBoard HParams, `hparams.yaml`, and a short description.

Change the workload with `--epochs`, `--train-steps`, and `--val-steps`, or change the sample seed with `--seed`.

For a run of about 4 GiB with the same 50 training and two validation iterations per epoch:

```bash
scripts/generate_lightning_preview.py --epochs 583 --stored-png --synthetic-loss \
    --output samples/lightning-4g
```

`--stored-png` disables PNG compression without changing the pixels.

`--synthetic-loss` reports `(1 + ε) / (step + 1)` as the train and validation loss, with seeded uniform noise `ε`
between −0.1 and 0.1.

The real CNN loss stays under `model_loss` and still drives the optimizer.

### Compatibility Scope

- **Version marker:** `brain.Event:2` identifies the event format, not the TensorBoard or TensorFlow release.
- **Historical releases:** TensorBoard 1.15.0 and 2.0.0 produced identical event bytes for the original `tb115` fixture.
  The `tf115-native` and `tf200-native` fixtures use the corresponding TensorFlow runtimes on CPython 3.7.
- **TF1 compatibility API:** `tf1` and `tf1-writer` use graph mode on TensorFlow 2.21.0, independently of the actual
  1.15.0 runtime.
- **Modern writers:** the pinned format cases use TensorBoard 2.21.0, TensorBoardX 2.6.5 and CPU PyTorch 2.12.0.

To regenerate historical TensorBoard helpers, select the release explicitly:

```bash
uv run --with tensorboard==1.15.0 --script scripts/generate_historical_cases.py --release 1.15.0 -o samples/history-115
uv run --with tensorboard==2.0.0 --script scripts/generate_historical_cases.py --release 2.0.0 -o samples/history-200
```

For the native `generate_tf115_native.py` and `generate_tf200_native.py` writers, pass an existing CPython 3.7 to
`uv run --python`.

The cross-writer media generator takes `--source tests/data/producer-cases`, which supplies previously encoded and
verified images and clips.

Compare the upstream plugin inventories for
[1.15.0](https://github.com/tensorflow/tensorboard/blob/1.15.0/tensorboard/default.py),
[2.0.0](https://github.com/tensorflow/tensorboard/blob/2.0.0/tensorboard/default.py) and
[2.21.0](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/default.py).

Not every dashboard defines a summary type:

- Distributions derive from histogram buckets.
- [Metrics/Time Series](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/plugins/metrics/metrics_plugin.py)
  reads scalar, histogram and image series.
- The historical
  [What-If Tool](https://github.com/tensorflow/tensorboard/blob/1.15.0/tensorboard/plugins/interactive_inference/interactive_inference_plugin.py)
  reads example datasets and returns predictions to the client; those examples are not training events.
- Debugger V2 writes six separate streams rather than training summaries.

High-level encoders and the event format impose different limits:

- `add_video` in PyTorch and TensorBoardX encodes an animated GIF, whereas MP4 requires an explicitly supplied payload.
- TensorBoardX 2.6.5 has no `add_tensor` method; its file-writer fixture tests all 18 numeric and Boolean types in its
  schema in packed and repeated-field form.

How to read the table:

- **Native:** the named writer or its official plugin produced the data.
- **Encoded:** the low-level writer preserved bytes supplied by the caller; this does not mean the writer has that
  encoder.
- **Delegated:** the call went through Lightning's PyTorch writer.
- **Missing:** the named writer can produce this case, but the suite has no test for it through that writer.
- **Not supported:** the pinned writer or plugin has no such API; this is not a missing test.
- A cell can contain both a verified case and a missing variant; an unmarked cell claims only the cases it names.
- Historical encoded-media cases use each release's TensorBoard `RecordWriter`, separately from its native TensorFlow
  examples.

An additional TensorFlow 1.2.0 fixture on CPython 3.6 exercises the older practice of storing a tensor's series name in
`node_name` when `tag` is empty.

TensorBoard 1.15 uses `plugins/tensorboard_text/tensors.json` to select dashboard series; tboardfs exports both the
registered and unregistered string tables.

| Data and test coverage | TF/TB 1.15.0 | TF/TB 2.0.0 | TF 2.21.0 graph mode | TF eager / standalone TB 2.21.0 | TensorBoardX 2.6.5 | PyTorch 2.12.0 CPU | Lightning 2.6.6 |
| -- | -- | -- | -- | -- | -- | -- | -- |
| Scalars | Native | Native | Native | Native | Grouped subruns | Grouped subruns and double precision | Training metrics; delegated grouped subruns and double precision |
| Histograms | Classic buckets | Tensor buckets | Classic buckets | Tensor buckets | Samples and raw buckets | Samples and raw buckets | Delegated model weights and raw buckets |
| Distributions | Derived percentiles | Derived percentiles | Derived percentiles | Derived percentiles | Derived percentiles | Derived percentiles | Derived percentiles |
| PR curves | Native weighted, unweighted and precomputed | Native weighted, unweighted and precomputed | Weighted, unweighted and precomputed | Weighted, unweighted and precomputed | Weighted, unweighted and precomputed | Weighted, unweighted and precomputed | Delegated weighted, unweighted and precomputed |
| Meshes and point clouds | Colored and uncolored; batched surface and cloud | Colored and uncolored; batched surface and cloud | Batched surfaces and clouds with and without colors | Batched surfaces and clouds; optional faces and colors | Batched uncolored surfaces and colored clouds; also colored surface and uncolored cloud | Colored surfaces and clouds; batched uncolored surfaces and clouds | Batched colored surfaces and uncolored clouds; also uncolored surface and colored cloud |
| Embeddings | TSVs, sprite, bookmarks and checkpoint config | TSVs, sprite, bookmarks and checkpoint config | TSVs, sprite, bookmarks and checkpoint config | TSVs, binary vectors, sprite, bookmarks and checkpoint config | Multiple embeddings; labels and sprites independently optional | Multiple embeddings; labels and sprites independently optional | Delegated multiple embeddings; labels and sprites independently optional |
| Text | Native vector and rank 3; wire table | Native vector and rank 3; wire table | Native ranks 1–3 | Unicode table and native rank 3 | Native text | Native text | Delegated Markdown |
| HParams | Native mixed domains and session end | Native mixed domains and session end | Experiment, trial, end and mixed domains | Experiment, trial, end and mixed domains | Trial and metric | Child run with explicit discrete domain and metric | Metrics, YAML and delegated child run with explicit discrete domain |
| Custom scalar layouts | Margin and multiline | Margin and multiline | Margin and multiline | Margin and multiline | Margin and multiline | Margin and multiline | Delegated margin and multiline |
| Graphs | Session graph and Sequential model summary | Eager trace, traced `tf.function` and Sequential model summary | Session graph and meta graph | Function and Keras graphs | Model, ONNX and OpenVINO imports | Traced model | Logger delegates model graph |
| Still images | Native PNG; encoded alternatives | Native PNG; encoded alternatives | Native PNG; encoded alternatives | Batched PNG; encoded alternatives | PNG grids and boxes; encoded alternatives | PNG grids and boxes; encoded alternatives | Delegated PNG grids and boxes; encoded alternatives |
| Animation | Encoded | Encoded | Encoded | Encoded | Native GIF; encoded alternatives | Native GIF; encoded alternatives | Delegated GIF; encoded alternatives |
| Audio | WAV; encoded alternatives | WAV; encoded alternatives | WAV; encoded alternatives | Batched WAV; encoded alternatives | Stereo WAV; encoded alternatives | Mono WAV; encoded alternatives | Delegated WAV; encoded alternatives |
| Video containers | Encoded | Encoded | Encoded | Encoded | Encoded | Encoded | Delegated encoded |
| Extended encoded media (cases below) | Encoded | Encoded | Encoded | Encoded through both writers | Encoded | Encoded | Delegated encoded |
| Generic tensors | Native float32, float64, complex64 and binary strings; wider wire types | Native float32, float64, complex64 and binary strings; wider wire types | Matrices, float64, complex64 and binary strings | Numeric including float64 and complex64, binary strings, resource and variant; full wire matrix | File writer: 18 numeric/Boolean wire types, UTF-8 and binary strings, resource | 12 native dtypes, three tested as raw-only | Delegated float64, complex64 and raw-only float16 |
| Profiler | CPU session trace and v1 timeline JSON | CPU `.trace` and `compat.v1` timeline JSON | Timed session and v1 timeline | CPU XPlane | Not supported by its writer | CPU trace excerpt | Native `PyTorchProfiler` |
| TensorFlow Debugger | Native watched tensor and wire numeric summary | Native `compat.v1` watched tensor and wire numeric summary | Native `tfdbg` tensor dump | Six Debugger V2 streams; automatic CPU `Square` capture | Not supported by its writer | Not supported by its writer | Not supported by its logger |
| Beholder | Native arrays capture and section metadata | Native graph-mode arrays capture and section metadata | Not included in TB 2.21 | Not included in TB 2.21 | Not supported by its writer | Not supported by its writer | Not supported by its logger |

Open3D is a separate plugin, not a TensorBoard mesh encoding.

Its [PyTorch fixture](tests/test_open3d_plugin.py) covers repeated steps and two geometry items per batch; its
[TensorFlow fixture](tests/test_open3d_advanced.py) covers line sets, instance detections, point labels and a textured
mesh.

The [cross-writer media tests](tests/test_encoded_media_writers.py) cover eight low-level writer paths, including both
TF eager and standalone TensorBoard 2.21.

All eight preserve still images, two-frame GIF/APNG/WebP, five audio formats and six silent video containers.

The **extended encoded media** row adds 82 source values on each writer path.

The suite reconstructs each writer's original TFRecord from shared payload bytes and checks it against the hash of the
independently generated file before testing its exported paths and bytes:

- [Still images](tests/test_raster_formats.py): the full PNG color/depth and baseline/progressive JPEG sets, including
  CMYK, plus [single-frame GIF routing](tests/test_media_formats.py).
- [Animation](tests/test_raster_formats.py): moving partial frames with GIF disposal, APNG blending and lossless/lossy
  animated WebP.
- [Audio](tests/test_audio_variants.py): ADTS AAC, AAC and ALAC in M4A, PCM AIFF, float WAV and the
  [full integer PCM sample-width set](tests/test_media_formats.py).
- [Video](tests/test_av_clips.py): synchronized sound and motion in all six containers, not just the silent clips.
- [Geospatial TIFF](tests/test_geotiff_cases.py), [HEIC and AVIF](tests/test_heic_cases.py), and
  [RF64 and BW64](tests/test_wave64_cases.py): both the legacy media fields and TF2 plugin tensor layouts.
- [Native raster and audio formats](tests/test_native_media.py): JPEG 2000, JPEG XL, OpenEXR and MPEG Layer II/III.
- [Additional containers](tests/test_media_containers.py): QOI, HDR, ICO, SVG, CAF, program/transport streams, ASF and
  controls that distinguish a codec from its container.

Reader behavior and native producer coverage are tested separately:

- The [historical mesh API](https://github.com/tensorflow/tensorboard/blob/1.15.0/tensorboard/plugins/mesh/summary.py)
  accepts leading dimensions; the [batched geometry test](tests/test_geometry_scene.py) checks vertex order and triangle
  indices across two objects.
- The [projector tests](tests/test_embedding_cases.py) check that vector rows still match their labels and sprite tiles
  when observations change order.
- The [historical](tests/test_historical_variants.py) and [modern graph](tests/test_tf221_extra.py) checkpoint fixtures
  check the Projector config and its visible assets; the filesystem does not extract checkpoint vectors.
- The [rank-3 text test](tests/test_text_tables.py) uses a constructed event: the Markdown preview shows one plane, but
  JSON retains all planes.
- The [dtype tests](tests/test_torch_formats.py) check byte-exact recovery of noncanonical PyTorch values; a raw-only
  result is not a missing test.
- The [binary string fixture](tests/test_tf2_plugin_formats.py) contains embedded NUL and invalid UTF-8 bytes, not just
  non-ASCII text.
- The [historical PR API](https://github.com/tensorflow/tensorboard/blob/1.15.0/tensorboard/plugins/pr_curve/summary.py)
  accepts predictions or precomputed counts; supplying an equivalent tensor does not exercise either producer call.

CPU profiles do not establish accelerator coverage: a real TPU operation profile requires a TPU capture, not the
[schema-generated example](tests/test_legacy_tool_assets.py).

For video with synchronized sound, the fixture uses H.264/AAC in MP4 and MOV, MPEG-4/PCM in AVI, VP9/Opus in WebM,
H.264/FLAC in Matroska and Theora/Vorbis in Ogg.

To export examples you can play or display, use new output filenames:

```bash
tboardfs-file get tests/data/producer-cases/av-clips /videos/mp4/000008.mp4 -o motion.mp4
tboardfs-file get tests/data/producer-cases/rasters /images/moving-apng/000001.png -o motion.png
tboardfs-file get tests/data/producer-cases/tensorboardx-plugins /audio/interval/000003.wav -o interval.wav
tboardfs-file get tests/data/producer-cases/geometry-scene /meshes/terrain/000012.ply -o terrain.ply
```

The MP4 contains a moving square and a tone that changes pitch at the midpoint; the WAV contains 220 Hz in the left
channel and 330 Hz in the right.

The suite checks decoded content and byte preservation, not playback in a specific browser.

| Media in a verified event | Filesystem result |
| -- | -- |
| PNG (gray, indexed, gray + alpha, RGB, RGBA), baseline/progressive JPEG, BMP, WebP | `images/`, with the detected extension |
| Animated PNG and WebP | `images/<tag>/<step>.png` or `.webp`, including all frames |
| Static and animated GIF, including an `images` plugin tensor | `videos/<tag>/<step>.gif`, with the original frames |
| WAV (8-, 16-, 24-, 32-bit PCM and float32), MP3, FLAC, Ogg/Vorbis, WebM/Opus, ADTS AAC, AAC or ALAC in M4A, PCM AIFF | `audio/`, with the detected extension |
| MP4, QuickTime, AVI, WebM, Matroska, Ogg/Theora in a custom `video` summary | `videos/`, with `.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`, `.ogg` respectively |
| TIFF and AVIF images; RIFF-wrapped MIDI audio | Original bytes as `.tiff`, `.avif` and `.rmi` across all eight low-level writer paths; these previously used `.bin` |
| Classic GeoTIFF and GeoTIFF in BigTIFF (standalone TB 2.21.0, legacy image and TF2 image tensor) | Both `.tiff`; the 3 × 2 raster retains its uint16 samples, 30 m grid and EPSG:32633 CRS |
| HEIC and AVIF (standalone TB 2.21.0, legacy image and TF2 image tensor) | `.heic` for HEVC, `.avif` for AV1, based on explicit major or compatible brands; generic HEIF alone gives `.heif` and conflicting codec brands give `.bin` |
| RF64 and BW64 (standalone TB 2.21.0, legacy audio and TF2 audio tensor) | Both `.wav`; the 64-bit size chunk and all 800 original PCM samples survive export |
| Standard MIDI without a RIFF wrapper | `audio/<tag>/<step>.mid`; the note sequence is preserved |
| Embedded NumPy array | `.npy` for an eligible binary tensor payload; dtype, byte order and storage order are unchanged |
| Embedded ZIP and NumPy archive (standalone TB 2.21.0, binary string tensor) | `.npz` when every member has an NPY name and a recognized NPY header; plain ZIP stays `.zip` |
| JPEG 2000 (JP2 and codestream), JPEG XL (boxed and codestream), OpenEXR, QOI, Radiance HDR, ICO and SVG | Original image files as `.jp2`, `.j2k`, `.jxl`, `.exr`, `.qoi`, `.hdr`, `.ico` and `.svg` |
| MPEG Layer II and III, Core Audio Format, Opus in Ogg and WebM | `.mp2`, `.mp3`, `.caf`, `.ogg` and `.webm`; the suffix identifies the container when it differs from the codec |
| MPEG program and transport streams, ASF with WMV video | `.mpg`, `.ts` and `.asf`; an elementary MPEG video stream does not become a program stream |

**Standalone file detection is not dashboard support.**

| Fixture family | Detected suffix or intentional fallback |
| -- | -- |
| Office document, spreadsheet and presentation | `.docx`, `.xlsx`, `.pptx` when the manifest, main part and root relationship agree |
| OpenDocument text, spreadsheet and presentation | `.odt`, `.ods`, `.odp` when the first stored MIME member, manifest and content agree |
| glTF binary, PLY (ASCII and both binary byte orders), binary STL, vertex/face OBJ | `.glb`, `.ply`, `.stl`, `.obj`; ambiguous ASCII STL and OBJ syntax outside the tested subset stay text |
| PDF, HDF5, NetCDF, MAT v5, SQLite, Safetensors | `.pdf`, `.h5`, `.nc`, `.mat`, `.sqlite`, `.safetensors`; NetCDF-4 uses HDF5 and stays `.h5` |
| Parquet and Arrow IPC file | `.parquet`, `.arrow`; an IPC stream lacks the file footer and stays `.bin` |
| gzip, bzip2 and XZ | `.gz`, `.bz2`, `.xz`; the detector does not decompress the payload to guess a second suffix |

ZIP inspection does not extract files, load arrays or open documents.

The detector checks at most 128 entries and a 64 KiB central directory, and reads only bounded headers or metadata.

Canonical single-disk ZIP64 archives use the same subtype checks and limits as ordinary ZIP archives.

Unsupported ZIP64 layouts stay `.zip` when the detector recognizes the enclosing archive.

A recognized ZIP without enough evidence for a subtype stays `.zip`; a detected broken ZIP header gives `.bin`.

The extension is not a full integrity check.

The [TF2 image writer](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/plugins/image/summary_v2.py)
emits PNG and the
[TF2 audio writer](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/plugins/audio/summary_v2.py) emits
WAV.

TensorBoard's
[image MIME detector](https://github.com/tensorflow/tensorboard/blob/2.21.0/tensorboard/util/img_mime_type_detector.py)
recognizes BMP, GIF, JPEG and PNG; the other containers test filesystem export, not dashboard playback.

To compare extensions for an event file, run the development-only [detector comparison](scripts/compare_file_types.py):

```bash
scripts/compare_file_types.py tests/data/producer-cases/audio-variants/events.out.tfevents.audio-variants
```

It prints one JSON object per indexed binary item except graphs, whose extensions are set by the plugin schema.
`tboardfs` is the export extension; `puremagic` lists candidate extensions in rank order without using a filename hint.
The script does not change the event file or add `puremagic` to the project dependencies.

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

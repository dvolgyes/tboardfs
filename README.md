# tboardfs

Mount and inspect TensorBoard event logs as ordinary files.

`tboardfs` has two access modes:

- Filesystem mode mounts a log directory with FUSE.
- Command mode inspects or copies the virtual content of one event file.

Both modes use the same virtual paths for TensorBoard tabs such as `scalars`,
`images`, `tensors`, `meshes`, `hparams`, and `custom_scalars`.

## Filesystem Mode

Mount a TensorBoard log directory:

```bash
tboardfs SOURCE MOUNTPOINT
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

Copy the full virtual tree:

```bash
tboardfs-file copy-all events.out.tfevents.123 exported-event
```

Command mode exposes tab directories directly at the root, for example
`/scalars`, `/images`, and `/meshes`. It accepts a single event file, omits FUSE
control files, and does not include sibling sidecars.

`get` refuses to overwrite existing files unless `--force` is provided.
`copy-all` checks for existing target conflicts before writing, and also
requires `--force` to overwrite.

## Output Rules

`tboardfs-file list` prints virtual file paths to stdout, one per line.
`tboardfs-file get -o -` writes raw bytes to stdout. Status, warnings, and
errors go to stderr through Loguru/Click so stdout remains safe for pipelines.

## License

MIT

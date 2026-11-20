# Task: reliable live TensorBoard exports in tboardfs

**Target:** the maintained tboardfs repository, not the local 0.2.3 snapshot used for the review. Port the requirements
and regression cases, not the prototype patch. Start by recording the target branch and commit, reading its contributor
instructions, and running its baseline checks. For each item below, distinguish a reproduced defect from an
already-correct behavior or an intentional difference in the maintained version. The snapshot had no Git metadata, and
its last implementation edits were not verified as a set.

## User decisions to preserve

- Keep one distribution with two installed commands and no extras: `tboardfs` mounts a directory of runs;
  `tboardfs-file` lists, reads, or copies files from a single event file without mount privileges.
- Keep the mount kernel-enforced read-only; refresh automatically instead of requiring writable control files or symlink
  aliases.
- Offer both human-usable typed exports and a lossless view of the original records, including records with unknown
  fields or no typed representation.
- Validate both masked CRC32C checksums by default, bound record allocation, and distinguish an incomplete live tail
  from corruption.
- Preserve names with reversible component encoding and deterministic collision handling rather than silently dropping
  or renaming data unpredictably.
- Optimize first for many independent runs, then for tens of thousands of small files (typically below 1 MB), then for
  large individual files. A 2 GB cache was an example, not a required default.

## Work and acceptance criteria

### 1. Refresh only the active run

Check known event files in the accessed run by identity, size, and modification time on access. Resume ordinary
append-only growth at the last **complete** frame; do not advance the cursor to a size observed after parsing. Reindex a
replacement, even at the same size, and drop deleted files without waiting for directory discovery. Discover new event
files on access at an interval of about 2–3 minutes, in the active directory and at most its immediate parent; do not
walk the whole source tree. Track projector/profile sidecars independently of event changes and give each sidecar one
owning run.

**Accept when:** cold and warm views agree after partial append, tail completion, a final append at EOF, equal-size
replacement, truncation, and deletion. A new event or sidecar appears after the discovery interval without touching an
existing event file. An empty source does not expose the process working directory, and browsing one run does not parse
idle siblings.

### 2. Separate exact records from typed previews

Index the byte offsets and lengths of validated TFRecord frames. A proposed public path is
`raw/<encoded-event-basename>/<20-digit-decimal-byte-offset>.tfrecord` under each run. Keep the length header, both
checksums, and the exact payload bytes; reserializing a protobuf is not equivalent. On a live mount, expose the
validated prefix and retry an incomplete tail after growth. Report corruption with the source and offset; do not let one
corrupt file hide healthy files or runs. For a single-file command, reject corrupt or incomplete input with a nonzero
status and a useful diagnostic.

**Accept when:** concatenating a stable source's raw exports in offset order reproduces the source byte for byte,
including unknown records. Test bad length and payload checksums separately, an oversized frame, and a valid frame whose
optional plugin payload cannot be decoded. On an open file, successive reads must not silently mix generations: return
the original bytes or an explicit stale-source error. Include an in-place rewrite of an indexed frame followed by growth
on the same inode, even if the writer recomputes the frame CRC; checking only that the source got larger is
insufficient. Document the append-only assumption for incremental typed tables and how to replace a source when
rewriting history. CRC32C detects corruption; it does not establish authenticity.

### 3. Make typed data match what the producer wrote

| Case | Observable check |
| -- | -- |
| Steps | An omitted summary step is zero; steps above `2**53` remain exact after append. |
| Time | Cold and warm relative times have the same origin; a missing timestamp does not break kernel `stat`. |
| Metadata | Later values inherit the series metadata when their producer omits it. |
| Tensors | Packed and repeated encodings of the same dtype and shape agree, including vectors that must not become scalars. |
| Unsupported tensors | Bound shape expansion and preserve the raw record if a typed representation is not available. |
| Histograms | Both legacy buckets and v2 `[left, right, count]` tensors produce step files containing only that step. |
| Distributions | Compare percentiles and endpoints with a hand-calculated example containing two distinct observations at the same step and no epoch. |
| Images and audio | Compare every encoded sample byte for byte in producer order, excluding dimension fields and labels, for both legacy and v2 summaries. |
| Other plugins | Check PR-curve threshold orientation, HParams-declared metrics, batched mesh face indices, and preservation of unknown binary plugin metadata. |
| Plugin failures | An unsupported preview must not hide a healthy sibling or the original raw record. |

Use producer-generated fixtures for ordinary formats and small controlled frames for boundary cases. Compare mount and
single-file bytes for the same stable event file, excluding mount-only sidecars.

### 4. Separate virtual naming from host I/O

Share one virtual naming policy across mount, list, get, and copy. Preserve nested tags, empty components, literal
escape characters, dot components, and runs named after virtual tabs. Resolve file/directory conflicts deterministically
in either insertion order; for example, scalar tags `x` and `x.json/y` must both remain accessible. One candidate scheme
uses percent escapes, `~empty` for an empty component, and `~1`, `~2`, … before the extension for leaf collisions.

Independently enforce the selected output root at the host writer and the selected source root at the event/sidecar
reader. Do not follow symlink components or overwrite through a link, even with `--force`. Publish complete files and
preserve the distinction between fail, overwrite, and skip-existing policies. Keep data on stdout and diagnostics on
stderr; invalid formats, missing prefixes, and output conflicts must not produce tracebacks.

**Accept when:** the same unusual tags can be listed, read, and copied without loss or escape; a protected existing file
remains unchanged on refusal; a copy conflict reports partial progress accurately. Exercise these checks in disposable
directories.

### 5. Measure work and verify the installed workflows

Measure a cold root listing, one changing run among idle runs, and a late range read among tens of thousands of small
exports. Record files parsed, bytes read or rendered, memory, and elapsed time for the same workload before and after
the change. Prefer behavioral evidence over cache-size targets: a root listing should not parse event files in child
runs, and chunked reads of one generated file should not serialize the entire file for every chunk. Use compact offsets
and bounded caching where the measurements justify them.

Build a wheel and exercise both installed entry points outside the source tree. On a real disposable Linux FUSE mount,
check kernel read-only enforcement, append visibility, seek/read behavior, and clean unmount. Document the actual Python
and native library requirements separately from mount permissions, plus mountpoint creation, discovery timing, unmount,
and a single-file fallback example. Run the maintained repository's test suite and `pre-commit run --all` before
committing there.

## Evidence to reproduce, not current-repository claims

In the original review, the snapshot's 54 existing tests passed despite these observed failures: an equal-size
replacement served the old scalar, two histogram step files contained the same rows, and an altered payload checksum did
not change the exported paths or bytes. One `getattr` followed by eight short reads caused nine full serializations of a
366-byte scalar export. These are useful regression oracles, not evidence that the maintained repository still has the
same defects.

If the real repository exposes an incompatible public contract, collect the material conflicts before changing it.
Present them together, with at least three concrete alternatives per question and the compatibility or performance
consequences of each.

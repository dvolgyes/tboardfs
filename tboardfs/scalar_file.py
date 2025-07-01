"""ScalarFile class for handling scalar data files with automatic sorting."""

from pathlib import Path


class ScalarFile:
    """A file handler for scalar data that buffers writes and sorts on close."""

    def __init__(self, file_path: Path | str) -> None:
        """Initialize ScalarFile with a file path."""
        self.file_path = Path(file_path)
        self._buffer: list[tuple[int, float]] = []
        self._is_open = False
        self._closed = False

    def open(self) -> None:
        """Open the file for writing (creates in-memory buffer)."""
        if not self._closed:
            self._is_open = True

    def append(self, step: int, value: float) -> None:
        """Append a scalar data point."""
        if self._closed:
            return

        if not self._is_open:
            self.open()

        self._buffer.append((step, value))

    def sort(self) -> None:
        """Sort the buffer by step number."""
        if not self._closed:
            self._buffer.sort(key=lambda x: x[0])

    def close(self) -> None:
        """Sort buffer and write to file, then close."""
        if self._closed:
            return

        self.sort()

        if self._buffer:
            # Ensure parent directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write sorted data to file
            with self.file_path.open("w") as f:
                for step, value in self._buffer:
                    f.write(f"{step}\t{value}\n")

        self._closed = True
        self._is_open = False

    def __del__(self) -> None:
        """Ensure file is closed and sorted on destruction."""
        if not self._closed:
            self.sort()
            self.close()

    def __enter__(self) -> "ScalarFile":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: object
    ) -> None:
        """Context manager exit."""
        self.close()

import logging
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class BatchBuffer:
    def __init__(self, schema: pa.Schema, max_size_bytes: int | None):
        self.schema = schema
        self.max_size_bytes = max_size_bytes
        self._batches: list[pa.RecordBatch] = []
        self._current_size: int = 0

    def add(self, batch: pa.RecordBatch) -> None:
        self._batches.append(batch)
        self._current_size += batch.nbytes

    def is_full(self) -> bool:
        if self.max_size_bytes is None:
            return False
        return self._current_size >= self.max_size_bytes

    def to_table(self) -> pa.Table:
        if not self._batches:
            return pa.Table.from_batches([], schema=self.schema)
        return pa.Table.from_batches(self._batches, schema=self.schema)

    def clear(self) -> None:
        self._batches.clear()
        self._current_size = 0

    def __bool__(self) -> bool:
        return bool(self._batches)

    @property
    def current_size(self) -> int:
        return self._current_size


class ParquetStreamWriter:
    """
    A writer for writing streaming data to Parquet files with automatic file rollover.

    This class manages writing large or infinite datasets to multiple Parquet files
    (shards), automatically creating new files when a size threshold is reached.

    Parameters
    ----------
    path : str or Path
        Path where Parquet files will be written. If shard_size_bytes is None,
        this is the path to the single output file. If shard_size_bytes is set,
        this is the base directory where shards will be created.
    schema : pa.Schema
        PyArrow schema defining the structure of the data to be written.
    shard_size_bytes : int or None, default None
        Approximate maximum uncompressed memory size in bytes for each shard
        before starting to write to a new file. If None (default), sharding is
        disabled and a single file is written to path. If set to an integer,
        path is treated as a base directory and shards are created inside it.
    row_group_size : int or None, default None
        Maximum number of rows in written row group.
    buffer_size_bytes : int, default 16_777_216
        Maximum size in bytes of the in-memory buffer before flushing to disk.
        Must be <= shard_size_bytes.
    file_prefix : str or None, default None
        Prefix to use for generated filenames (only used when sharding is
        enabled). If None (default), the value of `path` will be used as the
        prefix and files will be named '{file_prefix}-{index}.parquet'.
    overwrite : bool, default False
        If True, deletes existing output file or directory before writing.
        If False, raises FileExistsError when the output exists.
        Default is False.
    **kwargs : dict, optional
        Additional keyword arguments passed to pyarrow.parquet.ParquetWriter.

    Attributes
    ----------
    path : Path
        The output path.
    schema : pa.Schema
        The PyArrow schema for the data.
    shard_size_bytes : int or None
        Maximum uncompressed size threshold for each file.
    row_group_size : int or None
        Maximum number of rows in written row group.
    buffer_size_bytes : int or None
        Maximum size of in-memory buffer before flushing.
    file_prefix : str
        Prefix used for naming files if sharding is enabled.
    writer : pq.ParquetWriter or None
        Current active Parquet writer instance.
    written_files : list[Path]
        List of absolute paths to all successfully created Parquet files.

    Methods
    -------
    write_batch
        Write a data batch to the output.
    flush
        Flush buffered data to the current shard.

    Raises
    ------
    FileExistsError
        If the output path already exists and overwrite is False.
    FileNotFoundError
        If the parent directory of the output path does not exist.
    ValueError
        If shard_size_bytes or buffer_size_bytes is negative.
    """

    def __init__(
        self,
        path: str | Path,
        schema: pa.Schema,
        shard_size_bytes: int | None = None,
        buffer_size_bytes: int = 16_777_216,
        file_prefix: str | None = None,
        row_group_size: int | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        # Validate size parameters
        if shard_size_bytes is not None and shard_size_bytes < 0:
            raise ValueError("shard_size_bytes must be non-negative")
        if buffer_size_bytes is not None and buffer_size_bytes < 0:
            raise ValueError("buffer_size_bytes must be non-negative")

        self.path: Path = Path(path)
        self.schema: pa.Schema = schema
        self.shard_size_bytes: int | None = shard_size_bytes
        self.buffer_size_bytes: int = buffer_size_bytes
        self.file_prefix: str = (
            file_prefix if file_prefix is not None else self.path.name
        )
        self.row_group_size: int | None = row_group_size
        self._writer_kwargs: dict = kwargs
        self.writer: pq.ParquetWriter | None = None
        self.written_files: list[Path] = []
        self._current_size: int = 0
        self._current_shard_index: int = 0
        self._current_shard_path: Path | None = None

        # Create a buffer for incoming data
        self._buffer = BatchBuffer(schema, self.buffer_size_bytes)

        # Handle existing output
        if self.path.exists():
            if overwrite:
                if self.path.is_file():
                    self.path.unlink()
                    logger.info(f"Removed existing file '{self.path}'")
                else:
                    shutil.rmtree(self.path)
                    logger.info(f"Removed existing directory '{self.path}'")
            else:
                raise FileExistsError(f"'{self.path}' already exists.")

        # Raise error if parent directory doesn't exist
        if not self.path.parent.exists():
            raise FileNotFoundError(f"'{self.path.parent}' does not exist.")

        # If sharding is enabled, create the output directory
        if self.shard_size_bytes is not None:
            self.path.mkdir(parents=False, exist_ok=False)

    def __enter__(self) -> "ParquetStreamWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _open_new_shard(self) -> pq.ParquetWriter:
        if self.writer:
            self.writer.close()

        if self.shard_size_bytes is None:
            self._current_shard_path = self.path
        else:
            current_shard_name = (
                f"{self.file_prefix}-{self._current_shard_index}.parquet"
            )
            self._current_shard_path = self.path / current_shard_name
            self._current_shard_index += 1

        logger.info(f"Opening file '{self._current_shard_path}'")

        self.writer = pq.ParquetWriter(
            str(self._current_shard_path),
            self.schema,
            **self._writer_kwargs,
        )
        self._current_size = 0
        self.written_files.append(self._current_shard_path.absolute())
        return self.writer

    def _is_shard_full(self) -> bool:
        if self.shard_size_bytes is None:
            return False
        return self._current_size > self.shard_size_bytes

    def _normalize_data(
        self, data: dict | pa.Table | pa.RecordBatch
    ) -> list[pa.RecordBatch]:
        batches: list[pa.RecordBatch] = []
        if isinstance(data, dict):
            # Create RecordBatch directly from dict
            batches.append(pa.RecordBatch.from_pydict(data, schema=self.schema))
        elif isinstance(data, pa.RecordBatch):
            # Cast the RecordBatch to schema
            batches.append(
                pa.Table.from_batches([data]).cast(self.schema).to_batches()[0]
            )
        elif isinstance(data, pa.Table):
            # Convert table to RecordBatches and cast
            batches.extend(data.cast(self.schema).to_batches())
        else:
            raise TypeError(
                "Data must be a dict, pyarrow.Table, or pyarrow.RecordBatch"
            )
        return batches

    def write_batch(self, data: dict | pa.Table | pa.RecordBatch) -> None:
        """
        Write a data batch to the output.

        The data is automatically cast to the schema defined at initialization.
        Data is buffered in memory as RecordBatches. When the buffer exceeds
        the buffer_size_bytes threshold (if set), the buffer is flushed to disk.
        When a shard exceeds shard_size_bytes, a new shard file is created.

        Parameters
        ----------
        data : dict, pa.Table, or pa.RecordBatch
            Data to write. If dict, keys are column names and values are lists.
            If pa.Table or pa.RecordBatch, it will be cast to the schema provided
            at initialization.

        Raises
        ------
        TypeError
            If data is not a dict, pyarrow.Table, or pyarrow.RecordBatch.
        pa.ArrowInvalid
            If the data cannot be cast to the specified schema.
        """
        # Convert input to list of RecordBatches and cast to schema
        batches = self._normalize_data(data)

        # Buffer the batches
        for batch in batches:
            self._buffer.add(batch)

        # Check if we need to flush the buffer (memory management)
        # or if the current shard is full
        if self._buffer.is_full() or self._is_shard_full():
            # If the shard is full and we have already written data to it,
            # rotate to a new shard before flushing the buffer.
            if self._is_shard_full() and self._current_size > 0:
                self._open_new_shard()
            self.flush()

    def flush(self) -> None:
        """
        Flush buffered RecordBatches to the current shard.

        Combines all buffered RecordBatches into a single Table and writes it
        to the Parquet file.

        This method is automatically called when the buffer size exceeds
        buffer_size_bytes, or when shard limits are reached. It can also be called
        manually to force writing buffered data.
        """
        if not self._buffer:
            return

        # Combine buffered batches into a single table (zero-copy)
        table = self._buffer.to_table()
        current_buffer_size = self._buffer.current_size

        # Ensure we have a writer open
        if self.writer is None:
            self._open_new_shard()

        # Write the combined table
        self.writer.write_table(table, self.row_group_size)
        self._current_size += current_buffer_size

        # Clear the buffer
        self._buffer.clear()

    def close(self) -> None:
        # Flush any remaining buffered data
        self.flush()
        # Close the current writer if open
        if self.writer:
            self.writer.close()
            logger.info(f"Closed file '{self._current_shard_path}'")
            self.writer = None
            self._current_shard_path = None

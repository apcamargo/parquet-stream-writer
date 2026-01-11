import logging
import shutil
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ParquetStreamWriter:
    """
    A writer for writing streaming data to Parquet files with automatic file rollover.

    This class manages writing large or infinite datasets to multiple Parquet files (shards),
    automatically creating new files when a size threshold is reached. It supports context
    management for safe resource cleanup.

    Parameters
    ----------
    path : str or Path
        Path where Parquet files will be written. If shard_size_bytes is None, this is the
        path to the single output file. If shard_size_bytes is set, this is the base
        directory where shards will be created. The parent directory must already exist.
    schema : pa.Schema
        PyArrow schema defining the structure of the data to be written.
    shard_size_bytes : int or None, default None
        Approximate maximum uncompressed memory size in bytes for each shard before starting
        to write to a new file. If None (default), sharding is disabled and a single
        file is written to path. If set to an integer, path is treated as a base directory
        and shards are created inside it.
    file_prefix : str or None, default None
        Prefix to use for generated filenames (only used when sharding is enabled).
        If None (default), uses the name of the `path`.
        Files will be named `{file_prefix}-{index}.parquet`.
    row_group_size : int or None, default None
        Number of rows per row group. If None (default), the row group size will be
        either the total number of rows or 1,048,576, whichever is smaller.
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
    file_prefix : str
        Prefix used for naming files (if sharding).
    row_group_size : int or None
        Number of rows per row group.
    writer : pq.ParquetWriter or None
        Current active Parquet writer instance.
    written_files : list[Path]
        List of absolute paths to all successfully created Parquet files.

    Methods
    -------
    write_batch
        Write a data batch to the output.

    Raises
    ------
    FileExistsError
        If the output path already exists and overwrite is False.
    FileNotFoundError
        If the parent directory of the output path does not exist.
    """

    def __init__(
        self,
        path: str | Path,
        schema: pa.Schema,
        shard_size_bytes: int | None = None,
        file_prefix: str | None = None,
        row_group_size: int | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        self.path = Path(path)
        self.schema = schema
        self.shard_size_bytes = shard_size_bytes
        self.file_prefix = file_prefix if file_prefix is not None else self.path.name
        self.row_group_size = row_group_size
        self._writer_kwargs = kwargs
        self.writer: pq.ParquetWriter | None = None
        self.written_files: list[Path] = []
        self._current_size = 0
        self._shard_index = 0

        # Handle existing output
        if self.path.exists():
            if overwrite:
                if self.path.is_file():
                    self.path.unlink()
                    logger.info(f"Removed existing filec '{self.path}'")
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
            shard_path = self.path
        else:
            shard_name = f"{self.file_prefix}-{self._shard_index}.parquet"
            shard_path = self.path / shard_name
            self._shard_index += 1

        logger.info(f"Opening file: {shard_path}")

        self.writer = pq.ParquetWriter(
            str(shard_path),
            self.schema,
            **self._writer_kwargs,
        )
        self._current_size = 0
        self.written_files.append(shard_path.absolute())
        return self.writer

    def write_batch(self, data: dict | pa.Table) -> None:
        """
        Write a data batch to the output.

        The data is automatically cast to the schema defined at initialization.
        If sharding is enabled and the current file exceeds the size threshold,
        a new file is created.

        Parameters
        ----------
        data : dict or pa.Table
            Data to write. If dict, keys are column names and values are lists.
            If pa.Table, it will be cast to the schema provided at initialization.

        Raises
        ------
        TypeError
            If data is not a dict or pyarrow.Table.
        pa.ArrowInvalid
            If the data cannot be cast to the specified schema.
        """
        if isinstance(data, dict):
            table = pa.Table.from_pydict(data)
            table = table.cast(self.schema)
        elif isinstance(data, pa.Table):
            # Ensure the table matches the schema
            table = data.cast(self.schema)
        else:
            raise TypeError("Data must be a dict or pyarrow.Table")

        batch_size = table.nbytes

        if self.writer is None:
            writer = self._open_new_shard()
        elif (
            self.shard_size_bytes is not None
            and self._current_size + batch_size > self.shard_size_bytes
        ):
            writer = self._open_new_shard()
        else:
            writer = self.writer

        writer.write_table(table, self.row_group_size)
        self._current_size += batch_size

    def close(self) -> None:
        if self.writer:
            self.writer.close()
            self.writer = None
            logger.info("Closed current file writer.")

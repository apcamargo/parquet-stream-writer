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
    base_path : str or Path
        Directory path where Parquet files will be written.
    schema : pa.Schema
        PyArrow schema defining the structure of the data to be written.
    shard_size_bytes : int, default 5_368_709_120
        Approximate maximum uncompressed memory size in bytes for each file before rolling
        over to a new file. Note that the actual file size on disk will likely be smaller
        due to compression. Default is 5,368,709,120 bytes (5 GiB).
    file_prefix : str, default "shard"
        Prefix to use for generated filenames. Files will be named `{file_prefix}-{index}.parquet`.
    compression : {'snappy', 'gzip', 'brotli', 'zstd', 'lz4', 'none'} or None, default 'zstd'
        Compression codec to use. Can be a string specifying the codec for all columns,
        or None for no compression. Default is 'zstd'.
    row_group_size : int or None, default 10_000
        Number of rows per row group. If None, uses PyArrow's default behavior.
        Default is 10,000.
    overwrite : bool, default False
        If True, removes and recreates the output directory if it already exists.
        If False, raises FileExistsError when the directory exists.
        Default is False.
    **kwargs : dict, optional
        Additional keyword arguments passed to pyarrow.parquet.ParquetWriter.

    Attributes
    ----------
    base_path : Path
        The base directory path for output files.
    schema : pa.Schema
        The PyArrow schema for the data.
    shard_size_bytes : int
        Maximum uncompressed size threshold for each file.
    file_prefix : str
        Prefix used for naming files.
    compression : str or None
        The compression codec configuration.
    row_group_size : int or None
        Number of rows per row group.
    shard_index : int
        Current file number (incremented for each new file).
    writer : pq.ParquetWriter or None
        Current active Parquet writer instance.
    current_size : int
        Accumulated uncompressed size in bytes of the current file.
    written_files : list[Path]
        List of absolute paths to all successfully created Parquet files.
    """

    def __init__(
        self,
        base_path: str | Path,
        schema: pa.Schema,
        shard_size_bytes: int = 5_368_709_120,  # 5 GiB
        file_prefix: str = "shard",
        compression: Literal["snappy", "gzip", "brotli", "zstd", "lz4", "none"]
        | None = "zstd",
        row_group_size: int | None = 10_000,
        overwrite: bool = False,
        **kwargs,
    ) -> None:
        self.base_path = Path(base_path)
        self.schema = schema
        self.shard_size_bytes = shard_size_bytes
        self.file_prefix = file_prefix
        self.compression = compression
        self.row_group_size = row_group_size
        self._writer_kwargs = kwargs
        self.shard_index = 0
        self.writer: pq.ParquetWriter | None = None
        self.current_size = 0
        self.written_files: list[Path] = []

        # Handle existing output directory
        if self.base_path.exists():
            if overwrite:
                if self.base_path.is_file():
                    self.base_path.unlink()
                    logger.info(f"Removed existing file: {self.base_path}")
                else:
                    shutil.rmtree(self.base_path)
                    logger.info(f"Removed existing directory: {self.base_path}")
            else:
                raise FileExistsError(
                    f"Output directory '{self.base_path}' already exists."
                )

        self.base_path.mkdir(parents=True, exist_ok=False)

    def __enter__(self) -> "ParquetStreamWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _open_new_shard(self) -> pq.ParquetWriter:
        if self.writer:
            self.writer.close()

        shard_name = f"{self.file_prefix}-{self.shard_index}.parquet"
        shard_path = self.base_path / shard_name

        logger.info(f"Opening file {self.shard_index}: {shard_path}")

        writer_kwargs = {
            "compression": self.compression,
            "use_dictionary": True,
        }
        writer_kwargs.update(self._writer_kwargs)

        self.writer = pq.ParquetWriter(
            str(shard_path),
            self.schema,
            **writer_kwargs,
        )
        self.current_size = 0
        self.written_files.append(shard_path.absolute())
        self.shard_index += 1
        return self.writer

    def write_batch(self, data: dict | pa.Table) -> None:
        """
        Write a batch of data (as a dict of lists or a pyarrow.Table).

        The data is automatically cast to the schema defined at initialization.
        Automatically rolls over to a new file when size limit is exceeded.

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

        if (
            self.writer is None
            or self.current_size + batch_size > self.shard_size_bytes
        ):
            writer = self._open_new_shard()
        else:
            writer = self.writer

        writer.write_table(table, self.row_group_size)
        self.current_size += batch_size

    def close(self) -> None:
        if self.writer:
            self.writer.close()
            self.writer = None
            logger.info("Closed current file writer.")

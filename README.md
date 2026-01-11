# parquet-stream-writer

`parquet-stream-writer` enables streaming data to be written to [Parquet](https://parquet.apache.org/) files with automatic sharding (splitting data across multiple files). When a file reaches a user-defined size limit, the writer automatically creates a new file. This prevents the accumulation of unwieldy, monolithic Parquet files during stream processing.

## Installation

You can install `parquet-stream-writer` from PyPI using `pip` or from conda-forge with [Pixi](https://pixi.sh/).

### Using pip

```sh
pip install parquet-stream-writer
```

### Using pixi

```sh
pixi init my_workspace && cd my_workspace
pixi add parquet-stream-writer
```

## Usage

The library's core class is `ParquetStreamWriter`, which works as a context manager and lets you write data incrementally using its `write_batch` method.

```py
import pyarrow as pa
from parquet_stream_writer import ParquetStreamWriter

# Define your schema
schema = pa.schema(
    [("col_a", pa.int64()), ("col_b", pa.string()), ("col_c", pa.bool_())]
)

# Simulate a data stream
def data_stream():
    for i in range(1_000):
        yield {"col_a": [i, i + 1], "col_b": ["foo", "bar"], "col_c": [True, False]}

# Initialize an instance of `ParquetStreamWriter` and write data to `output_data.parquet`
with ParquetStreamWriter("output_data.parquet", schema, overwrite=True) as writer:
    for batch in data_stream():
        writer.write_batch(batch)
```

### Writing with automatic sharding

By default, `ParquetStreamWriter` writes to a single Parquet file. However, you can enable automatic sharding to split the output into multiple files based on a specified size threshold. To do that, use the `shard_size_bytes` to set the approximate maximum uncompressed size for each file. In this mode, `path` acts as the base directory where shards will be written.

When sharding is enabled, the prefix of the generated files defaults to the name of the output directory. For example, if `path="my_dataset"`, the files will be named `my_dataset-0.parquet`, `my_dataset-1.parquet`, etc. You can override this using the `file_prefix` parameter.

```py
with ParquetStreamWriter(
    "my_dataset",                        # Base directory path
    schema,
    shard_size_bytes=50 * 1024 * 1024,   # Shards will be approx. 50 MiB each
    file_prefix="prefix",                # Custom prefix. Files: prefix-0.parquet, ...
) as writer:
    for batch in data_stream():
        writer.write_batch(batch)
```

### Configuring row group size

The `row_group_size` parameter controls how rows are grouped together within the file. By default, it is set to `None`, which means the group size will be either the total number of rows or 1,048,576, whichever is smaller. Setting a specific value, like 10,000, can make searching and filtering faster because it allows the reader to skip over groups of rows that don't match what you're looking for.

```python
with ParquetStreamWriter(
    "output_data.parquet",
    schema,
    overwrite=True,
    row_group_size=10_000
) as writer:
    for batch in data_stream():
        writer.write_batch(batch)
```

### Passing additional parameters to `ParquetWriter`

`ParquetStreamWriter` uses PyArrow's [`ParquetWriter`](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html#pyarrow.parquet.ParquetWriter) class under the hood. You can further customize the Parquet writing behavior by passing any additional parameters supported by `ParquetWriter` via `**kwargs`.

```py
with ParquetStreamWriter(
    "output_data.parquet",
    schema,
    overwrite=True
    compression="zstd"                  # Use ZSTD for compression
    use_content_defined_chunking=True,  # Write data pages according to content-defined chunk boundaries
) as writer:
    for batch in data_stream():
        writer.write_batch(batch)
```

### Accessing created files

After the writer closes, you can inspect which files it created via the `written_files` attribute.

```py
# The 'writer' object stores a list of the files it created
print("Data was written to the following files:")
for file_path in writer.written_files:
    print(file_path)
```

## `ParquetStreamWriter` API reference

```
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
file_prefix : str or None, default None
    Prefix to use for generated filenames (only used when sharding is
    enabled). If None (default), the value of `path` will be used as the
    prefix and files will be named '{file_prefix}-{index}.parquet'.
row_group_size : int or None, default None
    Number of rows per row group. If None (default), the row group size will
    be either the total number of rows or 1,048,576, whichever is smaller.
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
```

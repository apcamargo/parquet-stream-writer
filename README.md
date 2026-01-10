# parquet-stream-writer

`parquet-stream-writer` provides a memory-efficient way to write streaming data to Parquet. It buffers incoming records and writes them incrementally to disk. When a configurable size threshold is reached, it starts a new Parquet shard, avoiding the need to load the entire dataset into memory. This makes this library suitable for datasets that are too large to fit in the available memory or for continuously generated data.

## Installation

You can install `parquet-stream-writer` from PyPI using `pip` or from conda-forge with [Pixi](https://pixi.sh/).

### Using pip
```bash
pip install parquet-stream-writer
```

### Using pixi
```bash
pixi init my_workspace && cd my_workspace
pixi add parquet-stream-writer
```

## Usage

The core class is `ParquetStreamWriter`. It works as a context manager to ensure files are closed properly.

```python
import pyarrow as pa
from parquet_stream_writer import ParquetStreamWriter

# Define your schema
schema = pa.schema([
    ("timestamp", pa.int64()),
    ("event_type", pa.string()),
    ("value", pa.float64())
])

# Simulate a data stream
def data_stream():
    for i in range(100):
        yield {
            "timestamp": [i],
            "event_type": ["reading"],
            "value": [float(i)]
        }

# Initialize the writer
# This will write to ./output_data/shard-0.parquet, ./output_data/shard-1.parquet, etc.
with ParquetStreamWriter("output_data", schema, overwrite=True) as writer:
    for batch in data_stream():
        writer.write_batch(batch)
```

### Configuring file size and naming

You can configure when new files are created and how they are named. Additional [PyArrow](https://arrow.apache.org/docs/python/index.html) parameters can be passed through via `**kwargs`.

```python
with ParquetStreamWriter(
    "data_stream",
    schema,
    shard_size_bytes=50 * 1024 * 1024,   # Shards will be approx. 50 MiB each
    file_prefix="events",                # Output name: events-0.parquet
    compression="snappy"
) as writer:
    for batch in stream:
        writer.write_batch(batch)
```

### Accessing created files

After the writer closes, you can inspect which files were actually created. This is useful for logging or triggering downstream processes.

```python
with ParquetStreamWriter("output", schema) as writer:
    writer.write_batch(batch_1)
    writer.write_batch(batch_2)

# The 'writer' object stores a list of the files it created
print("Data was written to the following files:")
for file_path in writer.written_files:
    print(file_path)
```

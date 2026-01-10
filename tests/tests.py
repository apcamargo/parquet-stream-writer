import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_stream_writer import ParquetStreamWriter


@pytest.fixture
def output_dir(tmp_path):
    """Fixture for temporary output directory."""
    path = tmp_path / "parquet_output"
    yield path
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def schema():
    """Fixture for a simple PyArrow schema."""
    return pa.schema([("id", pa.int64()), ("value", pa.string())])


def test_basic_write(output_dir, schema):
    """Test writing a single batch of data."""
    with ParquetStreamWriter(output_dir, schema) as writer:
        data = {"id": [1, 2, 3], "value": ["a", "b", "c"]}
        writer.write_batch(data)

    # Verify file creation
    assert output_dir.exists()
    files = list(output_dir.glob("*.parquet"))
    assert len(files) == 1
    assert files[0].name == "shard-0.parquet"

    # Verify content
    table = pq.read_table(files[0])
    assert table.column("id").to_pylist() == [1, 2, 3]
    assert table.column("value").to_pylist() == ["a", "b", "c"]


def test_sharding(output_dir, schema):
    """Test that multiple files are created when size limit is exceeded."""
    # Create a small batch to force rollover
    data = {"id": [1], "value": ["x" * 100]}
    batch_table = pa.Table.from_pydict(data, schema=schema)
    batch_size = batch_table.nbytes

    # Set limit smaller than 2 batches so every batch creates a new file (after the first)
    # Actually, the logic is: if current + new > limit, open new.
    # So if limit < batch_size, it will open new for every batch.

    with ParquetStreamWriter(
        output_dir, schema, shard_size_bytes=batch_size - 1, file_prefix="test-data"
    ) as writer:
        for _ in range(3):
            writer.write_batch(batch_table)

    files = sorted(list(output_dir.glob("*.parquet")))
    assert len(files) == 3
    assert files[0].name == "test-data-0.parquet"
    assert files[1].name == "test-data-1.parquet"
    assert files[2].name == "test-data-2.parquet"

    # Check total rows
    full_table = pq.read_table(output_dir)
    assert len(full_table) == 3


def test_written_files_tracking(output_dir, schema):
    """Test that the written_files attribute correctly tracks created files."""
    with ParquetStreamWriter(output_dir, schema, file_prefix="track") as writer:
        writer.write_batch({"id": [1], "value": ["a"]})
        # Force new file
        writer._open_new_shard()
        writer.write_batch({"id": [2], "value": ["b"]})

    assert len(writer.written_files) == 2
    assert writer.written_files[0].name == "track-0.parquet"
    assert writer.written_files[1].name == "track-1.parquet"
    assert all(f.is_absolute() for f in writer.written_files)


def test_schema_casting(output_dir, schema):
    """Test that data is cast to the schema."""
    with ParquetStreamWriter(output_dir, schema) as writer:
        # Input has int32, schema has int64. Should cast fine.
        small_int_data = pa.Table.from_pydict(
            {"id": [1, 2], "value": ["a", "b"]},
            schema=pa.schema([("id", pa.int32()), ("value", pa.string())]),
        )
        writer.write_batch(small_int_data)

    table = pq.read_table(output_dir)
    assert table.schema.field("id").type == pa.int64()

    # Also test dict input with casting
    with ParquetStreamWriter(
        output_dir / "cast_test", schema, overwrite=True
    ) as writer:
        # Input is int32, should be cast to int64
        dict_data = {"id": [3, 4], "value": ["c", "d"]}
        writer.write_batch(dict_data)

    table = pq.read_table(output_dir / "cast_test")
    assert table.schema.field("id").type == pa.int64()


def test_schema_validation_failure(output_dir, schema):
    """Test that incompatible data raises an error."""
    with ParquetStreamWriter(output_dir, schema) as writer:
        # Try to write string to int column
        bad_data = {"id": ["not-an-int"], "value": ["a"]}

        with pytest.raises(pa.ArrowInvalid):
            writer.write_batch(bad_data)


def test_overwrite_behavior(output_dir, schema):
    """Test overwrite parameter."""
    output_dir.mkdir()
    (output_dir / "old.txt").touch()

    # Fail if exists and overwrite=False
    with pytest.raises(FileExistsError):
        ParquetStreamWriter(output_dir, schema, overwrite=False)

    # Succeed if overwrite=True
    with ParquetStreamWriter(output_dir, schema, overwrite=True) as writer:
        writer.write_batch({"id": [1], "value": ["a"]})

    assert not (output_dir / "old.txt").exists()
    assert (output_dir / "shard-0.parquet").exists()


def test_overwrite_file(output_dir, schema):
    """Test overwrite=True removes an existing file at the path."""
    output_dir.touch()  # Create a file, not a directory
    with ParquetStreamWriter(output_dir, schema, overwrite=True) as writer:
        writer.write_batch({"id": [1], "value": ["a"]})
    assert output_dir.is_dir()
    assert (output_dir / "shard-0.parquet").exists()


def test_empty_batch(output_dir, schema):
    """Test that empty batches don't create files unnecessarily."""
    with ParquetStreamWriter(output_dir, schema) as writer:
        writer.write_batch({"id": [], "value": []})
        writer.write_batch({"id": [1], "value": ["a"]})
    files = list(output_dir.glob("*.parquet"))
    assert len(files) == 1


def test_large_single_batch(output_dir, schema):
    """Test that a single batch exceeding shard size writes correctly."""
    large_data = {"id": list(range(10000)), "value": ["x"] * 10000}
    with ParquetStreamWriter(output_dir, schema, shard_size_bytes=1) as writer:
        writer.write_batch(large_data)
    files = list(output_dir.glob("*.parquet"))
    assert len(files) == 1
    table = pq.read_table(files[0])
    assert len(table) == 10000


def test_no_writes_no_files(output_dir, schema):
    """Test that using context manager without writes creates no files."""
    with ParquetStreamWriter(output_dir, schema) as writer:
        pass
    files = list(output_dir.glob("*.parquet"))
    assert len(files) == 0


def test_writer_kwargs(output_dir, schema):
    """Test that kwargs are passed to ParquetWriter."""
    with ParquetStreamWriter(output_dir, schema, version="2.6") as writer:
        writer.write_batch({"id": [1], "value": ["a"]})
    files = list(output_dir.glob("*.parquet"))
    assert len(files) == 1

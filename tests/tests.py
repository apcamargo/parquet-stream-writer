import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_stream_writer import ParquetStreamWriter


@pytest.fixture
def output_path(tmp_path):
    """Fixture for temporary output path (file or dir)."""
    path = tmp_path / "parquet_output"
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    yield path
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


@pytest.fixture
def schema():
    """Fixture for a simple PyArrow schema."""
    return pa.schema([("id", pa.int64()), ("value", pa.string())])


def test_basic_write(output_path, schema):
    """Test writing a single batch of data (Default: Single File)."""
    output_file = output_path.with_suffix(".parquet")

    with ParquetStreamWriter(output_file, schema) as writer:
        data = {"id": [1, 2, 3], "value": ["a", "b", "c"]}
        writer.write_batch(data)

    assert output_file.exists()
    assert output_file.is_file()

    table = pq.read_table(output_file)
    assert table.column("id").to_pylist() == [1, 2, 3]
    assert table.column("value").to_pylist() == ["a", "b", "c"]


def test_sharding(output_path, schema):
    """Test that multiple files are created when size limit is exceeded."""
    data = {"id": [1], "value": ["x" * 100]}
    batch_table = pa.Table.from_pydict(data, schema=schema)
    batch_size = batch_table.nbytes

    with ParquetStreamWriter(
        output_path, schema, shard_size_bytes=batch_size - 1, file_prefix="test-data"
    ) as writer:
        for _ in range(3):
            writer.write_batch(batch_table)

    assert output_path.is_dir()
    files = sorted(list(output_path.glob("*.parquet")))
    assert len(files) == 3
    assert files[0].name == "test-data-0.parquet"
    assert files[1].name == "test-data-1.parquet"
    assert files[2].name == "test-data-2.parquet"

    full_table = pq.read_table(output_path)
    assert len(full_table) == 3


def test_written_files_tracking(output_path, schema):
    """Test that the written_files attribute correctly tracks created files."""
    with ParquetStreamWriter(
        output_path, schema, shard_size_bytes=1000, file_prefix="track"
    ) as writer:
        writer.write_batch({"id": [1], "value": ["a"]})
        writer._open_new_shard()
        writer.write_batch({"id": [2], "value": ["b"]})

    assert len(writer.written_files) == 2
    assert writer.written_files[0].name == "track-0.parquet"
    assert writer.written_files[1].name == "track-1.parquet"
    assert all(f.is_absolute() for f in writer.written_files)


def test_schema_casting(output_path, schema):
    """Test that data is cast to the schema."""
    output_file = output_path.with_suffix(".parquet")
    with ParquetStreamWriter(output_file, schema) as writer:
        small_int_data = pa.Table.from_pydict(
            {"id": [1, 2], "value": ["a", "b"]},
            schema=pa.schema([("id", pa.int32()), ("value", pa.string())]),
        )
        writer.write_batch(small_int_data)

    table = pq.read_table(output_file)
    assert table.schema.field("id").type == pa.int64()


def test_schema_validation_failure(output_path, schema):
    """Test that incompatible data raises an error."""
    with ParquetStreamWriter(output_path, schema) as writer:
        bad_data = {"id": ["not-an-int"], "value": ["a"]}
        with pytest.raises(pa.ArrowInvalid):
            writer.write_batch(bad_data)


def test_overwrite_behavior(output_path, schema):
    """Test overwrite parameter."""
    output_path.mkdir()
    (output_path / "old.txt").touch()

    with pytest.raises(FileExistsError):
        ParquetStreamWriter(output_path, schema, overwrite=False)

    with ParquetStreamWriter(output_path, schema, overwrite=True) as writer:
        writer.write_batch({"id": [1], "value": ["a"]})

    assert output_path.is_file()
    assert not (output_path / "old.txt").exists()


def test_empty_batch(output_path, schema):
    """Test that empty batches don't create files unnecessarily."""
    with ParquetStreamWriter(output_path, schema) as writer:
        writer.write_batch({"id": [], "value": []})
        writer.write_batch({"id": [1], "value": ["a"]})

    assert output_path.is_file()


def test_large_single_batch(output_path, schema):
    """Test that a single batch exceeding shard size writes correctly."""
    large_data = {"id": list(range(10000)), "value": ["x"] * 10000}
    with ParquetStreamWriter(output_path, schema, shard_size_bytes=1) as writer:
        writer.write_batch(large_data)

    assert output_path.is_dir()
    files = list(output_path.glob("*.parquet"))
    assert len(files) == 1
    assert files[0].name == f"{output_path.name}-0.parquet"


def test_no_writes_no_files(output_path, schema):
    """Test that using context manager without writes creates no files."""
    with ParquetStreamWriter(output_path, schema) as writer:
        pass
    assert not output_path.exists()


def test_writer_kwargs_passed(output_path, schema):
    """Test that kwargs are accepted by ParquetStreamWriter."""
    with ParquetStreamWriter(output_path, schema, version="2.6") as writer:
        writer.write_batch({"id": [1], "value": ["a"]})
    assert output_path.exists()


def test_kwargs_verification(output_path, schema):
    """Verify that kwargs actually affect the underlying writer."""
    # Using write_statistics=False as a verifiable kwarg
    output_file = output_path.with_suffix(".parquet")
    with ParquetStreamWriter(output_file, schema, write_statistics=False) as writer:
        writer.write_batch({"id": [1, 2, 3], "value": ["a", "b", "c"]})

    metadata = pq.read_metadata(output_file)
    assert metadata.row_group(0).column(0).is_stats_set is False


def test_directory_constraints_single_file(tmp_path, schema):
    """Test that FileNotFoundError is raised when parent dir of single file doesn't exist."""
    non_existent_path = tmp_path / "missing_parent" / "output.parquet"
    with pytest.raises(FileNotFoundError):
        # The library should not create parent directories
        with ParquetStreamWriter(non_existent_path, schema) as writer:
            writer.write_batch({"id": [1], "value": ["a"]})


def test_directory_constraints_sharding(tmp_path, schema):
    """Test that FileNotFoundError is raised when parent dir of shard dir doesn't exist."""
    non_existent_shard_dir = tmp_path / "missing_parent" / "shards"
    with pytest.raises(FileNotFoundError):
        # The library should not create parent directories, even for sharding
        ParquetStreamWriter(non_existent_shard_dir, schema, shard_size_bytes=1024)


def test_sharding_directory_creation(tmp_path, schema):
    """Test that the library DOES create the immediate shard directory if parent exists."""
    shard_dir = tmp_path / "new_shard_dir"
    assert not shard_dir.exists()

    with ParquetStreamWriter(shard_dir, schema, shard_size_bytes=1024) as writer:
        writer.write_batch({"id": [1], "value": ["a"]})

    assert shard_dir.exists()
    assert shard_dir.is_dir()

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_ROWS = 250_000


class ChunkWriter:
    def __init__(self, path: Path, chunk: int = CHUNK_ROWS):
        self.path = path
        self.chunk = chunk
        self.buffer: list[dict] = []
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.rows = 0

    def add(self, records: list[dict]) -> None:
        if not records:
            return
        self.buffer.extend(records)
        if len(self.buffer) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        frame = pd.DataFrame(self.buffer)
        self.buffer = []
        if self.writer is None:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="snappy")
        else:
            table = pa.Table.from_arrays(
                [self._column(frame, field) for field in self.schema], schema=self.schema
            )
        self.writer.write_table(table)
        self.rows += table.num_rows

    def _column(self, frame: pd.DataFrame, field: pa.Field) -> pa.Array:
        if field.name not in frame.columns:
            return pa.nulls(len(frame), field.type)
        try:
            return pa.array(frame[field.name], type=field.type, from_pandas=True)
        except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            raise ValueError(
                f"{self.path.name}: column {field.name} changed type mid-file, "
                f"expected {field.type}"
            ) from exc

    def close(self) -> int:
        self.flush()
        if self.writer is not None:
            self.writer.close()
        else:
            pd.DataFrame().to_parquet(self.path, index=False)
        return self.rows


def conform(table: pa.Table, schema: pa.Schema) -> pa.Table:
    columns = []
    for field in schema:
        if field.name in table.schema.names:
            column = table.column(field.name)
            columns.append(column if column.type == field.type else column.cast(field.type))
        else:
            columns.append(pa.nulls(table.num_rows, field.type))
    return pa.Table.from_arrays([c.combine_chunks() for c in columns], schema=schema)


def merge(parts: list[Path], target: Path) -> int:
    parts = [part for part in parts if part.exists()]
    schemas = [pq.read_schema(part) for part in parts]
    if not schemas:
        pd.DataFrame().to_parquet(target, index=False)
        return 0
    schema = pa.unify_schemas(schemas, promote_options="permissive")
    writer, rows = pq.ParquetWriter(target, schema, compression="snappy"), 0
    try:
        for part in parts:
            table = pq.read_table(part)
            if table.num_rows:
                writer.write_table(conform(table, schema))
                rows += table.num_rows
    finally:
        writer.close()
    return rows

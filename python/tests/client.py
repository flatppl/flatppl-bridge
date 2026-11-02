"""A minimal protocol client for the tests: Arrow parts out, NumPy values back."""

from __future__ import annotations

import email.parser
import email.policy
import json

import numpy as np
import pyarrow as pa
import requests


def part(value) -> bytes:
    """Encode a NumPy value in the part layout of docs/protocol.md."""
    array = np.asarray(value)
    if array.ndim == 0:
        column = pa.array([array.item()])
    else:
        column = pa.array(array.reshape(-1))
        for size in reversed(array.shape[1:]):
            column = pa.FixedSizeListArray.from_arrays(column, size)
    table = pa.table({"value": column})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def decode(data: bytes):
    """Decode a reply part: a NumPy array, or a dict for a record."""
    table = pa.ipc.open_stream(data).read_all()
    valueset = table.schema.metadata[b"flatppl.valueset"].decode()
    if table.num_columns == 1 and table.column_names == ["value"]:
        column = table.column(0).combine_chunks()
        values = column
        dims = []
        while pa.types.is_fixed_size_list(values.type):
            dims.append(values.type.list_size)
            values = values.flatten()
        array = values.to_numpy(zero_copy_only=False)
        if valueset.startswith("cartpow"):
            return array.reshape([len(column)] + dims)
        return array.reshape(dims) if dims else array[0]
    return {name: table.column(name).to_numpy() for name in table.column_names}


class Reply:
    def __init__(self, response: requests.Response) -> None:
        self.status = response.status_code
        self.content_type = response.headers.get("content-type", "")
        self.body = response.content

    def json(self):
        return json.loads(self.body)

    def parts(self) -> dict:
        assert self.status == 200, self.body
        message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            b"Content-Type: " + self.content_type.encode() + b"\r\n\r\n" + self.body
        )
        return {
            p.get_param("name", header="content-disposition"): decode(p.get_payload(decode=True))
            for p in message.iter_parts()
        }


def query(url: str, text: str, parts: dict | None = None, options: dict | None = None, token: str | None = None) -> Reply:
    files = [("query", ("query", text.encode(), "text/x-flatppl; charset=utf-8"))]
    if options is not None:
        files.append(("options", ("options", json.dumps(options).encode(), "application/json")))
    for name, value in (parts or {}).items():
        data = value if isinstance(value, bytes) else part(value)
        files.append((name, (name, data, "application/vnd.apache.arrow.stream")))
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return Reply(requests.post(f"{url}/v1/query", files=files, headers=headers, timeout=60))

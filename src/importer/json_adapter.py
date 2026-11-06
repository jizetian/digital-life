"""Stream a Moect JSON export one message at a time.

The real export is one top-level object: a small ``metadata`` object, then a
``messages`` array that can be hundreds of megabytes. This reader never calls
``json.load`` on the file. It fills a bounded buffer and ``JSONDecoder.raw_decode``s
one value at a time, so memory stays near the chunk size plus the current message.
"""

from __future__ import annotations

import codecs
import json
from collections.abc import Iterator
from pathlib import Path

from importer.schema_detector import SchemaError

DEFAULT_CHUNK_SIZE = 1024 * 1024
# One message object, not the whole export. Merged forwards can be large;
# a single record above this is rejected instead of buffering the file.
MAX_VALUE_CHARS = 32 * 1024 * 1024


class TextChunkReader:
    def __init__(self, path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self._fh = path.open("rb")
        self._decoder = codecs.getincrementaldecoder("utf-8-sig")()
        self._buf = ""
        self._eof = False
        self._json = json.JSONDecoder()

    def close(self) -> None:
        self._fh.close()

    def _pull(self) -> None:
        if self._eof:
            return
        data = self._fh.read(self.chunk_size)
        if data == b"":
            self._buf += self._decoder.decode(b"", final=True)
            self._eof = True
            return
        self._buf += self._decoder.decode(data)

    def skip_ws(self) -> None:
        while True:
            self._buf = self._buf.lstrip()
            if self._buf or self._eof:
                return
            self._pull()

    def exhausted(self) -> bool:
        self.skip_ws()
        return self._eof and not self._buf

    def startswith(self, prefix: str) -> bool:
        self.skip_ws()
        while len(self._buf) < len(prefix) and not self._eof:
            self._pull()
        return self._buf.startswith(prefix)

    def discard(self, count: int) -> None:
        self._buf = self._buf[count:]

    def decode_value(self):
        while True:
            self.skip_ws()
            if not self._buf:
                raise SchemaError("unexpected end of JSON")
            try:
                value, index = self._json.raw_decode(self._buf)
            except json.JSONDecodeError as exc:
                if self._eof or len(self._buf) >= MAX_VALUE_CHARS:
                    raise SchemaError(f"invalid JSON: {exc.msg}") from exc
                self._pull()
                continue
            # A number or literal that ends exactly on a chunk boundary may still
            # continue in the next chunk. Objects and strings are complete only
            # once raw_decode succeeds, but rereading is cheap and always safe.
            if index == len(self._buf) and not self._eof:
                previous = len(self._buf)
                self._pull()
                if len(self._buf) != previous or not self._eof:
                    continue
            self._buf = self._buf[index:]
            return value


def iter_moect_events(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[tuple]:
    """Yield ``("metadata", dict)`` then ``("message", index, object)`` events.

    ``index`` is the zero-based position in the ``messages`` array.
    """
    reader = TextChunkReader(Path(path), chunk_size=chunk_size)
    try:
        if not reader.startswith("{"):
            raise SchemaError("top-level JSON value must be an object")
        reader.discard(1)
        metadata_seen = False
        messages_seen = False
        while True:
            if reader.exhausted():
                raise SchemaError("unterminated top-level object")
            if reader.startswith("}"):
                reader.discard(1)
                break
            if reader.startswith(","):
                reader.discard(1)
                continue
            key = reader.decode_value()
            if not isinstance(key, str):
                raise SchemaError("object key must be a string")
            if not reader.startswith(":"):
                raise SchemaError(f"expected ':' after key {key!r}")
            reader.discard(1)
            if key == "metadata":
                metadata = reader.decode_value()
                if not isinstance(metadata, dict):
                    raise SchemaError("metadata must be an object")
                metadata_seen = True
                yield ("metadata", metadata)
            elif key == "messages":
                if not metadata_seen:
                    raise SchemaError("metadata must appear before messages")
                if not reader.startswith("["):
                    raise SchemaError("messages must be an array")
                reader.discard(1)
                messages_seen = True
                index = 0
                while True:
                    if reader.exhausted():
                        raise SchemaError("unterminated messages array")
                    if reader.startswith("]"):
                        reader.discard(1)
                        break
                    if reader.startswith(","):
                        reader.discard(1)
                        continue
                    item = reader.decode_value()
                    yield ("message", index, item)
                    index += 1
            else:
                reader.decode_value()
        if not metadata_seen:
            raise SchemaError("missing metadata")
        if not messages_seen:
            raise SchemaError("missing messages")
        if not reader.exhausted():
            raise SchemaError("trailing data after top-level object")
    finally:
        reader.close()

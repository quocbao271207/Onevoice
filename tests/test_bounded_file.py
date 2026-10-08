import pytest

from src.utils import bounded_file
from src.utils.bounded_file import read_stable_regular_file


def test_bounded_file_reads_the_same_regular_file_twice(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b'{"safe": true}')

    assert read_stable_regular_file(
        path,
        maximum_bytes=100,
        label="Test config",
    ) == b'{"safe": true}'


def test_bounded_file_rejects_changed_bytes_with_stable_identity(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "config.json"
    path.write_bytes(b'{"safe": true}')
    real_fdopen = bounded_file.os.fdopen

    class ChangedSecondRead:
        def __init__(self, handle):
            self._handle = handle
            self._reads = 0

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):
            return self._handle.__exit__(*args)

        def fileno(self):
            return self._handle.fileno()

        def read(self, size=-1):
            payload = self._handle.read(size)
            self._reads += 1
            if self._reads == 2:
                return b"X" + payload[1:]
            return payload

        def seek(self, offset, whence=0):
            return self._handle.seek(offset, whence)

    monkeypatch.setattr(
        bounded_file.os,
        "fdopen",
        lambda descriptor, mode: ChangedSecondRead(real_fdopen(descriptor, mode)),
    )

    with pytest.raises(RuntimeError, match="changed while reading"):
        read_stable_regular_file(
            path,
            maximum_bytes=100,
            label="Test config",
        )

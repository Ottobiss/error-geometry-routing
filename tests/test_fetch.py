import io
import sys
import types
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_cityscapes import HttpRangeFile  # noqa: E402


class FakeSession:
    """Отдаёт байты по Range; каждый третий запрос «обрывается»."""

    def __init__(self, blob):
        self.blob, self.calls = blob, 0

    def get(self, url, headers, **kw):
        self.calls += 1
        if self.calls % 3 == 0:
            raise ConnectionError("reset")
        start, end = map(int, headers["Range"][6:].split("-"))
        r = types.SimpleNamespace(status_code=206, content=self.blob[start:end + 1],
                                  headers={"Content-Range": f"bytes {start}-{end}/{len(self.blob)}"})
        r.raise_for_status = lambda: None
        return r


def test_remote_zip_extracts_only_prefix(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    buf = io.BytesIO()
    files = {f"leftImg8bit/{split}/a/{i}.png": bytes([i]) * (1000 + i)
             for split in ("train", "val") for i in range(5)}
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    raw = HttpRangeFile(FakeSession(buf.getvalue()), "url")
    zf = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=4096))
    val = [i for i in zf.infolist() if i.filename.startswith("leftImg8bit/val/")]
    assert len(val) == 5
    for info in val:
        assert zf.read(info) == files[info.filename]

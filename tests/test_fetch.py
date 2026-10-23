import io
import os
import re
import types
import zipfile

from download import acdc
from download.cityscapes import HttpRangeFile


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


class FakeStreamResponse:
    """Ответ сервера ACDC: архив целиком, кусками, без Range."""

    def __init__(self, blob, chunk=777):
        self.blob, self.chunk, self.read_bytes = blob, chunk, 0

    def iter_content(self, _):
        for i in range(0, len(self.blob), self.chunk):
            self.read_bytes += min(self.chunk, len(self.blob) - i)
            yield self.blob[i:i + self.chunk]

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def test_acdc_stream_extracts_val_and_stops_early(monkeypatch, tmp_path):
    buf = io.BytesIO()
    files = {}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", b"readme")
        for cond in ("fog", "night", "rain", "snow"):
            for split in ("test", "train", "val", "val_ref"):
                z.writestr(f"rgb_anon/{cond}/{split}/", b"")
                for i in range(3):
                    name = f"rgb_anon/{cond}/{split}/GP01/{i}_rgb_anon.png"
                    files[name] = bytes([i]) * (2000 + i)
                    z.writestr(name, files[name])
        z.writestr("rgb_anon/zzz_tail.bin", os.urandom(50000))   # несжимаемый хвост
    resp = FakeStreamResponse(buf.getvalue())
    monkeypatch.setattr(acdc.requests, "get", lambda *a, **k: resp)
    pattern = re.compile(r"^rgb_anon/(fog|night|rain|snow)/val/.+\.png$")

    saved, matched = acdc.run_once("url", pattern, str(tmp_path), stop_after=4)
    assert (saved, matched) == (12, 12)
    for name, data in files.items():
        out = tmp_path / name
        assert out.exists() == bool(pattern.search(name))
        if out.exists():
            assert out.read_bytes() == data
    assert resp.read_bytes < len(buf.getvalue()) - 40000     # хвост архива не читался

    saved, matched = acdc.run_once("url", pattern, str(tmp_path), stop_after=4)
    assert (saved, matched) == (0, 12)                        # повторный запуск ничего не пишет

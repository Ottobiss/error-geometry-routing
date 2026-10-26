"""Скачивает из архива Cityscapes на сервере только нужные файлы, не качая архив целиком.

ZIP читается удалённо через HTTP Range: сначала оглавление в конце архива, затем
нужные записи по одной. Уже скачанные файлы пропускаются, так что после обрыва
скрипт можно просто перезапустить.

Логин — через cityscapesscripts (спросит логин/пароль в терминале).

    python -m download.cityscapes --check
    python -m download.cityscapes
"""
import argparse
import io
import os
import time
import zipfile

from cityscapesscripts.download.downloader import get_available_packages, login

FILE_URL = "https://www.cityscapes-dataset.com/file-handling/?packageID={}"


class HttpRangeFile(io.RawIOBase):
    """Файл только для чтения поверх HTTP Range-запросов, с повторами при обрывах."""

    def __init__(self, session, url, relogin=None, retries=8):
        self.session, self.url, self.relogin, self.retries = session, url, relogin, retries
        self.pos = 0
        r = self._get(0, 0)
        content_range = r.headers.get("Content-Range", "")
        if r.status_code != 206 or "/" not in content_range:
            raise RuntimeError(f"Сервер не поддерживает Range (status {r.status_code}).")
        self.size = int(content_range.rsplit("/", 1)[1])

    def _get(self, start, end):
        for attempt in range(self.retries):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                     allow_redirects=False, timeout=60)
                if r.status_code == 302 and self.relogin:      # сессия истекла
                    self.session = self.relogin()
                    continue
                r.raise_for_status()
                if r.status_code == 206 and len(r.content) != end - start + 1:
                    raise IOError("неполный ответ")
                return r
            except Exception as e:
                wait = min(2 ** attempt, 60)
                print(f"  обрыв ({e}); повтор через {wait} с")
                time.sleep(wait)
        raise IOError(f"Не удалось получить байты {start}-{end}")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self._get(self.pos, self.pos + n - 1).content
        b[:n] = data
        self.pos += n
        return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="leftImg8bit_trainvaltest.zip")
    ap.add_argument("--prefix", default="leftImg8bit/val/",
                    help="какие файлы брать из архива (начало пути внутри zip)")
    ap.add_argument("--dest", default="data/cityscapes")
    ap.add_argument("--check", action="store_true",
                    help="только проверить доступ и показать, что будет скачано")
    args = ap.parse_args()

    session = login()
    packages = {p["name"]: p["packageID"] for p in get_available_packages(session=session)}
    url = FILE_URL.format(packages[args.package])

    raw = HttpRangeFile(session, url, relogin=login)
    zf = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=4 << 20))
    entries = [i for i in zf.infolist() if i.filename.startswith(args.prefix) and not i.is_dir()]
    total = sum(i.compress_size for i in entries)
    print(f"Архив {raw.size / 2**30:.2f} ГБ; по префиксу '{args.prefix}': "
          f"{len(entries)} файлов, {total / 2**30:.2f} ГБ")
    if args.check:
        return

    for k, info in enumerate(entries, 1):
        out = os.path.join(args.dest, info.filename)
        if os.path.exists(out) and os.path.getsize(out) == info.file_size:
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".part"
        with zf.open(info) as src, open(tmp, "wb") as dst:   # zipfile проверяет CRC
            dst.write(src.read())
        os.replace(tmp, out)
        if k % 25 == 0 or k == len(entries):
            print(f"  {k}/{len(entries)}")
    print(f"Готово: {len(entries)} файлов в {args.dest}")


if __name__ == "__main__":
    main()

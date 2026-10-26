"""Скачивает из архива ACDC только нужные кадры (по умолчанию val), не сохраняя архив на диск.

Сервер ACDC отдаёт архив без входа, но не поддерживает HTTP Range, а ссылка одноразовая.
Поэтому архив читается потоком от начала: перед каждым файлом в ZIP стоит локальный
заголовок с размером, нужные файлы распаковываются и пишутся, остальные пропускаются.
Трафик — весь архив (~15.6 ГБ) или его часть до последнего нужного файла; на диск — только
нужные файлы. После обрыва скрипт начинает поток заново, уже скачанные файлы не переписывает.

API сайта (из его фронтенда): GET /api/packages — список пакетов, GET /api/getPackageUri/<id>
— токен ссылки, файл — /api/downloadPackage/<токен>/<имя>.

    python -m download.acdc
    python -m download.acdc --pattern "^rgb_anon/(fog|night|rain|snow)/(val|train)/"
"""
import argparse
import os
import re
import struct
import time
import zlib

import requests

API = "https://acdc.vision.ee.ethz.ch/api"
LOCAL = b"PK\x03\x04"
VAL_PATTERN = r"^rgb_anon/(fog|night|rain|snow)/val/.+\.png$"


def package_url(name):
    packages = requests.get(f"{API}/packages", timeout=60).json()["packages"]
    ids = {p["name"]: p["packageId"] for p in packages}
    if name not in ids:
        raise SystemExit(f"Нет пакета {name}; есть: {sorted(ids)}")
    token = requests.get(f"{API}/getPackageUri/{ids[name]}", timeout=60).json()["token"]
    return f"{API}/downloadPackage/{token}/{name}"


class Stream:
    """Последовательное чтение тела HTTP-ответа точными порциями."""

    def __init__(self, response):
        self.it = response.iter_content(1 << 20)
        self.buf = b""
        self.done = 0

    def read(self, n):
        while len(self.buf) < n:
            chunk = next(self.it, None)
            if chunk is None:
                raise EOFError("поток закончился раньше времени")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        self.done += n
        return out

    def skip(self, n):
        while n > 0:
            n -= len(self.read(min(n, 1 << 22)))


def run_once(url, pattern, dest, stop_after):
    """Один проход по потоку. Возвращает (сохранено, всего подходящих)."""
    saved = matched = 0
    left_groups, current = set(), None   # группы «условие/сплит» с нужными файлами, уже пройденные
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        s = Stream(r)
        t0 = time.time()
        while s.read(4) == LOCAL:
            _, flag, method, _, _, crc, csize, usize, nlen, xlen = struct.unpack(
                "<HHHHHIIIHH", s.read(26))
            name = s.read(nlen).decode("utf-8")
            s.read(xlen)
            if flag & 8 or csize == 0xFFFFFFFF:
                raise RuntimeError(f"{name}: размер не в заголовке (data descriptor / zip64)")
            m = pattern.search(name) if not name.endswith("/") else None
            # Группа «условие/сплит»: когда все нужные группы пройдены, дальше читать незачем.
            group = "/".join(name.split("/")[1:3])
            if current is not None and group != current:
                left_groups.add(current)
                current = None
            if not m:
                s.skip(csize)
                if stop_after and len(left_groups) >= stop_after:
                    break
                continue
            current = group
            matched += 1
            out = os.path.join(dest, name)
            if os.path.exists(out) and os.path.getsize(out) == usize:
                s.skip(csize)
                continue
            data = s.read(csize)
            data = zlib.decompress(data, -15) if method == 8 else data
            if zlib.crc32(data) != crc or len(data) != usize:
                raise IOError(f"{name}: не сошлась CRC или размер")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out + ".part", "wb") as f:
                f.write(data)
            os.replace(out + ".part", out)
            saved += 1
            if matched % 25 == 0:
                mb = s.done / 2**20
                print(f"  {matched} файлов, прочитано {mb / 1024:.1f} ГБ, "
                      f"{mb / (time.time() - t0):.1f} МБ/с")
    return saved, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="rgb_anon_trainvaltest.zip")
    ap.add_argument("--pattern", default=VAL_PATTERN, help="регулярное выражение для путей внутри zip")
    ap.add_argument("--dest", default="data/acdc")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="остановить поток, когда пройдено столько групп «условие/сплит» с нужными "
                         "файлами; по умолчанию 4 для шаблона val (fog/night/rain/snow), иначе 0 — "
                         "читать до конца")
    ap.add_argument("--retries", type=int, default=5)
    args = ap.parse_args()

    pattern = re.compile(args.pattern)
    if args.stop_after is None:
        args.stop_after = 4 if args.pattern == VAL_PATTERN else 0
    for attempt in range(1, args.retries + 1):
        try:
            saved, matched = run_once(package_url(args.package), pattern, args.dest, args.stop_after)
            print(f"Готово: подходящих файлов {matched}, новых сохранено {saved}, папка {args.dest}")
            return
        except Exception as e:
            print(f"Обрыв ({e}); попытка {attempt}/{args.retries}, начинаю поток заново")
            time.sleep(min(2 ** attempt, 60))
    raise SystemExit("Не удалось скачать: исчерпаны попытки")


if __name__ == "__main__":
    main()

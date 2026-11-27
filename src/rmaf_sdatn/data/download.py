import hashlib
import re
import shutil
import zlib
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree
from zipfile import ZipFile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..locking import file_lock
from ..serialization import atomic_destination

STATE_COMMIT = "efe508dfe76b33fe437e2e6d3274534b50cc2f37"
STATE_REPO = "https://raw.githubusercontent.com/Bob05757/Renewable-energy-generation-input-feature-variables-analysis"
STATE_FILES = {"G5": "Solar station site 5 (Nominal capacity-110MW).xlsx",
               "G7": "Solar station site 7 (Nominal capacity-30MW).xlsx"}
STATE_BLOBS = {"G5": ("fb4928f65bae447bda6b17e8bb9f5dd46c6e448d", 3465493),
               "G7": ("155868860aeadbad846aba44a8ff02eb5ed6d960", 4302012)}
PVOD_LANDING = "https://www.scidb.cn/en/detail?dataSetId=f8f3d7af144f441795c5781497e56b62"
PVOD_V4 = ("https://download.scidb.cn/download?fileId=6197068d00eb5848da3afc08"
           "&path=%2FV4%2FPVODdatasets_v1.0.zip&fileName=PVODdatasets_v1.0.zip")
PVOD_SIZE = 7983824
PVOD_SHA256 = "0f726b42235bf99a2c118a3cb7544722150489b01af3a2853a7c9778b538883e"
OEDI = "https://oedi-data-lake.s3.amazonaws.com"
PRIZE = "pvdaq/2023-solar-data-prize/"


def session():
    client = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    client.mount("https://", HTTPAdapter(max_retries=retries))
    return client


def fetch(client, url, destination, expected_size=None, blob=None, headers=None, digest=None, resume=False):
    destination = Path(destination)
    with file_lock(destination.with_suffix(destination.suffix+".lock"), timeout=3600):
        return _fetch(client, url, destination, expected_size, blob, headers, digest, resume)


def _fetch(client, url, destination, expected_size, blob, headers, digest, resume):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def verify(path):
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        if blob:
            content = path.read_bytes()
            prefix = f"blob {len(content)}\0".encode()
            return hashlib.sha1(prefix + content).hexdigest() == blob
        if digest:
            return hashlib.sha256(path.read_bytes()).hexdigest() == digest
        return True

    if destination.exists() and expected_size is not None and verify(destination):
        return destination
    part = destination.with_suffix(destination.suffix+".part")
    offset = part.stat().st_size if resume and part.exists() else 0
    if offset and expected_size is not None and offset >= expected_size:
        if verify(part):
            part.replace(destination)
            return destination
        offset = 0
    request_headers = dict(headers or {})
    if offset:
        request_headers["Range"] = f"bytes={offset}-"
    with client.get(url, stream=True, timeout=(15, 90), headers=request_headers) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        if response.status_code == 206 and (not append
                or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
            raise ValueError(f"Invalid download range for {destination.name}")
        with part.open("ab" if append else "wb") as file:
            for chunk in response.iter_content(1024*1024):
                file.write(chunk)
        if not verify(part):
            raise ValueError(f"Download size or Git blob digest mismatch for {destination.name}")
        if destination.suffix == ".xlsx":
            import zipfile
            with zipfile.ZipFile(part) as archive:
                archive.getinfo("xl/workbook.xml")
        part.replace(destination)
    return destination


def china(cfg, dataset):
    root = Path(cfg["root"])/"raw"
    if dataset in {"state_grid", "all"}:
        with session() as client:
            for site, name in STATE_FILES.items():
                blob, size = STATE_BLOBS[site]
                file = root/"state_grid"/name
                path = f"data_original/solar_stations/{quote(name)}"
                api = "https://api.github.com/repos/Bob05757/Renewable-energy-generation-input-feature-variables-analysis"
                try:
                    fetch(client, f"{api}/contents/{path}?ref={STATE_COMMIT}", file, size, blob,
                          headers={"Accept": "application/vnd.github.raw+json"})
                except (requests.RequestException, ValueError):
                    fetch(client, f"{STATE_REPO}/{STATE_COMMIT}/{path}", file, size, blob)
                print(file, flush=True)
    if dataset in {"pvod", "all"}:
        folder = root/"pvod"
        with session() as client:
            path = fetch(client, PVOD_V4, folder/"PVODdatasets_v1.0.zip", PVOD_SIZE, digest=PVOD_SHA256)
        with ZipFile(path) as archive:
            wanted = {"metadata.csv", *(f"station{num:02}.csv" for num in range(10))}
            names = {Path(entry).name: entry for entry in archive.namelist() if Path(entry).name in wanted}
            if names.keys() != wanted:
                raise ValueError(f"Unexpected PVOD archive layout. Verify the publisher record: {PVOD_LANDING}")
            for name, entry in names.items():
                destination = folder/name
                valid = (destination.exists() and zlib.crc32(destination.read_bytes()) == archive.getinfo(entry).CRC)
                if not valid:
                    with atomic_destination(destination) as temporary:
                        with archive.open(entry) as source, temporary.open("wb") as output:
                            shutil.copyfileobj(source, output)
        print(f"PVOD V4 published station files: {folder}", flush=True)


def list_oedi(client, prefix):
    query = {"list-type": "2", "prefix": prefix}
    while True:
        response = client.get(OEDI, params=query, timeout=(15, 30))
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        for item in root.findall("{*}Contents"):
            yield item.findtext("{*}Key"), int(item.findtext("{*}Size"))
        token = root.findtext("{*}NextContinuationToken")
        if not token:
            break
        query["continuation-token"] = token


def pvdaq(cfg, ids, download=False, match=None, max_bytes=None, workers=4, chunk_mb=8):
    root = Path(cfg["root"])/"raw"/"pvdaq"
    patterns = cfg["pvdaq"]["download_patterns"]
    with session() as client:
        for site in ids:
            prefix = f"{PRIZE}{site}{'_5_min' if site == '7333' else ''}_OEDI/"
            pattern = re.compile(match or patterns[str(site)], re.I)
            selected = [(key, size) for key, size in list_oedi(client, prefix) if size and pattern.search(key)]
            if not selected:
                raise ValueError(f"Nothing matched the archive inventory under {prefix}.")
            total = sum(size for _, size in selected)
            print(f"{site}: {len(selected)} objects, {total:,} bytes", flush=True)
            if download and max_bytes is not None and total > max_bytes:
                raise ValueError(f"{site}: selected files exceed --max-bytes.")
            for key, size in selected:
                if download:
                    url, destination = f"{OEDI}/{quote(key)}", root/Path(key).name
                    if workers > 1 and size > chunk_mb*1024**2:
                        from .transfers import ranged_download
                        file = ranged_download(url, destination, size, workers, chunk_mb*1024**2)
                    else:
                        file = fetch(client, url, destination, size, resume=True)
                    print(file, flush=True)
                else:
                    print(f"{size:>12}  {key}", flush=True)

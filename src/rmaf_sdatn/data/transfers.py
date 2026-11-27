"""Bounded parallel HTTP range transfers with verified, reusable parts."""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from ..serialization import write_json
from ..locking import file_lock


def ranged_download(url, destination, size, workers=4, chunk_bytes=8*1024**2, timeout=90):
    lock = Path(destination).with_suffix(Path(destination).suffix+".lock")
    with file_lock(lock, timeout=3600):
        return _ranged_download(url, destination, size, workers, chunk_bytes, timeout)


def _ranged_download(url, destination, size, workers, chunk_bytes, timeout):
    if workers < 1 or chunk_bytes < 1:
        raise ValueError("Download workers and chunk size must be positive.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size:
        return destination
    response = requests.head(url, timeout=(15, timeout))
    response.raise_for_status()
    if int(response.headers.get("Content-Length", -1)) != size:
        raise ValueError("Remote object size differs from the inventory.")
    etag = response.headers.get("ETag")
    folder = destination.with_suffix(destination.suffix+".parts")
    folder.mkdir(exist_ok=True)
    identity = {"url": url, "size": size, "etag": etag, "chunk_bytes": chunk_bytes}
    manifest = folder/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
        raise ValueError(f"Download source changed. Choose a new destination or remove obsolete parts at {folder}.")
    write_json(identity, manifest)
    connections = threading.local()

    def transfer(start):
        end = min(start + chunk_bytes, size)-1
        part = folder/f"{start:012d}.bin"
        if part.exists() and part.stat().st_size == end-start+1:
            return part
        temporary = part.with_suffix(".tmp")
        headers = {"Accept-Encoding": "identity"}
        if etag:
            headers["If-Match"] = etag
        last_error = None
        if not hasattr(connections, "client"):
            connections.client = requests.Session()
        for _ in range(3):
            try:
                present = temporary.stat().st_size if temporary.exists() else 0
                if present > end-start+1:
                    temporary.unlink()
                    present = 0
                if present == end-start+1:
                    os.replace(temporary, part)
                    return part
                resume_at = start + present
                headers["Range"] = f"bytes={resume_at}-{end}"
                started = time.monotonic()
                with connections.client.get(url, headers=headers, stream=True, timeout=(45, timeout)) as stream:
                    stream.raise_for_status()
                    expected = f"bytes {resume_at}-{end}/{size}"
                    if stream.status_code != 206 or stream.headers.get("Content-Range") != expected:
                        raise ValueError(f"Server returned an invalid range for {start}: {stream.headers.get('Content-Range')}")
                    if etag and stream.headers.get("ETag") != etag:
                        raise ValueError("Remote object changed during the transfer.")
                    with temporary.open("ab" if present else "wb") as output:
                        for chunk in stream.iter_content(256*1024):
                            output.write(chunk)
                            if time.monotonic()-started > timeout:
                                raise requests.Timeout("Part transfer exceeded its time limit; partial bytes are retained.")
                if temporary.stat().st_size != end-start+1:
                    raise ValueError(f"Incomplete download part at offset {start}.")
                os.replace(temporary, part)
                return part
            except (requests.RequestException, OSError, ValueError) as error:
                last_error = error
        raise RuntimeError(f"Download failed at offset {start}. Parts are retained for retry.") from last_error

    starts = list(range(0, size, chunk_bytes))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(transfer, start) for start in starts]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % max(1, len(starts)//10) == 0 or completed == len(starts):
                print(f"{destination.name}: {completed}/{len(starts)} parts", flush=True)
    assembled = destination.with_suffix(destination.suffix+".part")
    with assembled.open("wb") as output:
        for start in starts:
            with (folder/f"{start:012d}.bin").open("rb") as part:
                while chunk := part.read(1024**2):
                    output.write(chunk)
    if assembled.stat().st_size != size:
        raise ValueError("Assembled object has an unexpected size.")
    os.replace(assembled, destination)
    # The final object is complete. Delete only explicitly constructed part files.
    for start in starts:
        (folder/f"{start:012d}.bin").unlink()
    manifest.unlink()
    for temporary in folder.glob("*.tmp"):
        temporary.unlink()
    folder.rmdir()
    return destination

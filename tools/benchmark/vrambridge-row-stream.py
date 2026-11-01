#!/usr/bin/env python3
"""Bounded exact row gathering: page cache, concurrent positional reads and explicit readiness."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, wait
import mmap
import os
from pathlib import Path
import threading


def row_plan(rows, row_bytes, tensor_offset, tensor_rows, page_size=4096):
    if (type(row_bytes) is not int or row_bytes <= 0 or type(tensor_offset) is not int or tensor_offset < 0 or
            type(tensor_rows) is not int or tensor_rows <= 0 or type(page_size) is not int or page_size <= 0):
        raise ValueError("Invalid tensor dimensions or page size.")
    spans = []
    pages = set()
    for row in rows:
        if type(row) is not int or not 0 <= row < tensor_rows:
            raise ValueError("Row ID is outside the tensor.")
        begin = tensor_offset + row * row_bytes
        end = begin + row_bytes
        spans.append((begin, end))
        pages.update(range(begin // page_size * page_size, (end - 1) // page_size * page_size + 1, page_size))
    return spans, sorted(pages)


def merge_pages(pages, page_size):
    ranges = []
    for begin in sorted(set(pages)):
        if ranges and ranges[-1][1] == begin:
            ranges[-1] = ranges[-1][0], begin + page_size
        else:
            ranges.append((begin, begin + page_size))
    return ranges


class PendingRows:
    def __init__(self, owner, spans, pages, hit_pages, miss_pages):
        self.owner = owner
        self.spans = spans
        self.pages = pages
        self.hit_pages = hit_pages
        self.miss_pages = miss_pages
        self.finished = False

    def gather(self):
        if self.finished:
            raise RuntimeError("A row request can be consumed only once.")
        try:
            resolved = {}
            for offset, future in self.pages.items():
                data = future.result()
                resolved[offset] = data
                if self.owner.capacity:
                    self.owner.cache[offset] = data
                    self.owner.cache.move_to_end(offset)
                    while len(self.owner.cache) > self.owner.capacity:
                        self.owner.cache.popitem(last=False)
            output = []
            for begin, end in self.spans:
                pieces = []
                while begin < end:
                    page = begin // self.owner.page_size * self.owner.page_size
                    count = min(end - begin, self.owner.page_size - (begin - page))
                    pieces.append(resolved[page][begin - page:begin - page + count])
                    begin += count
                row = b"".join(pieces)
                if len(row) != self.owner.row_bytes:
                    raise RuntimeError("Gather returned a truncated row.")
                output.append(row)
            return output
        finally:
            # Failed reads may still own buffers; drain them before accepting another request or closing the file.
            if self.pages:
                wait(self.pages.values())
            self.pages.clear()
            self.finished = True
            self.owner.pending = None


class RowStore:
    def __init__(self, path: Path, *, row_bytes: int, tensor_offset: int, tensor_rows: int,
                 cache_pages=256, workers=8, max_rows=48, max_pending_pages=128, direct=False):
        self.page_size = mmap.PAGESIZE
        row_plan([], row_bytes, tensor_offset, tensor_rows, self.page_size)
        if (type(cache_pages) is not int or cache_pages < 0 or type(workers) is not int or not 1 <= workers <= 32 or
                type(max_rows) is not int or max_rows <= 0 or
                type(max_pending_pages) is not int or max_pending_pages <= 0):
            raise ValueError("Invalid cache, worker or request bounds.")
        if direct and not hasattr(os, "O_DIRECT"):
            raise RuntimeError("Direct reads are unsupported on this platform.")
        self.fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if direct else 0))
        self.identity = os.fstat(self.fd)
        if tensor_offset + row_bytes * tensor_rows > self.identity.st_size:
            os.close(self.fd)
            raise ValueError("Tensor extends beyond the backing file.")
        self.row_bytes, self.tensor_offset, self.tensor_rows = row_bytes, tensor_offset, tensor_rows
        self.capacity, self.max_rows, self.max_pending_pages = cache_pages, max_rows, max_pending_pages
        self.workers, self.direct = workers, direct
        self.cache = OrderedDict()
        self.pending = None
        self.local = threading.local()
        self.buffers = []
        self.buffer_lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.closed = False
        self.peak_requested_pages = 0
        self.submitted_bytes = 0

    @property
    def payload_memory_bound(self):
        return ((self.capacity + self.max_pending_pages + self.workers) * self.page_size +
                (self.max_rows + 2) * self.row_bytes)

    def _read_page(self, offset):
        expected = min(self.page_size, self.identity.st_size - offset)
        if self.direct:
            if not hasattr(self.local, "buffer"):
                buffer = mmap.mmap(-1, self.page_size)
                with self.buffer_lock:
                    self.buffers.append(buffer)
                self.local.buffer = buffer
            buffer = self.local.buffer
            count = os.preadv(self.fd, [buffer], offset)
            data = buffer[:count]
        else:
            data = os.pread(self.fd, self.page_size, offset)
        if len(data) != expected:
            raise RuntimeError(f"Short positional read at offset {offset}: {len(data)} instead of {expected}.")
        return data

    def submit(self, rows):
        if self.closed or self.pending is not None:
            raise RuntimeError("Close or consume the previous request before submitting another.")
        if len(rows) > self.max_rows:
            raise ValueError("Request exceeds the bounded row count.")
        spans, offsets = row_plan(rows, self.row_bytes, self.tensor_offset, self.tensor_rows, self.page_size)
        if len(offsets) > self.max_pending_pages:
            raise ValueError("Request exceeds the bounded page count.")
        current = os.fstat(self.fd)
        if (current.st_size, current.st_mtime_ns) != (self.identity.st_size, self.identity.st_mtime_ns):
            raise RuntimeError("Backing file changed during row streaming.")
        futures = {}
        hits = 0
        for offset in offsets:
            if offset in self.cache:
                future = Future()
                future.set_result(self.cache[offset])
                self.cache.move_to_end(offset)
                hits += 1
            else:
                future = self.pool.submit(self._read_page, offset)
                self.submitted_bytes += min(self.page_size, self.identity.st_size - offset)
            futures[offset] = future
        self.peak_requested_pages = max(self.peak_requested_pages, len(offsets))
        self.pending = PendingRows(self, spans, futures, hits, len(offsets) - hits)
        return self.pending

    def close(self):
        if self.closed:
            return
        self.pool.shutdown(wait=True)
        for buffer in self.buffers:
            buffer.close()
        os.close(self.fd)
        self.cache.clear()
        if self.pending is not None:
            self.pending.pages.clear()
            self.pending.finished = True
            self.pending = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

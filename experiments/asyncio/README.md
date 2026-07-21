# Asyncio

Demonstrations of Python's `asyncio` library, benchmarking asynchronous I/O against its synchronous equivalent.

***

## Content

### `api.py` — HTTP Requests

Fetches 99 pages from a public REST API both synchronously (using `requests`) and asynchronously (using `aiohttp` with `asyncio.gather`). The synchronous version processes each request in series; the asynchronous version fans all requests out simultaneously and awaits them together.

**Result:** the asynchronous approach is significantly faster for network I/O, where latency dominates and the GIL is not a bottleneck.

### `file.py` — File Writes

Writes 1,000 files both synchronously and asynchronously. The asynchronous version uses `loop.run_in_executor` to offload each blocking write call to a thread pool.

**Result:** the asynchronous version does not outperform the synchronous one. File writes are CPU-bound blocking syscalls; the overhead of managing a thread pool per write negates any concurrency benefit.

***

## Running

```bash
pip install -r requirements.txt

# HTTP benchmark
python api.py

# File write benchmark (fetches API data first, then cleans up)
python file.py
```

***

## Key Takeaway

`asyncio` shines for **I/O-bound work with high latency** (network calls, database queries). It brings little to no benefit for **blocking syscalls** like disk writes, where offloading to a thread pool adds overhead without meaningful parallelism.

"""
file.py — Synchronous vs. Asynchronous File Writes
====================================================

Benchmarks writing 1,000 files both synchronously and asynchronously.
API response data (fetched via `api.py`) is used as the file content so
that each write has a realistic, non-trivial payload.

The asynchronous version uses `loop.run_in_executor` to offload each
blocking `open`/`write` call to a thread-pool executor. All writes are
then awaited concurrently via `asyncio.gather`.

Run this script directly to print elapsed wall-clock time for each
approach. Written files are cleaned up automatically after the benchmark.

Key finding
-----------
Unlike network I/O, file writes are blocking syscalls. The overhead of
dispatching each write to a separate thread negates any concurrency benefit,
so the asynchronous approach does not outperform the synchronous one here.
"""

import os
import time
import asyncio
from api import main as api_main


N_FILES = 1000


def write_file(data, filename):
    """
    Write data to a file synchronously.

    Parameters
    ----------
    data : list of str
        Lines to write.
    filename : str
        Destination file path.
    """
    with open(filename, "w") as f:
        f.writelines(data)


async def write_file_async(data, filename):
    """
    Write data to a file by offloading the blocking write to a thread
    pool executor, keeping the event loop free during the operation.

    Parameters
    ----------
    data : list of str
        Lines to write.
    filename : str
        Destination file path.
    """
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, write_file, data, filename)

    except Exception as e:
        print(e)


async def main(items):
    """
    Schedule all file writes concurrently and await their completion.

    Parameters
    ----------
    items : list of str
        Content to write to each file.

    Returns
    -------
    list of str
        The original items (passed through for convenience).
    """
    tasks = [write_file_async(items, f"file{i}.txt") for i in range(N_FILES)]
    await asyncio.gather(*tasks)

    return items


if __name__ == "__main__":

    items = asyncio.run(api_main())

    start = time.time()
    for i in range(N_FILES):
        write_file(items, f"file{i}.txt")

    print("synchronous:")
    print(time.time() - start)

    start = time.time()
    asyncio.run(main(items))
    print("asynchronous:")
    print(time.time() - start)
    # write operation is blocking and the overhead of executing in a separate thread is not worth it

    for i in range(N_FILES):
        os.remove(f"file{i}.txt")

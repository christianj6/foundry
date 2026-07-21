"""
api.py — Synchronous vs. Asynchronous HTTP Requests
=====================================================

Benchmarks fetching 99 pages from a public REST API using two approaches:

- **Synchronous** (`requests`): each request blocks until the response is
  received before the next one is issued.
- **Asynchronous** (`aiohttp` + `asyncio.gather`): all requests are issued
  concurrently; the event loop awaits all responses together.

Run this script directly to print elapsed wall-clock time for each approach.
"""

import time
import asyncio
import aiohttp
import requests


ENDPOINT = "https://jsonplaceholder.typicode.com/todos/"
PAGES = list(range(1, 100))


def request_api(url):
    """
    Fetch a single URL synchronously.

    Parameters
    ----------
    url : str
        The URL to request.

    Returns
    -------
    bytes
        Raw response content.
    """
    return requests.get(url).content


async def request_api_async(url):
    """
    Fetch a single URL asynchronously using aiohttp.

    Parameters
    ----------
    url : str
        The URL to request.

    Returns
    -------
    str
        Response body as text.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()


async def main():
    """
    Schedule all async requests concurrently and await their completion.

    Returns
    -------
    list of str
        Response bodies for all requested pages.
    """
    tasks = [request_api_async(ENDPOINT + str(p)) for p in PAGES]

    return await asyncio.gather(*tasks)


if __name__ == "__main__":

    start = time.time()
    responses_async = asyncio.get_event_loop().run_until_complete(main())
    print("asynchronous:")
    print(time.time() - start)

    start = time.time()
    responses_sync = [request_api(ENDPOINT + str(p)) for p in PAGES]
    print("synchronous:")
    print(time.time() - start)

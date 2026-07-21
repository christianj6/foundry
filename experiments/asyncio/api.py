import time
import asyncio
import aiohttp
import requests


ENDPOINT = "https://jsonplaceholder.typicode.com/todos/"
PAGES = list(range(1, 100))


def request_api(url):
    return requests.get(url).content


async def request_api_async(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()


async def main():
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

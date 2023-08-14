import os
import time
import asyncio
from api import main as api_main


N_FILES = 1000


def write_file(data, filename):
    with open(filename, "w") as f:
        f.writelines(data)


async def write_file_async(data, filename):
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, write_file, data, filename)

    except Exception as e:
        print(e)


async def main(items):
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

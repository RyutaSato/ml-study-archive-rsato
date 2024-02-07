"""全ての実験プログラムのエントリポイント
このプログラムは、原則変更禁止。
破壊的変更を加える際は、メジャーバージョンを上げること。

Supported BaseFlow: 2.0.0
"""
import json
from multiprocessing import Process, Queue, Lock, cpu_count

from schemas import Params

import uvicorn
from fastapi import FastAPI
from _main import worker

app = FastAPI()
queue = Queue()
lock = Lock()
processes = []
not_finished = []
process_num = 8  # cpu_count() - 1
if process_num <= 1:
    process_num = 1
    from threading import Thread as Process
    from threading import Lock
    from queue import Queue

    lock = Lock()
    queue = Queue()


@app.on_event("startup")
def startup_event():
    for _ in range(process_num):
        p = Process(target=worker, args=(queue, lock))
        p.start()
        processes.append(p)


@app.post("/in_queue")
def run(params: Params):
    queue.put(params)
    return {"message": "ok"}


@app.get("/kill", status_code=200)
def kill():
    while not queue.empty():
        not_finished.append(queue.get().json())
    for _ in processes:
        queue.put(None)
    return {"message": "ok"}


@app.get("/new", status_code=200)
def new():
    p = Process(target=worker, args=(queue, lock))
    p.start()
    processes.append(p)
    return {"message": "ok", "total": len(processes)}


@app.on_event("shutdown")
def shutdown_event():
    while not queue.empty():
        not_finished.append(queue.get().json())
    # Read existing data from the file
    try:
        with open("results/not_finished.json", "r") as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []
    # Add new data to the existing list
    existing_data.extend(not_finished)

    # Write the updated list back to the file
    with open("results/not_finished.json", "w") as f:
        json.dump(existing_data, f, indent=4)

    for _ in processes:
        queue.put(None)
    for p in processes:
        p.join()


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8080)

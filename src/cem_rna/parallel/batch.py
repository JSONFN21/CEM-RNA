from collections import deque
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import math
import multiprocessing
import time
import traceback
from ..scorer import score_batch
from ..checks import check_workers
from .parallel_scorer import _worker_environment


class PoolClient:
    def __init__(self, connection):
        self.connection = connection

    def score(self, *arguments):
        self.connection.send(('score', arguments))
        _, values, cpu_seconds = self.connection.recv()
        return values, cpu_seconds


def _optimize(target, config, connection, optimizer):
    try:
        result = optimizer(target, config, pool=PoolClient(connection)).run()
        connection.send(('done', result))
    except Exception:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


def optimize_targets(targets, *, config, workers, concurrent_puzzles, seeds=None,
                     optimizer=None, score_function=score_batch, sequence_index=1):
    from ..cem import CEM
    seeds = list(seeds) if seeds is not None else None
    check_workers(workers, concurrent_puzzles, seeds)
    optimizer = optimizer or CEM
    targets = list(targets)
    if not targets:
        return []
    settings = [config] if seeds is None else [replace(config, seed=seed) for seed in seeds]
    jobs = [(target, setting) for target in targets for setting in settings]
    if not jobs:
        return []
    if workers == 1:
        return [optimizer(target, setting).run() for target, setting in jobs]

    _worker_environment()
    context = multiprocessing.get_context('spawn')
    limit = min(len(jobs), concurrent_puzzles or workers * 2)
    queued = deque(enumerate(jobs))
    results = [None] * len(jobs)
    optimizers, connections, requests, futures = {}, {}, {}, {}
    ready = deque()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_worker_environment)

    def launch():
        while queued and len(connections) < limit:
            index, (target, setting) = queued.popleft()
            parent, child = context.Pipe()
            process = context.Process(target=_optimize, args=(target, setting, child, optimizer))
            process.start()
            child.close()
            connections[index], optimizers[index] = parent, process

    def finish(index):
        connections.pop(index).close()
        process = optimizers.pop(index)
        process.join()
        process.close()

    try:
        launch()
        while connections or queued:
            for index, connection in list(connections.items()):
                if not connection.poll():
                    continue
                message, value = connection.recv()
                if message == 'done':
                    results[index] = value
                    finish(index)
                elif message == 'error':
                    raise RuntimeError(value)
                else:
                    arguments = list(value)
                    sequences = arguments[sequence_index]
                    count = max(2, math.ceil(2 * workers / len(connections)))
                    size = max(1, min(32, len(sequences) // count))
                    chunks = [sequences[i:i + size] for i in range(0, len(sequences), size)]
                    requests[index] = dict(arguments=arguments, chunks=deque(enumerate(chunks)), values=[None] * len(chunks), remaining=len(chunks), cpu=0.0)
                    ready.append(index)
            launch()
            while ready and len(futures) < workers:
                index = ready.popleft()
                request = requests[index]
                chunk_index, chunk = request['chunks'].popleft()
                arguments = list(request['arguments'])
                arguments[sequence_index] = chunk
                future = pool.submit(score_function, *arguments)
                futures[future] = index, chunk_index
                if request['chunks']:
                    ready.append(index)
            for future in [f for f in futures if f.done()]:
                index, chunk_index = futures.pop(future)
                values, cpu = future.result()
                request = requests[index]
                request['values'][chunk_index] = values
                request['cpu'] += cpu
                request['remaining'] -= 1
                if request['remaining'] == 0:
                    values = [value for chunk in request['values'] for value in chunk]
                    connections[index].send(('scored', values, request['cpu']))
                    del requests[index]
            for index, process in optimizers.items():
                if process.exitcode is not None and not connections[index].poll():
                    raise RuntimeError(f'Optimizer for target {index} exited without a result')
            if futures:
                wait(list(futures), timeout=0.002, return_when=FIRST_COMPLETED)
            else:
                time.sleep(0.002)
    finally:
        for process in optimizers.values():
            if process.is_alive():
                process.terminate()
            process.join()
            process.close()
        for connection in connections.values():
            connection.close()
        pool.shutdown(wait=True, cancel_futures=True)
    return results

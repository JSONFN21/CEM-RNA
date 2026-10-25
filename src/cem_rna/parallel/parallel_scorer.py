# one pool of scoring workers for all the puzzles
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import threading
from ..scorer import score_batch
from ..checks import check_workers


# give each scoring task one cpu without starting more thread teams
def _worker_environment():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'


class ScoringPool:
    def __init__(self, workers=1, *, score_function=score_batch, sequence_index=1):
        check_workers(workers)
        self.workers = workers
        self.score_function, self.sequence_index = score_function, sequence_index
        self.executor = None
        self.slots = threading.Semaphore(workers * 2)

    def __enter__(self):
        if self.workers > 1:
            _worker_environment()
            self.executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context('spawn'), initializer=_worker_environment)
        return self

    def _submit(self, *args):
        self.slots.acquire()
        try:
            future = self.executor.submit(self.score_function, *args)
        except BaseException:
            self.slots.release()
            raise
        future.add_done_callback(lambda _: self.slots.release())
        return future

    # split the batch into small jobs and keep the scoring settings as they are
    def score(self, *arguments):
        if self.executor is None:
            return self.score_function(*arguments)
        sequences = arguments[self.sequence_index]
        count = min(len(sequences), self.workers * 2)
        size = max(1, (len(sequences) + count - 1) // count)
        futures = []
        for i in range(0, len(sequences), size):
            chunk = list(arguments)
            chunk[self.sequence_index] = sequences[i:i + size]
            futures.append(self._submit(*chunk))
        batches = [future.result() for future in futures]
        return [value for values, _ in batches for value in values], sum(cpu for _, cpu in batches)

    def __exit__(self, *exception):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

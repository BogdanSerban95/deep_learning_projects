import multiprocessing
from rectangle_detection.config import *


'''
We need to define a class that instantiates a set of processes that each run a function
These processes handle a shared resource in which they append data
We need a 'master' class that spawns all the processes
'''


class LoaderWorker(multiprocessing.Process):
    def __init__(self, queue, worker_func, fn_args):
        multiprocessing.Process.__init__(self)
        self.queue = queue
        self.worker_func = worker_func
        self.fn_args = fn_args

    def run(self):
        self.worker_func(self.queue, **self.fn_args)


class BaseDataLoader(object):
    def __init__(self, num_workers, worker_func, worker_func_args):
        self.queue = multiprocessing.Queue(MAX_QUEUE_SIZE)
        self.workers = []
        for i in range(num_workers):
            self.workers.append(LoaderWorker(queue=self.queue, worker_func=worker_func, fn_args=worker_func_args))
            self.workers[i].start()

    def get_next_batch(self, batch_size):
        for i in range(batch_size):
            yield self.queue.get(True)

    def stop_all(self):
        for worker in self.workers:
            worker.terminate()


class DataLoader(object):
    def __init__(self):
        pass



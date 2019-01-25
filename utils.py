import os
from datetime import datetime
import multiprocessing
from multiprocessing import Process
import pandas as pd
import os.path as osp
import shutil


def time_this(func):
    def new_func(*args, **kwargs):
        before = datetime.now()
        x = func(*args, **kwargs)
        after = datetime.now()
        print("Function {} elapsed Time: {}".format(func.__name__, after-before))
        return x
    return new_func


@time_this
def multiprocess_parse(data, func):
    num_processes = multiprocessing.cpu_count() - 1

    def group_async_task(params, queue):
        queue.put(func(params))

    def run_multiprocess():
        queue = multiprocessing.Queue()
        ntask_per_process = data.__len__() // num_processes + 1
        p = []
        for i in range(num_processes):
            sub_p = data[i * ntask_per_process : min(i * ntask_per_process + ntask_per_process, data.__len__())]
            p.append(Process(target=group_async_task, args=(sub_p,queue,)))
            p[-1].start()

        df = []
        for task_group in p:
            df.append(queue.get())
        for task_group in p:
            task_group.join()
        return df

    return run_multiprocess()


def create_folder(fold_path, exist_ok=True):
    if osp.exists(fold_path) and not exist_ok:
        shutil.rmtree(fold_path)
    if not osp.exists(fold_path):
        os.makedirs(fold_path)


import numpy as np
import time
import sys

class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()                 # 强行刷新缓冲区以防止由于缓冲区没满而不打印的情况

    def flush(self):
        self.terminal.flush()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

class AverageMeterList:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]       # 第一个为loss的和，第二个为第几个step

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v

class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:        # print开始时间
            print(starting_msg, time.ctime(time.time()))


    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress            # 训练所有step所需的总时间
        self.est_remaining = self.est_total - self.elapsed  # 训练所有step剩余时间
        self.est_finish = int(self.start + self.est_total)  # 训练所有step结束时间


    def str_est_finish(self):
        return str(time.ctime(self.est_finish))

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

from multiprocessing.pool import ThreadPool

class BatchThreader:

    def __init__(self, func, args_list, batch_size, prefetch_size=4, processes=12):
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size

        self.pool = ThreadPool(processes=processes)
        self.async_result = []

        self.func = func
        self.left_args_list = args_list
        self.n_tasks = len(args_list)

        # initial work
        self.__start_works(self.__get_n_pending_works())


    def __start_works(self, times):
        for _ in range(times):
            args = self.left_args_list.pop(0)
            self.async_result.append(
                self.pool.apply_async(self.func, args))


    def __get_n_pending_works(self):
        return min((self.prefetch_size + 1) * self.batch_size - len(self.async_result)
                   , len(self.left_args_list))



    def pop_results(self):

        n_inwork = len(self.async_result)

        n_fetch = min(n_inwork, self.batch_size)
        rtn = [self.async_result.pop(0).get()
                for _ in range(n_fetch)]

        to_fill = self.__get_n_pending_works()
        if to_fill == 0:
            self.pool.close()
        else:
            self.__start_works(to_fill)

        return rtn







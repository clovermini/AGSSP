import torch
import psutil
import gc
import threading

'''
https://zhuanlan.zhihu.com/p/618894919
用了 TorchTracemalloc 上下文管理器，它可以方便地计算出 GPU 和 CPU 的消耗(以 MB 计)。
for epoch in range(num_epochs):
    with TorchTracemalloc() as tracemalloc:
        model.train()

        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            XXX
    # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage  以下单位均为 MB
    accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
    accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
    accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
    accelerator.print(
        "GPU Total Peak Memory consumed during the train (max): {}\n".format(
            tracemalloc.peaked + b2mb(tracemalloc.begin)
        )
    )

    accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
    accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
    accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
    accelerator.print(
        "CPU Total Peak Memory consumed during the train (max): {}\n".format(
            tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
        )
    )
'''

def b2mb(x):
    """ Converting Bytes to Megabytes. """
    return int(x / 2**20)


class TorchTracemalloc:
    """ This context manager is used to track the peak memory usage of the process. """

    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        # Reset the peak gauge to zero
        torch.cuda.reset_max_memory_allocated()

        # 返回当前的显存占用
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()

        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()

        return self

    def cpu_mem_used(self):
        """Get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()

        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()

        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)

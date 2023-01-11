import time
import habana_frameworks.torch as ht

class hpu_perf:
    def __init__(self, module, log=True, mark_step=True, memoryinfo=False, sync=False):
        if log:
            print(f" {module}: start")
        self.module = module
        self.stime = time.perf_counter()
        self.mark = mark_step
        self.mem = memoryinfo
        self.sync = sync
        self.log = log
        if self.mem:
            ht.hpu.reset_peak_memory_stats()
        self.prelog = None

    def checknow(self, log):
        if self.mark:
            ht.core.mark_step()
            if self.sync:
                ht.core.hpu.default_stream().synchronize()
        if self.mem:
            print(ht.hpu.memory_summary())

        tmp = time.perf_counter()
        if self.log:
            print(" {}: {} takes {:.2f} ms".format(self.module, log, (tmp - self.stime)*1000))
        self.stime = tmp

    def checkahead(self, log):
        if self.mark:
            ht.core.mark_step()
            if self.sync:
                ht.core.hpu.default_stream().synchronize()
        if self.mem:
            print(ht.hpu.memory_summary())

        tmp = time.perf_counter()
        if self.prelog is not None and self.log:
            print(" {}: {} takes {:.2f} ms".format(self.module, self.prelog, (tmp - self.stime)*1000))
        self.stime = tmp
        self.prelog = log

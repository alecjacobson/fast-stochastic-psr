"""Sample GPU-process ownership during benchmarks without touching other jobs."""
import os
import subprocess
import threading
import time


class GpuProcessAudit:
    """A sampled isolation check, not proof that no short-lived job overlapped."""
    def __init__(self, pci_bus_id, interval=1.):
        self.pci_bus_id = pci_bus_id
        self.interval = interval
        self.own_pid = os.getpid()
        self.started = time.perf_counter()
        self.samples = 0
        self.foreign = []
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        try:
            raw = subprocess.check_output(
                ['nvidia-smi','-i',self.pci_bus_id,'--query-compute-apps=pid',
                 '--format=csv,noheader,nounits'],text=True,timeout=5)
            pids = sorted({int(v.strip()) for v in raw.splitlines() if v.strip()})
            foreign = [pid for pid in pids if pid != self.own_pid]
            self.samples += 1
            if foreign:
                self.foreign.append({'seconds':time.perf_counter()-self.started,'pids':foreign})
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            self.errors.append({'seconds':time.perf_counter()-self.started,
                                'error':f'{type(exc).__name__}: {exc}'})

    def start(self):
        self.sample()
        def poll():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=poll,daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.sample()
        return {'pci_bus_id':self.pci_bus_id,'interval_seconds':self.interval,
                'samples':self.samples,'foreign_process_samples':self.foreign,
                'errors':self.errors,
                'sampled_isolation_pass':not self.foreign and not self.errors,
                'qualification':'Periodic process samples can miss short-lived GPU work; no other job was stopped.'}

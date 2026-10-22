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
        self.observations = []
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        begin=time.perf_counter()-self.started
        try:
            raw = subprocess.check_output(
                ['nvidia-smi','-i',self.pci_bus_id,'--query-compute-apps=pid,used_memory',
                 '--format=csv,noheader,nounits'],text=True,timeout=5)
            rows=[v.split(',') for v in raw.splitlines() if v.strip()]
            pids = sorted({int(v[0].strip()) for v in rows})
            own_memory=[float(v[1].strip())*1024**2 for v in rows if int(v[0].strip())==self.own_pid and v[1].strip().isdigit()]
            self.observations.append({'started_seconds':begin,'seconds':time.perf_counter()-self.started,
                                      'pids':pids,'own_memory_bytes':sum(own_memory) if own_memory else None})
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
                'observations':self.observations,'own_pid':self.own_pid,
                'sampled_isolation_pass':not self.foreign and not self.errors,
                'qualification':'Periodic process samples can miss short-lived GPU work; no other job was stopped.'}


def assess_gpu_interval(record,start,end):
    """Conservative sampled isolation for a measured interval, including anchors.

    Both a preceding and following process observation must exist. Include
    those entire observations, reject gaps >2 sampling intervals and any error
    in the covered window. This still cannot rule out short-lived missed jobs.
    """
    if not 0 <= start < end:
        raise ValueError('require 0 <= start < end')
    observations=record.get('observations',[])
    before=[i for i,o in enumerate(observations) if o['seconds']<=start]
    after=[i for i,o in enumerate(observations) if o['started_seconds']>=end]
    result={'start_seconds':start,'end_seconds':end,'sampled_isolation_pass':False,
            'qualification':'Periodic observations and bracketing samples; not proof of exclusive GPU use or exact memory peak.'}
    if not before or not after:
        return {**result,'reason':'missing bracketing observations'}
    covered=observations[before[-1]:after[0]+1]
    gap=max((b['started_seconds']-a['seconds'] for a,b in zip(covered,covered[1:])),default=0.)
    foreign=sorted({p for o in covered for p in o['pids'] if p!=record['own_pid']})
    errors=[e for e in record['errors'] if covered[0]['started_seconds']<=e['seconds']<=covered[-1]['seconds']]
    memory=[o['own_memory_bytes'] for o in covered if o['own_memory_bytes'] is not None]
    result.update(sampled_isolation_pass=not foreign and not errors and gap<=2*record['interval_seconds'],
                  foreign_pids=foreign,errors=errors,maximum_observation_gap_seconds=gap,
                  sampled_process_memory_peak_bytes=max(memory) if memory else None,
                  samples=len(covered))
    return result

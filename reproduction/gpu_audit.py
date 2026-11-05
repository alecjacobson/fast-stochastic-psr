"""Sample GPU-process ownership during benchmarks without touching other jobs."""
import os
import subprocess
import threading
import time
import ctypes
import uuid


class _NvmlDeviceProcesses:
    """Query the actual CUDA device, including a MIG compute instance.

    The v2 CUDA UUID distinguishes a MIG instance from its parent. NVML's
    per-instance process query works without permission to inspect other MIGs.
    No GPU/MIG configuration or process state is changed.
    """
    def __init__(self, cuda_device):
        c=ctypes
        cuda=c.CDLL('libcuda.so.1')
        if cuda.cuInit(0)!=0:raise RuntimeError('CUDA driver initialization failed')
        raw=(c.c_ubyte*16)()
        if cuda.cuDeviceGetUuid_v2(c.byref(raw),int(cuda_device))!=0:
            raise RuntimeError('CUDA v2 device UUID unavailable')
        suffix=str(uuid.UUID(bytes=bytes(raw)))
        self.lib=c.CDLL('libnvidia-ml.so.1')
        if self.lib.nvmlInit_v2()!=0:raise RuntimeError('NVML initialization failed')
        self.handle=c.c_void_p();self.closed=False
        for prefix in ('GPU-','MIG-'):
            name=prefix+suffix
            if self.lib.nvmlDeviceGetHandleByUUID(name.encode(),c.byref(self.handle))==0:
                self.uuid=name;break
        else:
            self.close();raise RuntimeError('NVML handle for CUDA device unavailable')

    def query(self):
        c=ctypes
        class Process(c.Structure):
            _fields_=[('pid',c.c_uint),('memory',c.c_ulonglong),
                      ('gpu_instance',c.c_uint),('compute_instance',c.c_uint)]
        capacity=16
        for _ in range(4):
            count=c.c_uint(capacity);values=(Process*capacity)()
            code=self.lib.nvmlDeviceGetComputeRunningProcesses_v3(self.handle,c.byref(count),values)
            if code==7: # NVML_ERROR_INSUFFICIENT_SIZE; list grew concurrently.
                capacity=max(2*capacity,count.value+8);continue
            if code!=0:raise RuntimeError(f'NVML process query failed with code {code}')
            return [(int(v.pid),None if v.memory==2**64-1 else int(v.memory)) for v in values[:count.value]]
        raise RuntimeError('NVML process list changed repeatedly')

    def close(self):
        if not self.closed:self.lib.nvmlShutdown();self.closed=True


class GpuProcessAudit:
    """A sampled isolation check, not proof that no short-lived job overlapped."""
    def __init__(self, pci_bus_id, interval=1., cuda_device=None):
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
        self.device_processes=None
        self.backend_error=None
        if cuda_device is not None:
            try:self.device_processes=_NvmlDeviceProcesses(cuda_device)
            except (OSError,RuntimeError,AttributeError) as exc:self.backend_error=str(exc)

    def sample(self):
        begin=time.perf_counter()-self.started
        try:
            if self.backend_error:raise RuntimeError(self.backend_error)
            if self.device_processes is not None:
                rows=self.device_processes.query()
                pids=sorted({pid for pid,_ in rows})
                own_memory=[memory for pid,memory in rows if pid==self.own_pid and memory is not None]
            else:
                raw = subprocess.check_output(
                    ['nvidia-smi','-i',self.pci_bus_id,'--query-compute-apps=pid,used_memory',
                     '--format=csv,noheader,nounits'],text=True,timeout=5)
                rows=[v.split(',') for v in raw.splitlines() if v.strip()]
                pids = sorted({int(v[0].strip()) for v in rows})
                own_memory=[float(v[1].strip())*1024**2 for v in rows if int(v[0].strip())==self.own_pid and v[1].strip().isdigit()]
            self.observations.append({'started_seconds':begin,'seconds':time.perf_counter()-self.started,
                                      'pids':pids,'own_memory_bytes':sum(own_memory) if own_memory else None,
                                      'own_host_rss_bytes':int(open('/proc/self/statm').read().split()[1])*os.sysconf('SC_PAGE_SIZE')})
            foreign = [pid for pid in pids if pid != self.own_pid]
            self.samples += 1
            if foreign:
                self.foreign.append({'seconds':time.perf_counter()-self.started,'pids':foreign})
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
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
        if self.device_processes is not None:self.device_processes.close()
        return {'pci_bus_id':self.pci_bus_id,'interval_seconds':self.interval,
                'process_query_backend':'nvml_cuda_device' if self.device_processes is not None else 'nvidia-smi',
                'device_uuid':self.device_processes.uuid if self.device_processes is not None else None,
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

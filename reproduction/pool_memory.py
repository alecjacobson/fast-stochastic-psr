"""Event-based CuPy pool live-memory measurement, excluding native allocations."""
import cupy as cp
from cupy.cuda.memory_hook import MemoryHook


class PoolLivePeak(MemoryHook):
    """Count rounded malloc/free events on the calling thread's memory pool.

    This is not a device-memory peak: native NUFFT plans, CUDA contexts and
    allocations in other threads/allocators are outside the hook. End-state
    agreement with the default pool is recorded, not assumed.
    """
    name='GeoSPSRPoolLivePeak'

    def __init__(self):
        self.initial=cp.get_default_memory_pool().used_bytes()
        self.live=self.peak=self.initial
        self.allocations=0

    def malloc_postprocess(self,**kwargs):
        if kwargs['mem_ptr']:
            self.live+=kwargs['mem_size'];self.peak=max(self.peak,self.live)
            self.allocations+=1

    def free_postprocess(self,**kwargs):
        if kwargs['mem_ptr']:self.live-=kwargs['mem_size']

    def record(self):
        actual=cp.get_default_memory_pool().used_bytes()
        return {'initial_live_bytes':self.initial,'peak_live_bytes':self.peak,
                'final_tracked_live_bytes':self.live,'final_pool_live_bytes':actual,
                'end_state_matches':actual==self.live,'allocation_events':self.allocations,
                'scope':'calling-thread CuPy default-pool rounded live allocations; excludes native/device peak'}

import cupy as cp
from reproduction.pool_memory import PoolLivePeak


def test_live_peak_counts_reuse_and_free_with_existing_allocation():
    held=cp.empty(513,dtype='f')
    hook=PoolLivePeak()
    with hook:
        first=cp.empty(513,dtype='f')
        second=cp.empty(1031,dtype='d')
        actual_peak=cp.get_default_memory_pool().used_bytes()
        del second
        recycled=cp.empty(1031,dtype='d')
        del first,recycled
    r=hook.record()
    assert r['end_state_matches'] and r['peak_live_bytes']==actual_peak
    assert r['final_tracked_live_bytes']==r['initial_live_bytes']
    assert r['allocation_events']==3
    del held

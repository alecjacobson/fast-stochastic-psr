from reproduction.gpu_audit import assess_gpu_interval


def test_device_audit_keeps_foreign_processes_and_errors(monkeypatch):
    import reproduction.gpu_audit as audit

    class Device:
        uuid = 'MIG-test'
        closed = False
        def __init__(self, ordinal):
            assert ordinal == 0
        def query(self):
            return [(audit.os.getpid(), 1024), (123456789, 2048)]
        def close(self):
            self.closed = True

    monkeypatch.setattr(audit, '_NvmlDeviceProcesses', Device)
    monitor = audit.GpuProcessAudit('unused', cuda_device=0)
    record = monitor.stop()
    assert record['device_uuid'] == 'MIG-test'
    assert not record['sampled_isolation_pass']
    assert record['observations'][0]['own_memory_bytes'] == 1024
    assert monitor.device_processes.closed

    def unavailable(_):
        raise RuntimeError('permission denied')
    monkeypatch.setattr(audit, '_NvmlDeviceProcesses', unavailable)
    record = audit.GpuProcessAudit('unused', cuda_device=0).stop()
    assert not record['sampled_isolation_pass']
    assert 'permission denied' in record['errors'][0]['error']


def test_interval_isolation_requires_coverage_and_keeps_foreign_jobs_separate():
    record={'own_pid':11,'interval_seconds':1.,'errors':[],
            'observations':[{'started_seconds':float(i),'seconds':i+.1,'pids':[11] if i<5 else [11,22],
                             'own_memory_bytes':100+i} for i in range(8)]}
    clean=assess_gpu_interval(record,1.2,3.8)
    assert clean['sampled_isolation_pass'] and clean['sampled_process_memory_peak_bytes']==104
    assert not assess_gpu_interval(record,1.2,4.8)['sampled_isolation_pass'] # following anchor has foreign PID
    assert not assess_gpu_interval(record,0.,1.)['sampled_isolation_pass']
    assert not assess_gpu_interval(record,6.,9.)['sampled_isolation_pass']
    record['errors']=[{'seconds':2.5,'error':'query failed'}]
    assert not assess_gpu_interval(record,1.2,3.8)['sampled_isolation_pass']
    record['errors']=[]
    record['observations']=record['observations'][:1]+record['observations'][4:]
    assert not assess_gpu_interval(record,.2,3.8)['sampled_isolation_pass']

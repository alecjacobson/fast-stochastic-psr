"""Checked bounded-block archives for coherent Fourier query replay.

An archive is readable only after all declared draws are written and the
producer seals it. Hashes detect damaged/mixed files. This is not a sampler
checkpoint/resume protocol or a guarantee against storage-device failure.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def digest(path):
    checksum=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): checksum.update(block)
    return checksum.hexdigest()


def _json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


class CoefficientArchiveWriter:
    def __init__(self,path,count,metadata,mean_coefficients,geometry):
        self.path=Path(path)
        if count!=int(count) or count<1: raise ValueError('count must be positive integer')
        self.path.mkdir(parents=True,exist_ok=True)
        if any(self.path.iterdir()): raise ValueError('archive directory must be empty')
        self.record={'schema_version':1,'complete':False,'count':int(count),'written':0,'blocks':[],
                     'metadata':json.loads(json.dumps(metadata,allow_nan=False)),'files':{}}
        self._array_file('mean.npy',np.asarray(mean_coefficients))
        temporary=self.path/'geometry.tmp'
        with temporary.open('wb') as f: np.savez(f,**geometry)
        os.replace(temporary,self.path/'geometry.npz')
        self.record['files']['geometry.npz']=digest(self.path/'geometry.npz')
        _json(self.path/'manifest.json',self.record)

    def _array_file(self,name,array):
        temporary=self.path/(name+'.tmp')
        with temporary.open('wb') as f: np.save(f,array,allow_pickle=False)
        os.replace(temporary,self.path/name)
        self.record['files'][name]=digest(self.path/name)

    def append(self,start,coefficients):
        coefficients=np.asarray(coefficients)
        if self.record['complete']: raise ValueError('archive is sealed')
        if start!=self.record['written']: raise ValueError('blocks must be contiguous and ordered')
        if (coefficients.ndim!=4 or not len(coefficients) or len(set(coefficients.shape[1:]))!=1
                or coefficients.shape[-1]%2!=1 or coefficients.dtype not in [np.dtype('complex64'),np.dtype('complex128')]
                or not np.isfinite(coefficients).all()):
            raise ValueError('expected finite complex odd-cube coefficient batches')
        stop=start+len(coefficients)
        if stop>self.record['count']: raise ValueError('too many draws')
        shape=list(coefficients.shape[1:]);dtype=coefficients.dtype.str
        if 'shape' in self.record and (shape!=self.record['shape'] or dtype!=self.record['dtype']):
            raise ValueError('coefficient shape/dtype changed')
        self.record.update(shape=shape,dtype=dtype)
        name=f'{start:08d}-{stop:08d}.npy'
        self._array_file(name,coefficients)
        self.record['blocks'].append({'start':start,'stop':stop,'file':name})
        self.record['written']=stop
        _json(self.path/'manifest.json',self.record)

    def finish(self,validation):
        if self.record['complete']: raise ValueError('archive is sealed')
        if self.record['written']!=self.record['count']: raise ValueError('archive is incomplete')
        self.record['validation']=json.loads(json.dumps(validation,allow_nan=False))
        self.record['complete']=True
        _json(self.path/'manifest.json',self.record)


class CoefficientArchive:
    def __init__(self,path):
        self.path=Path(path)
        self.record=json.loads((self.path/'manifest.json').read_text())
        r=self.record
        if r.get('schema_version')!=1 or not r.get('complete') or r.get('written')!=r.get('count') or r.get('count',0)<1:
            raise ValueError('archive is incomplete or unsupported')
        position=0
        for b in r['blocks']:
            if b['start']!=position or b['stop']<=position or b['file']!=f'{position:08d}-{b["stop"]:08d}.npy':
                raise ValueError('invalid block coverage or filename')
            position=b['stop']
        if position!=r['count']: raise ValueError('invalid block coverage')

    def checked_path(self,name):
        if Path(name).name!=name or name not in self.record['files']:
            raise ValueError('unregistered archive file')
        path=self.path/name
        if digest(path)!=self.record['files'][name]: raise ValueError('archive file hash mismatch: '+name)
        return path

    def mean(self):
        return np.load(self.checked_path('mean.npy'),allow_pickle=False)

    def geometry(self):
        with np.load(self.checked_path('geometry.npz'),allow_pickle=False) as data:
            return {k:data[k] for k in data.files}

    def blocks(self):
        for b in self.record['blocks']:
            array=np.load(self.checked_path(b['file']),allow_pickle=False)
            if (list(array.shape)!=[b['stop']-b['start'],*self.record['shape']]
                    or array.dtype.str!=self.record['dtype']):
                raise ValueError('coefficient shape/dtype disagrees with manifest')
            yield {**b,'coefficients':array}
            del array

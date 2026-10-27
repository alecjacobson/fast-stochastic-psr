"""Measure compilation/loading in a fresh native-library cache directory."""
import json
import os
from pathlib import Path
import tempfile
from geospsr.native_range import library


def main():
    root=Path(__file__).resolve().parents[1]
    runtime=os.environ.get('GEOSPSR_RUNTIME',tempfile.gettempdir())
    previous=os.environ.get('GEOSPSR_RUNTIME')
    with tempfile.TemporaryDirectory(prefix='range-cold-library-',dir=runtime) as directory:
        try:
            os.environ['GEOSPSR_RUNTIME']=directory
            _,record=library()
        finally:
            if previous is None:os.environ.pop('GEOSPSR_RUNTIME',None)
            else:os.environ['GEOSPSR_RUNTIME']=previous
    path=root/'reproduction/results/range_compact/native/cold_library.json'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(record,indent=2)+'\n')
    print(record)


if __name__=='__main__':main()

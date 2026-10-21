import json,platform,subprocess
from pathlib import Path
import jax,numpy,flax,cupy,cufinufft
out=Path(__file__).resolve().parent/'results'
r={'upstream_commit':'2352f1ef98aab20e0303611c1fa2b0948acd2a05',
   'python':platform.python_version(),'platform':platform.platform(),
   'jax':jax.__version__,'flax':flax.__version__,'numpy':numpy.__version__,
   'cupy':cupy.__version__,'cufinufft':cufinufft.__version__,
   'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
   'cpu':subprocess.check_output(['lscpu'],text=True),'jax_default_matmul_precision':str(jax.config.jax_default_matmul_precision),
   'notes':'CUDA synchronized wall-clock measurements; GPU clocks not locked; no concurrent GPU jobs launched by this harness during timed runs'}
(out/'environment.json').write_text(json.dumps(r,indent=2))

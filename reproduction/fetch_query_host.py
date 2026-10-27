"""Fetch manifest-listed public controls (query-host by default) and check hashes."""
import argparse,hashlib,json,shutil,tempfile,urllib.request
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    root=Path(__file__).resolve().parents[1]/'reproduction/results/query_host'
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path)
    p.add_argument('--manifest',type=Path,default=root/'release_manifest.json')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());dest=(a.destination or a.manifest.parent).resolve()
    for asset in manifest['assets']:
        path=(dest/asset['relative_path']).resolve();path.relative_to(dest)
        if path.exists():
            if path.stat().st_size!=asset['size_bytes'] or digest(path)!=asset['sha256']:
                raise ValueError(f'existing file does not match recorded control: {path}')
        elif a.verify_only:raise FileNotFoundError(path)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
                partial=Path(tmp)/'download'
                with urllib.request.urlopen(asset['url'],timeout=120) as source,partial.open('wb') as target:
                    shutil.copyfileobj(source,target,4*1024**2)
                if partial.stat().st_size!=asset['size_bytes'] or digest(partial)!=asset['sha256']:
                    raise ValueError(f'download failed integrity check: {asset["filename"]}')
                partial.replace(path)
        print('verified',asset['relative_path'],flush=True)


if __name__=='__main__':main()

"""Download checksum-pinned upstream example data to an external cache."""
import argparse
from pathlib import Path
import shutil
import urllib.request
import zipfile

from reproduction.poissonrecon_data import SOURCES, prepare, sha256

ARCHIVE_SHA256 = 'a9ee6b6af62921e2223b5cd47f0e80d3345ec93379a7b9b67131d3c1fcf6c25c'


def fetch(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in SOURCES.items():
        download = root/url.rsplit('/', 1)[-1]
        expected = ARCHIVE_SHA256 if name == 'torso' else digest
        if not download.exists():
            temporary = download.with_suffix(download.suffix+'.part')
            with urllib.request.urlopen(url, timeout=120) as source, temporary.open('wb') as out:
                shutil.copyfileobj(source, out)
            if sha256(temporary) != expected:
                raise ValueError('download checksum mismatch')
            temporary.replace(download)
        if sha256(download) != expected:
            raise ValueError('cached download checksum mismatch')
        if name == 'torso':
            with zipfile.ZipFile(download) as archive:
                for filename in ['torso.points.ply', 'torso.envelope.ply']:
                    path = root/filename
                    if not path.exists():
                        with archive.open(filename) as source, path.open('wb') as out:
                            shutil.copyfileobj(source, out)
        source = root/(name+'.points.ply')
        if sha256(source) != digest:
            raise ValueError('source PLY checksum mismatch')
        output = root/(name+'.npy')
        if not output.exists():
            prepare(name, source, output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', default=str(Path.home()/'.cache/fast-stochastic-psr/poissonrecon'))
    fetch(p.parse_args().cache_dir)

#!/usr/bin/env python3
"""Recover the original bundled CPU dependency without following external links."""
import hashlib
import sys
import tarfile
from pathlib import Path, PurePosixPath

if len(sys.argv) != 3:
    raise SystemExit('Usage: prepare_nicslu.py archive .deps')
archive, destination = map(Path, sys.argv[1:])
if destination.resolve() != Path('.deps').resolve():
    raise SystemExit('Dependency extraction is restricted to this port\'s .deps directory')
target = destination / 'nicslu'
if target.exists():
    if (target / 'include/nicslu.h').is_file() and (target / 'lib/Makefile').is_file():
        print('Retaining existing dependency extraction:', target)
        raise SystemExit(0)
    raise SystemExit('Existing incomplete dependency directory requires inspection; refusing overwrite')
with tarfile.open(archive, 'r:bz2') as tar:
    accepted = []
    for member in tar.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'nicslu':
            raise SystemExit('Unsafe archive path: ' + member.name)
        if member.name.rstrip('/') == 'nicslu/demo/matrix' and member.issym():
            print('Excluded external symlink:', member.name, '->', member.linkname)
            continue
        if not (member.isfile() or member.isdir()):
            raise SystemExit('Unsupported archive link/device: ' + member.name)
        accepted.append(member)
    destination.mkdir(exist_ok=True)
    tar.extractall(destination, members=accepted, filter='data')
print('archive_sha256', hashlib.sha256(archive.read_bytes()).hexdigest())
print('extracted_entries', len(accepted))

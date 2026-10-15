#!/usr/bin/env python3
"""Fetch the four exact upstream pins from the original CMake project."""
import hashlib
import pathlib
import tarfile
import urllib.request

root = pathlib.Path(__file__).resolve().parent / '.deps'
pins = [
 ('cxxopts','jarro2783/cxxopts','refs/tags/v3.3.1'),
 ('fmt','fmtlib/fmt','refs/tags/7.1.2'),
 ('zipf','llersch/cpp_random_distributions','6ae38ecf5dbeb1150b80ecf051cb43cb88ba697a'),
 ('progress','prakhar1989/progress-cpp','7bfba0d22d19c41323aa35541618b6ebec9d737c')]
root.mkdir(exist_ok=True)
for name, repo, pin in pins:
    target = root / name
    if target.is_dir():
        continue
    archive = root / (name + '.tgz')
    if not archive.exists():
        with urllib.request.urlopen(f'https://codeload.github.com/{repo}/tar.gz/{pin}', timeout=60) as response:
            archive.write_bytes(response.read())
    print(name, hashlib.sha256(archive.read_bytes()).hexdigest())
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for item in members:
            p = pathlib.PurePosixPath(item.name)
            if p.is_absolute() or '..' in p.parts or not (item.isfile() or item.isdir()):
                raise RuntimeError('Unsafe archive member: ' + item.name)
        prefix = pathlib.PurePosixPath(members[0].name).parts[0]
        tf.extractall(root, members=members)
    (root / prefix).rename(target)

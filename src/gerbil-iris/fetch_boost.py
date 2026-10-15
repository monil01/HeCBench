"""Optional local-only Ubuntu 24.04 Boost runtime recovery; no system install."""
from pathlib import Path
import hashlib,subprocess,tarfile,io

checks={
 'system':'69f25ec5a0f321c91e2f1699ef29c550bf7b31f61d7648a8c4986ccf581f8e02',
 'thread':'d6b7411fd5368a49ffe88ad66d471bc57a43c2bb4eb105ee3f5fb921bfdf8924',
 'regex':'ad62758e356034a6ef5cc3257bb527439687b6c1a68c65ad89e9dae994efffa5',
 'filesystem':'154b74fe60f7fc6223b0ff86063b8eb9429bfd1f532085fce66b555ebf2bf96c',
 'atomic':'e31f9d318ceffbc44d7317a5d150b8cfbe65ebd0e9e4d9ea7c217eaaae9eaaf9'}
root=Path(__file__).resolve().parent/'.deps';packages=root/'packages';packages.mkdir(parents=True,exist_ok=True)
for name,expected in checks.items():
 package=f'libboost-{name}1.83.0';version='1.83.0-2.1ubuntu3.2'
 archive=packages/f'{package}_{version}_amd64.deb'
 if not archive.exists():subprocess.run(['apt-get','download',f'{package}={version}'],cwd=packages,check=True)
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected,f'Unexpected package checksum: {archive}'
 data=subprocess.check_output(['dpkg-deb','--fsys-tarfile',str(archive)])
 with tarfile.open(fileobj=io.BytesIO(data)) as stream:
  for entry in stream:
   assert not entry.name.startswith('/') and '..' not in Path(entry.name).parts
 subprocess.run(['dpkg-deb','-x',str(archive),str(root/'boost')],check=True)
print('Verified local Boost runtime extraction:',root/'boost')

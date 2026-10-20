#!/usr/bin/env python3
"""Exact source archive assets plus deterministic positive-pivot branch controls."""
from pathlib import Path
import argparse,tarfile,hashlib,json
P=Path(__file__).resolve().parent;R=P.parents[1]
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);z=a.parse_args();out=z.output.resolve();assert not out.exists(),'Refuse any existing asset directory';out.mkdir()
archive=R/'src/slu-cuda/src/nicslu.tar.bz2';record=[]
with tarfile.open(archive)as t:
 for member,name in [('nicslu/demo/ASIC_100k.mtx','ASIC_100k.mtx'),('nicslu/test/add32.mtx','add32.mtx')]:
  data=t.extractfile(member).read();p=out/name;p.write_bytes(data);record.append(dict(path=str(p),source_member=member,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),role='Originalbundledscientificasset, notsubstitutematrix'))
for n in [16,2048]:
 p=out/('positive_pair_'+str(n)+'.mtx')
 with p.open('w')as f:
  f.write('%%MatrixMarket matrix coordinate real general\n% NamedblockdiagonalSPDbranchcontrol, notoriginalcircuitasset\n'+str(n)+' '+str(n)+' '+str(2*n)+'\n')
  for k in range(1,n+1,2):
   for i,j,v in [(k,k,4),(k+1,k,1),(k,k+1,1),(k+1,k+1,3)]:f.write(f'{i} {j} {v}\n')
 record.append(dict(path=str(p),source_member=None,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role='Deterministicsourcepipelinepflagcontrol; positivediagonalSchurpivots, noactiveclampcertificationunlessindependentlyproved.'))
(out/'cases.txt').write_text('ASIC_100k\t0\t'+str(out/'ASIC_100k.mtx')+'\nadd32\t0\t'+str(out/'add32.mtx')+'\npositive_pair_16\t1\t'+str(out/'positive_pair_16.mtx')+'\npositive_pair_2048\t1\t'+str(out/'positive_pair_2048.mtx')+'\n')
(out/'asset_provenance.json').write_text(json.dumps(dict(archive_path=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),cases=record,no_new_downloads=True),indent=2)+'\n');print(out)

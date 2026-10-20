#!/usr/bin/env python3
"""Fresh CPU fixture regeneration and exact hash verification; no GPU."""
import pathlib,json,argparse,subprocess,sys,hashlib,os
P=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--output',type=pathlib.Path,required=True);a.add_argument('--proof',type=pathlib.Path,required=True);z=a.parse_args();cmd=[sys.executable,str(P/'generate_assets.py'),'--output',str(z.output)];subprocess.run(cmd,check=True);want=json.load(open(P/'asset_hashes.json'));rows=[]
for name,meta in want['files'].items():
 p=z.output/name;assert p.stat().st_size==meta['bytes']and sha(p)==meta['sha256'],name;rows.append(dict(path=str(p),**meta))
d=dict(status='all_CPU_binary_assets_byte_exact',files=rows,command=cmd,numpy=json.load(open(z.output/'manifest.json'))['numpy'],GPU_execution=False,scope='CurrentpinnedCPUenvironmentbinaryfixtures/oracle exact; manifestruntimepaths not measured GPUreproduction.');z.proof.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))

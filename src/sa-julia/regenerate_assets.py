#!/usr/bin/env python3
"""Cold reconstruction of all essential CPU inputs; no GPU outputs are needed."""
from pathlib import Path
import argparse,subprocess,sys,hashlib,json,array
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--native-assets',type=Path);a=p.parse_args();P=Path(__file__).resolve().parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);cache=a.native_assets.resolve()if a.native_assets else out/'original_asset';assert array.array('i').itemsize==4
plan=json.loads((P/'study_plan.json').read_text());names=[c['input_file']for c in plan['cases']]+['primary_cases.txt','controls_cases.txt','study_plan.json'];expected_hashes={c['input_file']:c['sha256']for c in plan['cases']}
for name in ['primary_cases.txt','controls_cases.txt','study_plan.json']:expected_hashes[name]=hashlib.sha256((P/name).read_bytes()).hexdigest()
subprocess.run([sys.executable,str(P/'fetch_original_asset.py'),str(cache)],check=True)
subprocess.run([sys.executable,str(P/'prepare_assets.py'),str(out),'--genome',str(cache/'genome.txt')],check=True)
records=[]
for name in names:
 f=out/name;expected=P/name;actual=hashlib.sha256(f.read_bytes()).hexdigest();target=expected_hashes[name]
 # On an unpacked source packet plan/case text remain committed; binaries need recipes only.
 assert target is not None and actual==target,(name,actual,target)
 records.append({'file':name,'bytes':f.stat().st_size,'sha256':actual,'matches_frozen_CPU_asset':True})
(out/'regeneration_provenance.json').write_text(json.dumps({'status':'pass','Python':sys.version,'CPU_only':True,'array_int_itemsize':4,'source_genome_archive':'exactMD5+sizevalidatedbyfetcher; retrievedorcacheusedrecordedthere','all_assets':records},indent=2)+'\n');print('ALL_CPU_ASSETS_BYTE_EXACT',len(records))

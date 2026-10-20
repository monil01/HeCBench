#!/usr/bin/env python3
"""Regenerate CPU-owned PRNA assets in a new directory, preserving measured files."""
from pathlib import Path
import argparse,subprocess,sys,json,hashlib
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--native-assets',type=Path,help='Existing exact fetcheddata_tables/test.seq cache; otherwise fetchdeclaredDVCobjects');a=p.parse_args();B=Path(__file__).resolve().parent;R=B.parents[1];out=a.output.resolve();assert out!=B and not out.is_relative_to(B), 'Use a new directory outside measured PRNA sources';out.mkdir(parents=True,exist_ok=True)
cache=a.native_assets.resolve() if a.native_assets else out/'native_asset_cache';commands=[]
if a.native_assets is None:commands.append([sys.executable,str(B/'fetch_original_assets.py'),str(cache)])
commands.extend([['g++','-O3','-std=c++17','-DFLOAT',str(B/'generate_fixture.cpp')]+[str(R/'src/prna-cuda'/n) for n in ['base.c','param.c','util.c']]+['-o',str(out/'generate_fixture')],[str(out/'generate_fixture'),str(cache/'data_tables'),str(cache/'test.seq')],[sys.executable,str(B/'independent_small_partition.py'),str(out)]])
logs=[]
for cmd in commands:
 r=subprocess.run(cmd,cwd=out,text=True,capture_output=True,check=True);logs.append({'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
expected=json.loads((B/'cold_regeneration_audit.json').read_text())['assets'];actual={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in expected};assert expected==actual, 'Coldassetbytesdiffer; preserve this environment/version difference and do notoverwrite frozen references'
(out/'regeneration_provenance.json').write_text(json.dumps({'status':'all9exact','expected':expected,'actual':actual,'steps':logs,'source_fixture_files_unchanged':True},indent=2)+'\n');print(out)

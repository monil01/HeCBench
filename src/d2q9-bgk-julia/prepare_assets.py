#!/usr/bin/env python3
"""Recreate all deterministic fixture, parameter and exact-comparison mask files."""
from pathlib import Path
import json,numpy as np
import generate_fixtures
P=Path(__file__).resolve().parent
def main():
 generate_fixtures.main();d=json.loads((P/'study_plan.json').read_text());cases=d['cases'];h='#pragma once\n'
 for typ,name,key in [('int','NXS','nx'),('int','NYS','ny'),('int','TS','steps'),('float','RHOS','density'),('float','ACCS','accel'),('float','OMS','omega')]:
  vals=[str(x[key]) if typ=='int' else repr(x[key])+'f' for x in cases];h+=f'const {typ} {name}[3]={{'+','.join(vals)+'};\n'
 h+=f'const int COUNT={d["count"]},PREFIX={d["prefix_count"]};\n';(P/'study_params.h').write_text(h)
 j='const NXS=Int32'+str([c['nx'] for c in cases])+'\nconst NYS=Int32'+str([c['ny'] for c in cases])+'\nconst TS='+str([c['steps'] for c in cases])+'\n'
 for name,key in [('RHOS','density'),('ACCS','accel'),('OMS','omega')]:j+='const '+name+'=Float32'+str([c[key] for c in cases])+'\n'
 j+=f'const COUNT={d["count"]};const PREFIX={d["prefix_count"]}\n';(P/'study_params.jl').write_text(j)
 mask=np.zeros(d['count'],dtype='u1');mask[:d['prefix_count']]=1;off=d['prefix_count']
 for x in cases:
  n=x['nx']*x['ny'];gt=n//128*x['steps']
  for _ in range(18):mask[off:off+16]=1;mask[off+16+n:off+32+n]=1;off+=n+32
  mask[off:off+16]=1;mask[off+gt+16:off+gt+32]=1;off+=gt+32;mask[off:off+gt+32]=1;off+=gt+32
 assert off==d['count'];mask.tofile(P/'exact_mask_u8.bin')
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Reproduce only the reviewed logical Cartesian index adaptation of frozen math."""
from pathlib import Path
import argparse,re,hashlib
P=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);z=a.parse_args();s=(P/'device_bodies.jl').read_text()
for name in ['cuda_mat_J_PI','cuda_mat_K_PI','cuda_mat_J_PI_dp','cuda_mat_K_PI_dp']:
 s,n=re.subn(r'(function '+name+r'\([^\n]*)(\))',r'\1,xl_lx,xl_ly,xl_gx,xl_gy\2',s);assert n==1
for native,target in [('threadIdx().x','xl_lx'),('threadIdx().y','xl_ly'),('blockIdx().x','xl_gx'),('blockIdx().y','xl_gy')]:s=s.replace('(Int32(CUDA.'+native+')-Int32(1))',target)
z.output.write_text('# KA source index mapping: explicit logical Cartesian IDs; math helpers unchanged.\n'+s);print(hashlib.sha256(z.output.read_bytes()).hexdigest())

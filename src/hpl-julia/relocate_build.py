#!/usr/bin/env python3
"""Explicit loader-path relocation for new reproductions; never edits measured files."""
import argparse,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('--build-script',type=pathlib.Path,required=True);p.add_argument('--openblas-library',type=pathlib.Path,required=True);p.add_argument('--output',required=True);a=p.parse_args()
s=a.build_script.read_text();old="/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/lib/julia/libopenblas64_.so";assert s.count(old)==1 and a.openblas_library.is_file();s=s.replace(repr(old),repr(str(a.openblas_library.resolve())));sys.argv=[str(a.build_script),a.output];exec(compile(s,str(a.build_script),'exec'),{'__file__':str(a.build_script.resolve()),'__name__':'__main__'})

#!/usr/bin/env python3
"""Existing unprivileged GCC/nvcc only. Source-bound PIC NICSLU cache, no OS changes."""
import argparse,pathlib,subprocess,hashlib,json,re,fcntl
P=pathlib.Path(__file__).resolve().parent;V=pathlib.Path('/tmp/hecbench-stencil-ports-slu-original/src/nicslu')
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['native','shared'],required=True);a.add_argument('--output',required=True);a.add_argument('--arch',default='sm_70');z=a.parse_args()
assert V.is_dir(),'Extract source NICSLU dependency first'
objs=re.findall(r'([A-Za-z0-9_]+)\.o', (V/'lib/Makefile').read_text());assert len(objs)==len(set(objs))
inputs=[V/'source'/(x+'.c')for x in objs]+[V/'util/nicslu_readfile.c']+list((V/'source').glob('*.h'))+list((V/'include').glob('*.h'))+[V/'util/nicslu_util.h']
h=hashlib.sha256(b'GCC-fPIC-O2-msse2-NO_ATOMIC-SSE2-20261020');
for p in sorted(inputs):h.update(str(p.relative_to(V)).encode());h.update(p.read_bytes())
cache=pathlib.Path('/tmp/hecbench-stencil-ports-slu-pic-cache')/h.hexdigest()[:16];cache.mkdir(parents=True,exist_ok=True)
with (cache/'build.lock').open('w')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);manifest=cache/'manifest.json'
 if not manifest.exists():
  paths=[]
  for name in objs:
   source=V/'source'/(name+'.c');output=cache/(name+'.o');subprocess.run(['gcc','-fPIC','-g','-O2','-msse2','-DNO_ATOMIC','-DSSE2','-c',str(source),'-o',str(output)],check=True);paths.append(output)
  subprocess.run(['ar','rcs',str(cache/'nicslu.a'),*[str(p)for p in paths]],check=True)
  subprocess.run(['gcc','-fPIC','-g','-O2','-msse2','-DNO_ATOMIC','-DSSE2','-I'+str(V/'include'),'-c',str(V/'util/nicslu_readfile.c'),'-o',str(cache/'nicslu_readfile.o')],check=True)
  subprocess.run(['ar','rcs',str(cache/'nicslu_util.a'),str(cache/'nicslu_readfile.o')],check=True)
  manifest.write_text(json.dumps({'source_binding':h.hexdigest(),'source_inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in inputs},'archives':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [cache/'nicslu.a',cache/'nicslu_util.a']},'scope':'SameoriginalNICSLUCsources, existingGCCO2/SSE2/NO_ATOMIC; PICpackagingpermitsCABIsharedwrapper andsamePICarchivesusednative/targets.'},indent=2)+'\n')
 m=json.load(open(manifest));assert m['source_binding']==h.hexdigest()
 for p,sha in m['archives'].items():assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==sha
 print('STUDY_CPU_LIBRARY_BINDING',manifest,hashlib.sha256(manifest.read_bytes()).hexdigest(),flush=True)
command=['/usr/bin/nvcc','-std=c++17','-O3','-arch='+z.arch,'-DVERIFY','-I'+str(P/'reference_history/include'),'-I'+str(V/'include'),'-I'+str(V/'util'),str(P/'full_study.cu'),str(P/'checked_preprocess.cpp'),str(P/'reference_history/src/symbolic.cpp'),str(P/'reference_history/src/Timer.cpp'),str(cache/'nicslu.a'),str(cache/'nicslu_util.a'),'-Xcompiler=-fopenmp','-Xcompiler=-pthread','-lz','-lcrypto','-o',z.output]
if z.mode=='shared':command+=['-DSTUDY_SHARED','--shared','-Xcompiler=-fPIC']
print('STUDY_BUILD_COMMAND',json.dumps(command),flush=True);subprocess.run(command,check=True)

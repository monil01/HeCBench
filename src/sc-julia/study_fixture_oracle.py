from pathlib import Path
import struct,json,hashlib,shutil
R=Path('/home/imo/HeCBench_julia');p=R/'src/sc-julia';h=p/'reference_history';h.mkdir(exist_ok=True)
for name in ['main.jl','Makefile']:
 if not(h/('original_'+name)).exists():shutil.copy2(p/name,h/('original_'+name))
cases=[]
for n in (8192,32768,98304):
 for ai in (0,1,5,10):cases.append((n,ai))
cases +=[(1,0),(33,0),(8193,0)]
f=[len(cases)];oracle=[];desc=[]
for ci,(n,ai)in enumerate(cases):
 a=[i+2 if (i*17+ci*3)%10<([0,5,10,9,1][ci%5])else 0 for i in range(n)];tasks=(n+8191)//8192;cut=int(tasks*struct.unpack('<f',struct.pack('<f',ai/10))[0]);a[-1]=n+1 if n%8192 else a[-1];cpu=a[:cut*8192];gpu=a[cut*8192:];out=[v for v in a if v!=0]
 cf=[0]*(tasks+1);gf=[0]*(tasks+1);cf[0]=gf[0]=1;cf[cut]=gf[cut]=1
 for task in range(cut):cf[task+1]+=cf[task]+sum(v!=0 for v in a[task*8192:(task+1)*8192])
 for task in range(cut,tasks):gf[task+1]+=gf[task]+sum(v!=0 for v in a[task*8192:(task+1)*8192])
 f +=[n,ai]+a;oracle +=[len(out)]+out+cf+gf+[-777]*(8)
 desc.append({'n':n,'alpha_tenths':ai,'tasks':tasks,'cut':cut,'retained':len(out),'CPU_retained':sum(v!=0 for v in cpu),'GPU_retained':sum(v!=0 for v in gpu),'pattern':'Exactpositiveindex+2 atdeterministicsourcecompatiblekeepmask,0removevalue;allzero/allkeep/mixed','full_tile':n%8192==0})
for name,a in [('fixtures_i32.bin',f),('oracle_i32.bin',oracle)]: (p/name).write_bytes(struct.pack('<'+'i'*len(a),*a))
(p/'independent_oracle.json').write_text(json.dumps({'cases':desc,'input_prefix_count':len(f),'output_count':len(f)+len(oracle),'method':'IndependentPythondirectstablefilter; exactsourcefixed32x256tilecut andCPU/GPUflagseed/prefixcounts, allvaluesexact. Physicalpaddinginitialized0, unusedtailsnotoperatoroutputs.','fixture_sha256':hashlib.sha256((p/'fixtures_i32.bin').read_bytes()).hexdigest(),'oracle_sha256':hashlib.sha256((p/'oracle_i32.bin').read_bytes()).hexdigest()},indent=2)+'\n');print({'cases':len(cases),'prefix':len(f),'output_count':len(f)+len(oracle)})

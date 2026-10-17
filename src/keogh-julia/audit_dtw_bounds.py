#!/usr/bin/env python3
"""Check saved GPU scores against independent radius-constrained DTW witnesses."""
from pathlib import Path
import struct,json,argparse,hashlib,math
P=Path(__file__).resolve().parent;ap=argparse.ArgumentParser();ap.add_argument('result');ap.add_argument('--output');a=ap.parse_args();N=262144;M=256;n=N-M+1;r=8;data=struct.unpack('<'+'f'*(N+2*n+3*M),(P/'study_input_f32.bin').read_bytes());subject=data[:N];avg=data[N:N+n];std=data[N+n:N+2*n];lower=data[N+2*n:N+2*n+M];upper=data[N+2*n+M:N+2*n+2*M];query=data[-M:];starts=[0,1,17,127,1020,8191,131072,n-1];witness=[]
for start in starts:
 x=[(subject[start+i]-avg[start])/std[start]for i in range(M)];previous=[math.inf]*(M+1);previous[0]=0.
 for i in range(M):
  current=[math.inf]*(M+1)
  for j in range(max(0,i-r),min(M,i+r+1)):current[j+1]=(x[i]-query[j])**2+min(previous[j+1],previous[j],current[j])
  previous=current
 bound=sum(max(lower[i]-x[i],0.,x[i]-upper[i])**2 for i in range(M));assert bound<=previous[-1]+1e-3;witness.append({'window_start':start,'CPU_bound_squared':bound,'CPU_DTW_squared':previous[-1]})
result=json.load(open(a.result));out={'benchmark':'keogh','configuration_id':result['configuration_id'],'source_result':a.result,'method':'IndependentPythonDouble squaredDTW dynamicprogram radius8 anddistance-to-envelope','source_input_sha256':hashlib.sha256((P/'study_input_f32.bin').read_bytes()).hexdigest(),'selected_windows':witness,'models':{}}
for model,m in result['models'].items():
 if not m.get('complete'):continue
 rows=[]
 for t in m['trials']:
  art=t['output_artifact'];d=Path(art['saved_path']).read_bytes();assert hashlib.sha256(d).hexdigest()==art['sha256'];scores=[struct.unpack_from('<f',d,k*4)[0]for k in starts];assert all(math.isfinite(v)and v>=0 and v<=w['CPU_DTW_squared']+1e-2 for v,w in zip(scores,witness));rows.append({'trial':t['trial'],'selected_scores':scores,'all_selected_bounds_hold':True,'artifact_sha256':art['sha256']})
 assert len(rows)==5;out['models'][model]=rows
s=json.dumps(out,indent=2)+'\n'
if a.output:Path(a.output).write_text(s)
else:print(s)

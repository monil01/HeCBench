from pathlib import Path
import json,struct,hashlib,re,math
R=Path('/home/imo/HeCBench_julia');p=R/'src/rsbench-julia';h=p/'reference_history';h.mkdir(exist_ok=True)
for name in ['main.jl','Makefile']:
 if not(h/('original_'+name)).exists():(h/('original_'+name)).write_bytes((p/name).read_bytes())
M=(1<<63)-1;A=2806196910506780709;seed=42
nucs=[34,5,4,4,27,21,21,21,21,21,9,9];maxn=34
src=(R/'src/rsbench-cuda/material.cu').read_text();names=['0_Sml']+list(map(str,range(1,12)));mats=[0]*(12*maxn)
for i,name in enumerate(names):
 values=list(map(int,re.search(r'int mats'+name+r'\[\]\s*=\s*\{([^}]+)\}',src).group(1).replace('\n','').split(',')))
 assert len(values)==nucs[i];mats[i*maxn:i*maxn+len(values)]=values
concs=[0.]*(12*maxn)
def rand():
 global seed
 seed=(A*seed+1)&M
 return seed/(1<<63)
for i,n in enumerate(nucs):
 for j in range(n):concs[i*maxn+j]=rand()
np=[1]*68;nw=[1]*68
for arr,total in [(np,68*12),(nw,68*3)]:
 for _ in range(total-68):rand();arr[seed%68]+=1
maxp=max(np);maxw=max(nw);poles=[[0.]*9 for _ in range(68*maxp)]
for i in range(68):
 for j in range(np[i]):
  row=[152.5*rand(),152.5*rand(),152.5*rand(),rand(),152.5*rand(),rand(),152.5*rand(),rand()];rand();row.append(seed%4);poles[i*maxp+j]=row
windows=[[0.]*5 for _ in range(68*maxw)]
coverage=[]
for i in range(68):
 space,rem=divmod(np[i],nw[i]);ctr=0;vis=[]
 for j in range(nw[i]):
  start=ctr;end=ctr+space-1;ctr+=space
  if j<rem:ctr+=1;end+=1
  windows[i*maxw+j]=[rand(),rand(),rand(),start,end];vis.extend(range(start,end))
 missing=sorted(set(range(np[i]))-set(vis));coverage.append({'nuclide':i,'assigned_poles':np[i],'windows':nw[i],'visited_exclusive_end':vis,'unvisited_assigned_indices':missing})
pseudo=[rand() for _ in range(68*4)];cases=[(n,d)for n in (1,33,257,513)for d in(0,1)]
f=[68,maxp,maxw,maxn,len(cases)]+[v for c in cases for v in c]+nucs+mats+nw+concs+pseudo+[v for row in poles for v in row]+[v for row in windows for v in row]
(p/'fixtures_f64.bin').write_bytes(struct.pack('<'+'d'*len(f),*f))
info={'cases':[{'lookups':n,'doppler':d}for n,d in cases],'nuclides':68,'maxp':maxp,'maxw':maxw,'maxn':maxn,'source_numL':4,'source_STARTING_SEED':1070,'source_INITIALIZATION_SEED':42,'input_prefix_count':len(f),'fixture_sha256':hashlib.sha256((p/'fixtures_f64.bin').read_bytes()).hexdigest(),'generation':'IndependentlyreproducedsourceHMsmallmaterialmap/concentrationLCG/order/n_polesavg12/n_windowsavg3/poles/windowparams/pseudoK0RS; unusedfixedstrides/structpaddinginitialized0outsideROI.','window_endpoint_coverage':coverage,'coverage_interpretation':'Initializerend=ctr+space-1vslookupstrict<end leaveslistedassignedpoleindicesunvisited. Sourcebehaviorpreserved; independentcoveragefactdoesnotaloneestablishareactorphysicscontractdefect.'}
(p/'independent_fixture.json').write_text(json.dumps(info,indent=2)+'\n');print({k:v for k,v in info.items()if k!='window_endpoint_coverage'})

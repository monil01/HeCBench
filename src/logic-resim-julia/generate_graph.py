#!/usr/bin/env python3
"""Source textual graph/truth metadata -> typed semantic fixtures (no GPU output)."""
from pathlib import Path
import json,struct,hashlib,collections
P=Path(__file__).resolve().parent;R=P.parent.parent;SRC=R/'src/logic-resim-cuda/Simulation/toy/intermediate.file';records=[];names=[];inputs=set();gatefuncs=collections.Counter();arities=collections.Counter();sourceproducer={};dependencies=[]
with SRC.open()as f:
 nw,ni,no,na,nfunc,nb,start=map(int,next(f).split());truth=[next(f).split()for _ in range(nfunc)];assign=[list(map(int,next(f).split()))for _ in range(na)];declarations=[next(f).split()for _ in range(ni+no)];packed=bytearray(nfunc*1025)
 for k,(degree,s)in enumerate(truth):
  degree=int(degree);assert len(s)==4**degree and set(s)<={'0','1','2'}
  for j,ch in enumerate(s):packed[k*1025+j//4]|=int(ch)<<(2*(j%4))
  packed[k*1025+1024]=degree
 for k,a in enumerate(declarations):
  width=int(a[0]);name=a[1]
  if width==1:names.append(name)
  else:
   msb,lsb=map(int,a[2:]);names.extend(name+'\\['+str(msb-j if msb>lsb else msb+j)+'\\]'for j in range(width))
  if k<ni:inputs.update(range(len(names)-width,len(names)))
 assert len(names)==nw
 for block in range(nb):
  arrays=[]
  for _ in range(3):
   a=list(map(int,next(f).split()));assert a[0]==len(a)-1;arrays.append([v-4 for v in a[1:]])
  gates=[]
  for serial in range(10000):
   a=list(map(int,next(f).split()));func=a[0]
   if func>=nfunc:assert len(a)==1;continue
   arity=int(truth[func][0]);assert len(a)==2+13*arity
   ins=[v-4 for v in a[1:1+arity]];out=a[1+arity]-4;delay=a[2+arity:];assert all(0<=v<nw for v in ins+[out]);assert out not in sourceproducer
   sourceproducer[out]=block;gatefuncs[func]+=1;arities[arity]+=1;gates.append([func,out,arity]+ins+[0]*(6-arity)+delay+[0]*(72-len(delay)))
  output={g[1]for g in gates};reads={v for g in gates for v in g[3:3+g[2]]};bad=sorted(reads&output);assert not bad,(block,bad[:5]);dependencies.append({'block':block,'GPU':block<start,'gates':len(gates),'no_within_batch_read_write_dependency':True,'source_pop':len(arrays[0]),'source_delete':len(arrays[1]),'source_add':len(arrays[2])});records.append((block,arrays,gates))
 assert not f.read().strip()
with(P/'graph_u64.bin').open('wb')as out:
 def write(a):out.write(struct.pack('<'+'Q'*len(a),*a))
 write([0x4c4f474943,nw,nfunc,nb,start,300000000,6,12,32])
 for block,arrays,gates in records:
  write([block,int(block<start),len(gates)]+[len(a)for a in arrays])
  for a in arrays:write(a)
  for g in gates:write(g)
(P/'truth_table_u8.bin').write_bytes(packed);(P/'wire_names.json').write_text(json.dumps(names,separators=(',',':'))+'\n');(P/'wire_aliases.json').write_text(json.dumps({'source_assignments':assign,'source_wireID_offset':4,'interpretation':'r<=3 assignsconstant4state value; r>3 aliasestoTransList[r-4] beforedeclarations/VCDparsing. CPUdecodedinitialhistoriesalreadyreflectsourcealiases.'},indent=2)+'\n')
report={'benchmark':'logic-resim','wires':nw,'truth_functions':nfunc,'graph_batches':nb,'GPU_batches':start,'GPU_gates':sum(len(g)for b,a,g in records if b<start),'CPU_gates':sum(len(g)for b,a,g in records if b>=start),'source_dump_off':2972036001,'named_dump_off':300000000,'arity_histogram':dict(arities),'function_gate_histogram':dict(gatefuncs),'all_batches':dependencies,'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'source_GPU_kernel_unchanged':True,'source_CPU_gatepolicy':'InitialoutputX; evaluateonlypostinitialinputtransitions, sourceinertialcancellation/delaylookup; explicitmethodboundary, nocanonicaltime0settlingclaim.','truth_initialization_policy':'Deterministicallypackedfromsourceplaintext, allunusedpaddingzero; originalactualfullrunGPUtruthbufferbyteidentical. SourceparseruninitializedORhazardseparatelyretained; no claimingoriginalCLIarbitraryheapcertified.','native_integration_adapted':True,'exclude_untouched_native_aggregates':True}
(P/'graph_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('GRAPH',nw,nb,report['GPU_gates'],report['CPU_gates'])

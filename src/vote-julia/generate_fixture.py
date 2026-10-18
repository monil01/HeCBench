from pathlib import Path
import json,struct
B=Path(__file__).resolve().parent;inputs=[];fixtures=[]
for n,kind in [(128,0),(1024,1),(1024,2),(128,3),(128,4)]:
 if kind==0:v=[0 if i<32 else (i if i%2 else 0) if i<64 else (0 if i%2 else i) if i<96 else 0xffffffff for i in range(n)]
 elif kind==1:v=[1<<(i//32) if i%32==i//32 else 0 for i in range(n)]
 elif kind==2:v=[0 if i%32==i//32 else 0xffffffff for i in range(n)]
 elif kind==3:v=[0]*n
 else:v=[1<<(i%32) for i in range(n)]
 fixtures.append(dict(n=n,kind=kind,offset=len(inputs)));inputs.extend(v)
cases=[];gold=[];prefix=0;out=0
for f in fixtures:
 n=f['n'];v=inputs[f['offset']:f['offset']+n]
 for op in [0,1]:
  expected=[]
  for start in range(0,n,32):
   pred=[x!=0 for x in v[start:start+32]];value=any(pred) if op==0 else all(pred);expected.extend([int(value)]*32)
  for repeat in [0,1,17,1024]:
   answer=expected if repeat else [0xdeadbeef]*n;cases.append(dict(op=op,n=n,fixture=f['kind'],input_offset=f['offset'],repeat=repeat,initial_mode=0,oracle_offset=len(gold),output_count=n));gold.extend(answer);prefix+=2*(n+32);out+=n+32
for mode in [0,1]:
 initial=[int(mode==1 and t%2==0 and field==2) for t in range(96) for field in range(3)]
 expected=initial.copy()
 for t in range(96):
  pred=[j>=48 for j in range(t//32*32,(t//32+1)*32)];expected[3*t]=int(any(pred));expected[3*t+1]=int(t>=48)
  if all(pred):expected[3*t+2]=1
 for repeat in [0,1,17,1024]:
  cases.append(dict(op=2,n=96,fixture=-1,input_offset=-1,repeat=repeat,initial_mode=mode,oracle_offset=len(gold),output_count=288));gold.extend(expected if repeat else initial);prefix+=320;out+=320
(B/'inputs_u32.bin').write_bytes(struct.pack('<'+'I'*len(inputs),*inputs));(B/'oracle_u32.bin').write_bytes(struct.pack('<'+'I'*len(gold),*gold));(B/'case_table.jl').write_text('const VOTE_CASES = '+repr([(a['op'],a['n'],a['input_offset'],a['repeat'],a['initial_mode'],a['oracle_offset'],a['output_count']) for a in cases]).replace('), (','),\n (')+'\n');(B/'case_table.h').write_text('struct VoteCase{int op,n,input_offset,repeat,initial_mode,oracle_offset,output_count;};\nstatic const VoteCase CASES[]={\n'+',\n'.join('{'+','.join(str(a[k]) for k in ['op','n','input_offset','repeat','initial_mode','oracle_offset','output_count'])+'}' for a in cases)+'\n};\n')
(B/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=prefix,complete_capture_count=prefix+out,physical_guards_per_side=16,warp_size=32,full_active_mask='0xffffffff',oracle='IndependentPython truth-table any/all on each32-elementlist. Directedinfo firstfieldwarpany(tx>=48),secondfieldpredicate,thirdconditionalwrite1iffwarpall, otherwiseactualinitialbituntouched. Sourceoriginalpattern, allzero/nonzero,1024threadonehot/onezero coversall32predicate-lanepositions.',timing='Sum48single-kernellaunch/contextcompletionhostwindows, originalGPUinternalrepeat0/1/17/1024; noouteramortization, noreset/copy/setup/JIT/oracle/capture inside. Fivecheckedtrials mean2-5.',scope='CUDA32lane fullmaskwarpcollective/statefulconditionalwriteoperator, fullactivewarpsonly. Source1024-thread+repeat0controls explicitnotoriginalCLI128/96/10millionrepeat. Repeat0iszero-work/untouched-outputcontrol; no measuredinstructioncount/vote-throughput/FLOProoflineclaim. OriginalMakefile10millionGPUinternalrepeats unmeasured.'),indent=2)+'\n')

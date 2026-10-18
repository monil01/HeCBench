from pathlib import Path
import struct,json
p=Path(__file__).resolve().parent;fixture=[];oracle=[];cases=[]
for ty in [1,2,3]:
 for ci,(n,t,blocks,threads) in enumerate([(128,2,1,64),(4099,3,3,128),(65536,1,8,256),(3897,1,3,128)]):
  lanes=2 if ty==1 else 1
  a=[-1.0 if ci<3 or i%3==0 else 0.0 if i%3==1 else -0.0 for i in range(n*lanes)]
  gold=[-1.0 if x==-1 else -2.0 for x in a]
  fixture += [ty,n,t,blocks,threads]+a
  oracle += [-777.0]*(8*lanes)+gold+[-777.0]*(8*lanes)
  cases.append(dict(type=ty,storage='packedhalf2' if ty==1 else 'Float32' if ty==2 else 'Float64',n=n,ntrials=t,blocks=blocks,threads=threads,lanes=lanes,nontrivial=ci==3))
p.joinpath('fixture.bin').write_bytes(struct.pack('<%sd'%len(fixture),*fixture));p.joinpath('oracle.bin').write_bytes(struct.pack('<%sd'%len(oracle),*oracle));p.joinpath('fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(fixture),output_count=len(oracle),oracle='Independentalgebraicfixedpoint -1 preserved at everyoperation; signedzero innertrial1 givesbeta2andoutput-2; mixedpackedlanesdifferent, nofakeFloat32half2',atol=0,rtol=0),indent=2))

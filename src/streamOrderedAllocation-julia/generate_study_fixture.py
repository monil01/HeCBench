from pathlib import Path
import struct,json
p=Path(__file__).resolve().parent;pre=[4];gold=[];cases=[]
for ci,n in enumerate([1,255,257,4097]):
 a=[((i*13+ci*7)%127-63)/8 for i in range(n)];b=[((i*19+ci*11)%113-56)/16 for i in range(n)];pre.extend([n,100]);pre.extend(a);pre.extend(b)
 for mode in range(3):
  # Integer numerator addition yields an exact binary32 sum on this range.
  out=[(round(x*16)+round(y*16))/16 for x,y in zip(a,b)]
  for it in range(100):gold.extend(out)
  gold.extend(a);gold.extend(b)
 cases.append(dict(n=n,repeat=100,paths=3,oracle='Integer numerator addition in sixteenth units; exact FP32 bounded dyadic result',source_guard_storage=False))
(p/'fixture.bin').write_bytes(struct.pack('<'+str(len(pre))+'d',*pre));(p/'oracle.bin').write_bytes(struct.pack('<'+str(len(gold))+'d',*gold));(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(pre),oracle_count=len(gold),artifact_count=len(pre)+len(gold),captured='All100 complete outputvectors perpath; finaliteration actualGPUinputechoes. Originaldeviceallocations havenoguards; noguardclaim.',instrumentation='Twoadditional finaliterationD2H inputechocopies insideROI beforeoriginalinputfrees, all4. Hostoutput100distinct preallocatedslots preserves sourceoverwrittenoutputs withoutmoredevicecopies.'),indent=2)+'\n');print(len(pre),len(gold))

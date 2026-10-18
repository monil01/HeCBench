from pathlib import Path
import json,struct
p=Path(__file__).resolve().parent;pre=[4];gold=[];cases=[];G=16384
for ci,(T,B,Q,P,it) in enumerate([(32,3,7,21,0),(64,320,320,640,3),(128,7,17,51,9),(256,5,3,9,1)]):
 ops=[1 if ci==2 else 2 if ci==3 else 1 if (i*7+i//3)%5<3 else 2 for i in range(P)];pre.extend([T,B,Q,P,it]);pre.extend(ops);whole=[]
 for off in range(0,P,Q):
  out=[T*(it if ops[off+j]==1 else 1)+off+j for j in range(Q)for lane in range(T)];whole.extend(out)
  gold.extend([G]*64+[0]*(Q*T)+[G]*64);gold.extend([G]*64+out+[G]*64)
  gold.extend([G,G]*64)
  for j in range(Q):gold.extend([off+j,ops[off+j]])
  gold.extend([G,G]*(B+64));gold.extend([G]*64+[Q+B]+[G]*64)
 gold.extend(whole);gold.append(P);cases.append(dict(T=T,blocks=B,logical_queue=Q,pool=P,iterations=it,terminal_padding_records=B,terminal_padding_bytes=8*B,chunks=P//Q,ops=ops))
(p/'fixture.bin').write_bytes(struct.pack('<'+str(len(pre))+'d',*pre));(p/'oracle.bin').write_bytes(struct.pack('<'+str(len(gold))+'d',*gold));(p/'fixture_description.json').write_text(json.dumps(dict(cases=cases,input_prefix_count=len(pre),oracle_count=len(gold),artifact_count=len(pre)+len(gold),oracle='Independentintegerclosedformexpectedvalue validatesrealexecutedpersistentqueue/heavyloop; operatornotreplacedbyformula.',captures='EachchunkactualGPUzeroinput/fulloutput/taskqueue+initializedterminalpadding/consumed+allguards,finalhostpoolandwrittenCPUcounter. FourextraD2H auditcopies/chunkinsideROIall4.',source_guard_policy='PhysicalQ+Btaskrecords plus64recordprefix/suffixguards, sourceunconditionalterminalqueuefetchesvalidphysically. Nativekernelbyteunchanged, conditionalhostallocationintervention.'),indent=2)+'\n');print(len(pre),len(gold))

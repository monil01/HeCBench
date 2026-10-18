from pathlib import Path
import json,struct,hashlib
p=Path(__file__).resolve().parent
sets=[[(8*g+i/128,0.) for g,n in enumerate([67,65,60]) for i in range(n)],[(i/4,0.) for i in range(128)],[(0.,0.),(0.,0.)]+[(8+2*i,0.) for i in range(63)]]
pre=[len(sets)];meta=[]
for ci,pts in enumerate(sets,1):
 N=len(pts);full=[[abs(x-u) for u,v in pts] for x,y in pts];neighbors=[[j for j in range(N) if j!=i and full[i][j]<1] for i in range(N)];D=max(map(len,neighbors));idx=[];dist=[]
 for i,row in enumerate(neighbors):
  idx.extend(row+[-1]*(D-len(row)));dist.extend([full[i][j] for j in row]+[float.fromhex('0x1.fffffep+127')]*(D-len(row)))
 pre.extend([N,D,1.]);pre.extend(z for xy in pts for z in xy);pre.extend(idx);pre.extend(dist)
 remaining=list(range(N));mask=[0]*N;records=[];logical=[]
 def candidate(seed):
  cl=[seed]
  while len(cl)<len(remaining):
   candidates=[j for j in remaining if j not in cl and all(full[j][q]<1 for q in cl)]
   if not candidates:break
   q=min(candidates,key=lambda j:(max(full[j][z] for z in cl),j));cl.append(q)
  return cl
 while remaining:
  before=remaining[:];candidates=[candidate(seed) for seed in remaining];winner=min(zip(remaining,candidates),key=lambda sc:(-len(sc[1]),before.index(sc[0])));seed,cl=winner
  remaining=[j for j in remaining if j not in cl]
  for j in cl:mask[j]=1
  logical.extend([len(before),len(cl),seed,len(remaining)]);logical.extend(cl);logical.extend(remaining);logical.extend(mask)
  records.append(dict(point_count=len(before),cardinality=len(cl),seed=seed,cluster=cl,remaining=remaining[:],clustered_mask=mask[:],diameter=max(full[a][b] for a in cl for b in cl)))
  if len(cl)<=1:break
 gold=[N,D,len(records)]+list(map(len,neighbors))+logical
 (p/f'oracle_case{ci}.bin').write_bytes(struct.pack('<'+str(len(gold))+'d',*gold))
 meta.append(dict(case=ci,N=N,D=D,threshold=1.,iterations=len(records),oracle_values=len(gold),records=records,source_stop_remaining=len(remaining),coordinate_domain='Exactdyadic1Dpointsembeddedintwo-dimensionalEuclideanplane; noquantization/thresholdambiguity',source_graph_valid=True))
(p/'fixture.bin').write_bytes(struct.pack('<'+str(len(pre))+'d',*pre))
(p/'fixture_description.json').write_text(json.dumps(dict(cases=meta,input_prefix_values=len(pre),graph_contract='SortednonselfneighborswithstrictEuclideandistance<threshold, -1/FLT_MAXpadding. DistancesexactdyadicFloat32; independentoracleusesallpaircoordinategeometry, notcompactcaches.',algorithm='Greedycomplete-linkcandidateperseed, minimummax-distancewithlowestIDtie; selectlargestcandidate, earliestactive-seedontie. Notglobaloptimalclique/QTpartitionclaim. Stopaftercardinality<=1asoriginal, retainingactualresidualunassignedlist.',coverage='Degree66exercisestworegisterprefetchslots, chainmultistepdiameterupdates/ties/removal, partial64threadtails, isolatedsource-stop; prefetch>12fallback/degree>768/default4096distributedMPIunverified.'),indent=2)+'\n')
print(len(pre),[(x['N'],x['D'],x['iterations'],x['source_stop_remaining']) for x in meta])

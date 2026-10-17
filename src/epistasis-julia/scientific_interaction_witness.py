"""Raw-genotype independent score and planted-interaction witness, no GPU run."""
from pathlib import Path
import array,json,math,sys
rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());f=json.loads((Path(__file__).parent/'study_fixture.json').read_text());N=f['patients'];S=f['snps'];raw=f['raw_genotypes'];ph=f['phenotype'];tables=[];scores=[float.fromhex('0x1.fffffep127')]*(S*S);exact=[]
def g(n):return (n+.5)*math.log(n)-(n-1)if n else 0.
for i in range(S):
 for j in range(i+1,S):
  counts=[[0]*9 for _ in range(2)]
  for patient in range(N):counts[ph[patient]][raw[patient*S+i]*3+raw[patient*S+j]]+=1
  assert [sum(c)for c in counts]==[ph.count(0),ph.count(1)]
  score=abs(sum(g(a+b+1)-g(a)-g(b)for a,b in zip(*counts)));scores[i*S+j]=score or float.fromhex('0x1.fffffep127');ex=abs(sum(math.lgamma(a+b+2)-math.lgamma(a+1)-math.lgamma(b+1)for a,b in zip(*counts)));exact.append(abs(score-ex));tables.append((i,j,counts))
# Repack independently using explicit set-bit positions, not the generator's shifting recurrence.
for phenotype,field in [(0,'packed_zero'),(1,'packed_one')]:
 subjects=[p for p in range(N)if ph[p]==phenotype];pp=(len(subjects)+31)//32;packed=[0]*(pp*S*2)
 for group in range(pp):
  patients=subjects[group*32:(group+1)*32];length=len(patients)
  for snp in range(S):
   for position,patient in enumerate(patients):
    genotype=raw[patient*S+snp]
    if genotype<2:packed[(group*S+snp)*2+genotype]|=1<<(length-position-1)
 assert packed==f[field]
 def_mask=(1<<(len(subjects)%32 or 32))-1
 assert def_mask==f['masks'][phenotype]
def mutual_info(group):
 joint={};cx={};cy={}
 for p in range(N):
  x=tuple(raw[p*S+s]for s in group);y=ph[p];joint[x,y]=joint.get((x,y),0)+1;cx[x]=cx.get(x,0)+1;cy[y]=cy.get(y,0)+1
 return sum(c/N*math.log2(c*N/(cx[x]*cy[y]))for(x,y),c in joint.items())
minimum=min(range(len(scores)),key=scores.__getitem__);assert [minimum//S,minimum%S]==[0,1]
report={'oracle':'Exact raw genotype/phenotype3x3 counts; Float64 evaluation of unchanged source approximation. Independent direct-bit-position repacking verifies all packed input words.','input_prefix_count':len(f['prefix']),'computed_pair_count':S*(S-1)//2,'untouched_sentinel_count':S*(S+1)//2,'phenotype_counts':[ph.count(0),ph.count(1)],'masks':f['masks'],'minimum_pair':[0,1],'planted_pair_score':scores[1],'source_approximation_vs_exact_logfactorial_max_abs':max(exact),'mutual_information_bits':{'snp0':mutual_info([0]),'snp1':mutual_info([1]),'pair0_1':mutual_info([0,1])},'trials':[],'claim_limits':'Synthetic XOR recovery does not validate statistical calibration or general association testing. Exact log-factorial discrepancy is source model approximation, distinct from Float32 arithmetic error. Full scores checked; GPU contingency tables are not separately exported.'}
for model,spec in r['models'].items():
 for t in spec.get('trials',[]):
  data=array.array('d');data.frombytes(Path(t['output_artifact']['saved_path']).read_bytes())
  if sys.byteorder!='little':data.byteswap()
  assert list(data[:len(f['prefix'])])==f['prefix']
  out=data[len(f['prefix']):];assert len(out)==len(scores);error=max(abs(a-b)for a,b in zip(out,scores));assert error<1e-3,(model,error)
  assert min(range(len(out)),key=out.__getitem__)==minimum
  report['trials'].append({'model':model,'trial':t['trial'],'exact_prefix':True,'all_scores_max_abs_error_vs_float64_source_formula':error,'planted_pair_rank1':True})
p=rp.parent/'scientific_interaction_witness.json';p.write_text(json.dumps(report,indent=2)+'\n');print(p)

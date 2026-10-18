from pathlib import Path
import itertools,struct,json
p=Path(__file__).resolve().parent;n=256
core=[[28,9,2,35,32,5],[13,4,21,36,6,2],[7,26,13,37,10,25],[29,21,39,1,9,16],[8,20,37,2,24,37],[7,25,23,34,28,29]]
cases=[];prefix=[n,3];gold=[]
for ci in range(3):
 if ci==0:
  a=[[100000 if r!=c else 0 for c in range(n)] for r in range(n)]
  for r in range(6):
   for c in range(6):a[r][c]=core[r][c]
 elif ci==1:
  a=[[100000+((r*17+c*13)%257) for c in range(n)] for r in range(n)]
  for r in range(n):a[r][(r*5+7)%n]=r%11
 else:
  a=[[100000 if r!=c else 0 for c in range(n)] for r in range(n)]
  for r in range(6):
   for c in range(6):a[r][c]=core[5-r][(c+2)%6]+3*r-2*c+20
 vals=sorted((sum(a[r][pm[r]] for r in range(6)),pm) for pm in itertools.permutations(range(6))) if ci!=1 else None
 if ci==1:assign=[(r*5+7)%n for r in range(n)];gap=99989
 else:assert vals[0][0]<vals[1][0];assign=list(vals[0][1])+list(range(6,n));gap=vals[1][0]-vals[0][0]
 inv=[assign.index(c) for c in range(n)];total=sum(a[r][assign[r]] for r in range(n));flat=[a[r][c] for c in range(n) for r in range(n)];prefix+=flat;gold+=assign+inv+[total]+flat
 rowmin=[min(row) for row in a];b=[[a[r][c]-rowmin[r] for c in range(n)] for r in range(n)];colmin=[min(b[r][c] for r in range(n)) for c in range(n)];z=[[b[r][c]-colmin[c] for c in range(n)] for r in range(n)];initialzeros=sum(v==0 for row in z for v in row)
 perfectzero=any(all(z[r][pm[r]]==0 for r in range(6)) for pm in itertools.permutations(range(6))) if ci!=1 else True
 cases.append(dict(case=ci,n=n,optimal_cost=total,unique_optimum_gap=gap,expected_assignment=assign,oracle='All720permutations of active6x6core plus offblockdominance>=100000 vsboundedcorecost; remaining250 forceddiagonal' if ci!=1 else 'Planteduniquepermutation, rowwise unique minima andpositivegap>=99989',initial_zero_count=initialzeros,initial_zero_graph_has_perfect_matching=perfectzero,requires_step6=not perfectzero))
pack=lambda f,v:(p/f).write_bytes(struct.pack('<'+'i'*len(v),*v))
pack('fixture.bin',prefix);pack('oracle.bin',gold)
(p/'fixture_description.json').write_text(json.dumps(dict(n=n,cases=cases,input_prefix_count=len(prefix),oracle_values=len(gold),artifact_values=len(prefix)+len(gold),expected_full_outputs='Fullrowassignment+fullinversecolumnassignment+totalcost+actualoriginalcostecho. Nativeglobalarrays havenooutputguards; Juliaextrascratchguards checked separately, notfakecommonartifactguards.',original_greedy_counterexample=dict(cost=[[1,2],[1,100]],greedy_cost=101,optimal_cost=3)),indent=2)+'\n')

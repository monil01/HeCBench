from pathlib import Path
import json,sys
P=Path(sys.argv[1]);N=22283
for x in ['queries','reffiles-tab','Results']: (P/x).mkdir(parents=True,exist_ok=True)
indices=[(i*677+19)%N for i in range(31)]
(P/'queries/study.sig').write_text('AffyProbeSetID regulation\n'+''.join(f'G{x:05d}\t{1 if k%3 else -1}\n'for k,x in enumerate(indices)))
a=[(i*73)%N-11141 for i in range(N)];b=[(i*97+13)%N-11141 for i in range(N)];c=[-v for v in a]
for name,vals in [('A_1_C_1.tab',a),('A_1_C_2.tab',b),('B_1_C_1.tab',c)]:
 (P/'reffiles-tab'/name).write_text('AffyProbeSetID\tregulation\n'+''.join(f'G{i:05d}\t{v}\n'for i,v in enumerate(vals)))
print(json.dumps({'genes':N,'query_genes':31,'profiles':3,'drug_sets':2,'set_sizes':[2,1],'original_default_random_generations':100000}))

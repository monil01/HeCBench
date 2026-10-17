"""Independent exact modular RNG, counting rank and full GPU-sort output witness."""
import array
import collections
import json
import pathlib
import sys
state=314159265;multiplier=1220703125;modulus=1<<46;original=[]
for _ in range(65536):
    total=0
    for _ in range(4):state=(state*multiplier)%modulus;total+=state
    original.append((total*512)//modulus)
indices=[48427,17148,23627,62548,4431];known=[0,18,346,64917,65463];final=original.copy();known_checks=0
for iteration in range(1,25):
    final[iteration]=iteration;final[iteration+24]=2048-iteration
    counts=collections.Counter(final);prefix=[];total=0
    for key in range(2048):total+=counts[key];prefix.append(total)
    for j,index in enumerate(indices):
        rank=prefix[final[index]-1];expected=known[j]+iteration if j<3 else known[j]-iteration
        assert rank==expected;known_checks+=1
exclusive=[prefix[k]-counts[k]for k in range(2048)]
expected=original+final+prefix+[final[i]for i in indices]+[120]+[prefix[i*256-1]if i else 0 for i in range(8)]+sorted(final)+exclusive+[0]*256
assert len(expected)==200974
result=json.loads(pathlib.Path(sys.argv[1]).read_text());rows=[]
for model,record in result['models'].items():
    for trial in record['trials']:
        got=array.array('i');got.frombytes(pathlib.Path(trial['output_artifact']['saved_path']).read_bytes());assert list(got)==expected
        rows.append({'model':model,'trial':trial['trial'],'all200974_retained_integers_match':True,'full_initial_GPU_LCG_and_final_GPU_sorted_array_match':True,'known_partial_rank_count':120,'full_sort_violations':0})
assert len(rows)==20
print(json.dumps({'benchmark':'is','trials':rows,'independent_method':'ArbitraryprecisionintegerLCGmod2^46,Counterhistograms/directprefixandhostsortfororacleonly;GPUimplementationsstillgenerate/sort/checkonGPU','known_partial_rank_checks_independently_reproduced':known_checks,'intermediate_scope':'Fullinitialanditeration24statesretained;intermediateiterationscheckedbynativefivepartialranktests/count,notfullintermediatehistograms','scratch_scope':'Only8liveoffsetsretained;remaining248inactiveinplacescanlanesnotmeaningfulscientificoutputs','class':'S','scope':'Default256threadgeometries/24iterationsonly'},indent=2))

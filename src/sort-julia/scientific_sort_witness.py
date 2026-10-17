"""Independent exact sorted multiset, penultimate digit state, and group offsets."""
from pathlib import Path
import array,json,sys,collections

def fixture(n):
    return [i%16 if n==4096 else 0xffffffff if i%17==0 else 0 if i%19==0 else ((i*2654435761)^((i>>3)*2246822519))&0xffffffff for i in range(n)]
def expected(input):
    final=sorted(input);scratch=sorted(final,key=lambda x:x&0x0fffffff);n=len(input);region=(n//4)//64*4;counts=[0]*1024
    for group in range(64):
        start=group*region;stop=n if group==63 else start+region
        for key in scratch[start:stop]:counts[(key>>28)*64+group]+=1
    scan=[];seed=0
    for count in counts:scan.append(seed);seed+=count
    assert seed==n and collections.Counter(final)==collections.Counter(input)
    return final+scratch+scan

def main(path):
    path=Path(path);r=json.loads(path.read_text());prefix=[];outputs=[];quality=[]
    for n in (4096,65540):
        original=fixture(n);prefix.extend(original+[n,64,3]);outputs.extend(expected(original));quality.append({'n':n,'distinct_keys':len(set(original)),'min_key':min(original),'max_key':max(original),'duplicates_present':len(set(original))<n,'last_group_four_key_vectors':n//4-(63*((n//4)//64)),'second_window_exercised':n==65540,'exact_multiset_preserved':True,'all_8_radix_digits_exercised':n==65540})
    assert len(prefix)==69642 and len(outputs)==141320;want=array.array('I',prefix+outputs).tobytes();records=[]
    for model,m in r['models'].items():
        assert len(m['trials'])==5
        for trial in m['trials']:
            raw=Path(trial['output_artifact']['saved_path']).read_bytes();assert raw[:69642*4]==want[:69642*4];assert raw==want
            records.append({'model':model,'trial':trial['trial'],'exact_input_prefix_values':69642,'full_output_values':141320,'sorted_scratch_and_offsets_exact':True})
    report={'result_path':str(path.resolve()),'all_model_trials':records,'input_properties':quality,'oracle':'IndependentPythonstablecomparison-sort,fullmultisetCounter,low28bittie-orderanddigit/groupintegerhistogram-prefixscan;notGPUradixpipelineoradjacency-onlymonotonicity.','scope':'Primaryauthoredradixonly;secondarynativeThrustlibrarycomparisonunmeasured. Divisible4sizes4096/65540,256threads64groups,3repeatswithoutinternalreset(oneunsorted/twoalready-sorted),all32bitsinlargercase.','intermediate_state':'Finalscratchisstablelow28sortofalreadyfullysortedkeysbecauselastfullsortstartsfromsortedarray;finaltopscanoffsetscomputedfromthatstate. Noassociated-keypayloadstabilityclaimbeyondkeybufferordering.'}
    out=path.parent/'scientific_sort_witness.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
if __name__=='__main__':main(sys.argv[1])

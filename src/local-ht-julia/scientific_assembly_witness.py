"""Independent read-window counters, quality policy, de Bruijn walk and hash witness."""
from pathlib import Path
import array,collections,json,struct,sys
K=21;HT=512;MAXW=16;MAXMER=121;CAP=64

def hash_key(key):
    h=0x3FB0BB5F^len(key);mult=0x5bd1e995;mask=(1<<32)-1;blocks=len(key)//4
    for block in range(blocks):
        word=sum(key[4*block+j]<<(8*j) for j in range(4));word=word*mult&mask;word^=word>>24;word=word*mult&mask;h=((h*mult)&mask)^word
    tail=key[blocks*4:]
    h^=sum(value<<(8*j) for j,value in enumerate(tail))
    if tail:h=h*mult&mask
    h^=h>>13;h=h*mult&mask;h^=h>>15
    return h%HT

def rating(votes,high):
    if votes==0:return 0
    if votes==1:return 1
    if votes<2:return 2
    if votes<3:return 3 if high<2 else 4
    if high<2:return 5
    if 2<high<3:return 6
    return 7

def choose(low,high):
    bases=sorted(range(4),key=lambda b:(rating(low[b],high[b]),high[b],low[b]),reverse=True)
    a,b,c=bases[:3];top=rating(low[a],high[a]);runner=rating(low[b],high[b]);ext=ord('X')
    if (0<top<=3 and runner==0)or(3<top<6 and runner<3)or(top==6 and runner<4)or(top==7 and runner<7):ext=b'ACGT'[a]
    elif top==7 and runner==7:
        if rating(low[c],high[c])==7 or low[a]==low[b]:ext=ord('F')
        else:ext=b'ACGT'[a if low[a]>low[b] else b]
    count=low[b'ACGT'.index(ext)] if ext in b'ACGT' else 0
    return ext,count

def recompute(fixture):
    reads=bytes(fixture['reads']);quals=fixture['quals'];contigs=bytes(fixture['contigs']);read_end=fixture['read_offsets'];read_groups=fixture['read_counts'];all_outputs=[];quality=[];first_read=0;first_contig=0
    for group in range(5):
        stop_read=read_groups[group];stop_contig=fixture['ctg_offsets'][group];seed=contigs[first_contig:stop_contig];first_contig=stop_contig;votes=collections.Counter();keys=set()
        for read in range(first_read,stop_read):
            start=0 if read==0 else read_end[read-1];stop=read_end[read];assert stop-start>=K+1
            for position in range(start,stop-K):
                key=reads[position:position+K];base=reads[position+K];phred=quals[position+K]-33;keys.add(key)
                if base not in b'ACGT':continue
                if phred>=10:votes[(key,base,False)]+=1
                if phred>=20:votes[(key,base,True)]+=1
        first_read=stop_read;records=[];extensions={}
        for key in sorted(keys):
            low=[votes[(key,base,False)] for base in b'ACGT'];high=[votes[(key,base,True)] for base in b'ACGT'];ext,count=choose(low,high);extensions[key]=ext
            records.append(list(key)+low+high+[ext,count,hash_key(key)])
        key=seed;walk=[];visited=[];seen=set();termination='limit'
        for _ in range(MAXW):
            if key in seen:termination='R';break
            seen.add(key);visited.append(key);extension=extensions.get(key,ord('X'))
            if extension in b'FX':termination=chr(extension);break
            walk.append(extension);key=key[1:]+bytes([extension])
        temp=list(seed)+walk;output=[len(walk)]+walk+[0]*(MAXW-len(walk))+temp+[0]*(MAXW+MAXMER-len(temp))+[len(records)]
        for record in records:output.extend(record)
        output.extend([0]*((CAP-len(records))*32));output.append(len(visited))
        for key in sorted(visited):output.extend(list(key)+[1])
        output.extend([0]*((MAXW-len(visited))*22));assert len(output)==2556;all_outputs.extend(output)
        quality.append({'group':group,'seed':seed.decode(),'occupied_keys':len(keys),'visited_keys':len(visited),'expected_walk':bytes(walk).decode(),'independent_graph_termination':termination,'observed_gpu_termination_flag_exported':False,'complete_records_include_actual_gpu_hash':True})
    assert [x['expected_walk'] for x in quality]==['ACTGCA','','','TGA','CGTA']
    assert [x['independent_graph_termination'] for x in quality]==['X','F','X','X','R']
    return all_outputs,quality

def main(result_path):
    path=Path(result_path);result=json.loads(path.read_text());fixture=json.loads((Path(__file__).parent/'study_fixture.json').read_text());expected,quality=recompute(fixture)
    prefix=fixture['contigs']+fixture['reads']+fixture['quals']+fixture['ctg_offsets']+fixture['read_offsets']+fixture['read_counts']+fixture['prefix_ht']
    for depth in fixture['depth']:prefix.extend(struct.unpack('<II',struct.pack('<d',depth)))
    prefix.extend([21,512,16,121,64,5,33,512,1]);assert len(prefix)==3211;want=array.array('I',prefix+expected).tobytes();records=[]
    for model,data in result['models'].items():
        assert len(data['trials'])==5
        for trial in data['trials']:
            raw=Path(trial['output_artifact']['saved_path']).read_bytes();assert raw[:3211*4]==want[:3211*4];assert raw==want
            records.append({'model':model,'trial':trial['trial'],'exact_input_prefix_values':3211,'all_canonical_stage_output_values':12780,'all_counts_quality_extensions_gpu_hashes_visited_walks_exact':True})
    report={'result_path':str(path.resolve()),'all_model_trials':records,'scientific_cases':quality,'oracle':'Independent Python Counter over read windows and Phred thresholds, direct quality decision, dictionary/de Bruijn graph walk, canonical Murmur2 integer hash. No CUDA table probing or copied golden res files.','numerical_evidence_strength':'Canonicalized complete occupied k-mer and visited records plus full initialized walk/temp buffers. Raw slot order and device pointer identity intentionally not compared.','native_reference':'Conditionally repaired lane-mask initialization, original minimal GPU probe had no observed mismatch despite source undefined value. Corrected baseline excluded from untouched-native aggregates.','scope':'K21, five small right-walk fixtures. Not full dataset loading, left reverse-complement/stitching, arbitrary depth, short reads, other K values or formal race-freedom certification.'}
    out=path.parent/'scientific_assembly_witness.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
if __name__=='__main__':main(sys.argv[1])

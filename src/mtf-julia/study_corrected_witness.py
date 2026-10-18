"""Corrected-secondary online-MTF oracle using recency ranks, plus decoder."""
from pathlib import Path
import json,array,hashlib,sys
P=Path(__file__).resolve().parent

def oracle(word):
 last=[-1]*256;ranks=[]
 for i,w in enumerate(word):
  j=last[w];rank=w+sum(x>=0 for x in last[w+1:])if j<0 else sum(x>j for x in last);ranks.append(rank);last[w]=i
 return ranks,sorted(range(256),key=lambda x:(last[x]<0,x if last[x]<0 else -last[x]))

def decoder(ranks):
 a=list(range(256));out=[]
 for r in ranks:w=a.pop(r);a.insert(0,w);out.append(w)
 return out

def main():
 result=Path(sys.argv[1]).resolve();d=json.loads(result.read_text());words=json.loads((P/'corrected_secondary_fixtures.json').read_text());prefix=[];expected=[];cases=[]
 for i,word in enumerate(words):
  prefix += [i,len(word),1]+word;ranks,alpha=oracle(word);assert decoder(ranks)==word;expected+=ranks+alpha;cases.append({'id':i,'length':len(word),'distinct_symbols':len(set(word)),'online_zero_rank_count':ranks.count(0),'rank_sum_equals_logical_prefix_D2H_bytes':sum(ranks),'logical_full_alphabet_H2D_bytes':256*len(word),'final_alphabet_is_permutation':sorted(alpha)==list(range(256))})
 assert len(prefix)==2626 and len(expected)==4903;captures=[];guards={}
 for model,rec in d['models'].items():
  tr=rec.get('trials',[]);log=Path(tr[0]['output_artifact']['saved_path']).parent/f'{model}_run1.log';lines=[x for x in log.read_text().splitlines()if x.startswith('STUDY_GUARDS ')];assert len(lines)==45 and all('exact=true'in x for x in lines);guards[model]={'raw_log':str(log),'sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'all_45_case_guards_pass':True,'explicit_guarded_allocations_per_case':4 if model=='cuda'else 5,'vendor_internal_allocation_guards':'Opaque; notclaimed'if model=='cuda'else None}
  for t in tr:
   p=Path(t['output_artifact']['saved_path']);a=array.array('I');a.frombytes(p.read_bytes());assert list(a[:2626])==prefix and list(a[2626:])==expected;captures.append({'model':model,'trial':t['trial'],'artifact':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'all_rank_finalalphabet_values_exact':True,'input_prefix_exact':True,'per_position_decoder_roundtrip':True})
 old=json.loads((P/'diagnostic_verification.json').read_text());output={'benchmark':'mtf','study_stratum':'corrected_secondary_online_algorithm','original_native_reference_repaired':True,'exclude_untouched_native_aggregates':True,'raw_result':str(result),'capture_count':len(captures),'input_prefix_count':len(prefix),'output_count':len(expected),'rank_values':2599,'final_alphabet_values':2304,'independent_mathematical_rule':'For a previously seen symbol, current rank equals count of symbols whose last occurrence is more recent. For a never-seen symbol w, rank=w+count(previously seen symbols greater than w). Final alphabet sorts seen symbols by decreasing recency, then unseen symbols ascending. This represents state as256 last-occurrence timestamps, independently of mutable-alphabet encoder movement.','decoder_rule':'Start ascending256byte alphabet; decoded symbol is alphabet[rank], then move chosen symbol to front. Every position is checked against original input.','known_answers':{'ABA':[65,66,1],'allbytes_ascending':'0..255','allbytes_descending':'255 repeated256times','repeated_zero':'allzero'},'source_original_bug_evidence':{'diagnostic':str(P/'diagnostic_verification.json'),'sha256':hashlib.sha256((P/'diagnostic_verification.json').read_bytes()).hexdigest(),'original_asan_retained':old['actual_CPU_ASan'],'original_init_only_semantics_still_wrong':old['independent_safe_witness']},'cases':cases,'logical_transfer_bytes_not_measured_traffic':{'alphabet_H2D':sum(c['logical_full_alphabet_H2D_bytes']for c in cases),'prefix_D2H':sum(c['rank_sum_equals_logical_prefix_D2H_bytes']for c in cases),'limitation':'Logical source-level alphabet/prefix payload only. NativeThrust temporary traffic and rank-result transfers/internal kernels opaque; do not label quotient as measured bandwidth.'},'guard_checks':guards,'captures':captures,'pass':len(captures)==20}
 (P/'corrected_independent_witness.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items()if k not in ['captures','guard_checks','source_original_bug_evidence','cases']},indent=2))
if __name__=='__main__':main()

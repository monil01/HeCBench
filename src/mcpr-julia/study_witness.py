"""Independent integer argmax-count/simplex witness for all three GPU layouts."""
import array,argparse,json,hashlib,collections
from pathlib import Path

def audit(path):
 result=json.loads(Path(path).read_text());reports=[];cache={}
 for model,info in result['models'].items():
  for trial in info.get('trials',[]):
   source=Path(trial['output_artifact']['saved_path']);blob=source.read_bytes();digest=hashlib.sha256(blob).hexdigest()
   if digest not in cache:
    a=array.array('d');a.frombytes(blob);off=0;inputs=[]
    for n,M in ((193,97),(5,3)):
     A=list(a[off:off+n*21]);off+=n*21;R=list(a[off:off+M*21]);off+=M*21;AS=list(a[off:off+n*21]);off+=n*21;RS=list(a[off:off+M*21]);off+=M*21;assert AS==[A[i*21+k]for k in range(21)for i in range(n)];assert RS==[R[m*21+k]for k in range(21)for m in range(M)];assert all(x==0 for x in a[off:off+3*n*21]);off+=3*n*21;assert list(a[off:off+7])==[n,21,M,3,192,96,32256];off+=7;inputs.append((n,M,A,R))
    assert off==25004;cases=[]
    for n,M,A,R in inputs:
     expected=[];counts=[];ties=0
     for i in range(n):
      rowcounts=collections.Counter()
      for m in range(M):
       scores=[A[i*21+k]+R[m*21+k]for k in range(21)];largest=max(scores);eligible=[k for k,s in enumerate(scores)if s==largest];winner=20 if 20 in eligible else min(eligible);rowcounts[winner]+=1;ties+=len(eligible)>1
      counts.extend(rowcounts[k]for k in range(21));expected.extend(rowcounts[k]/M for k in range(21))
     errors=[];simplex=[]
     for variant in range(3):
      got=list(a[off:off+n*21]);off+=n*21;error=max(abs(x-y)for x,y in zip(got,expected));assert error<=.001;assert [round(v*M)for v in got]==counts;assert all(0<=x<=1 for x in got);simplexerror=max(abs(sum(got[i*21:(i+1)*21])-1)for i in range(n));assert simplexerror<=1e-12;errors.append(error);simplex.append(simplexerror)
     report={'n':n,'M':M,'all_three_complete_probability_outputs_checked':3*n*21,'exact_integer_counts_recovered_all_variants':True,'maximum_probability_errors':errors,'simplex_max_abs_errors':simplex,'observed_tied_sample_rows':ties}
     if n==193:
      assert counts[3]==M and sum(counts[:21])==M
      margin=min(A[3]+R[m*21+3]-max(A[k]+R[m*21+k]for k in range(21)if k!=3)for m in range(M));assert margin>0;report['dominant_finite_fixture_category']=3;report['minimum_realized_dominance_margin']=margin
     else:
      actual=[max(range(21),key=lambda k:counts[i*21+k])for i in range(n)];assert actual==[20,0,20,4,20];report['tie_control_exact_winners']=actual
     cases.append(report)
    assert off==len(a)==37478;cache[digest]=cases
   reports.append({'model':model,'trial':trial['trial'],'artifact_sha256':digest,'cases':cache[digest]})
 return {'benchmark':'mcpr','result':str(Path(path).resolve()),'captured_trials_checked':len(reports),'distinct_complete_artifacts_independently_checked':len(cache),'verification_contract':'All three fullprobabilitymatrices, independent eligible-set argmax integercounts, row-simplex, layouttransposes andactualserializedinputs.','scientific_scope':'FinitefixedGaussian-noisecategoricalchoice frequencies andseparatenonstochastictiecontrols. Sharedrealizeddraws reused acrossparameterrows; notindependentdatasets. No trueGaussianpopulationprobabilityaccuracy, MCconvergenceororiginalCSVCLIcertificate.','reports':reports}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--out');args=p.parse_args();out=audit(args.result);text=json.dumps(out,indent=2)+'\n';print(text)
 if args.out:Path(args.out).write_text(text)

"""Exact paired CSR-to-dense witness from actual measured captures; no GPU work."""
import argparse,array,json,hashlib
from pathlib import Path
SHAPES=((17,19,88),(65,33,530))
def audit(path):
 d=json.loads(Path(path).read_text());reports=[]
 for model,info in d['models'].items():
  for trial in info.get('trials',[]):
   source=Path(trial['output_artifact']['saved_path']);a=array.array('d');a.frombytes(source.read_bytes());off=0;inputs=[]
   for m,n,nnz in SHAPES:
    dense=a[off:off+m*n];off+=m*n;rows=a[off:off+m+1];off+=m+1;assert all(x==-1 for x in a[off:off+nnz]);off+=nnz;assert all(x==0 for x in a[off:off+nnz]);off+=nnz;assert list(a[off:off+4])==[m,n,nnz,10];off+=4
    counts=[sum(v!=0 for v in dense[y*n:(y+1)*n])for y in range(m)];prefix=[0]
    for x in counts:prefix.append(prefix[-1]+x)
    assert list(rows)==prefix;inputs.append((m,n,nnz,dense,prefix))
   assert off==3796;cases=[]
   for m,n,nnz,dense,prefix in inputs:
    rows=list(a[off:off+m+1]);off+=m+1;cols=a[off:off+nnz];off+=nnz;values=a[off:off+nnz];off+=nnz;assert a[off]==nnz;off+=1;assert rows==prefix
    reconstructed=[0.]*(m*n)
    for y in range(m):
     first,last=int(rows[y]),int(rows[y+1]);rr=list(cols[first:last]);assert rr==sorted(set(rr))
     for k in range(first,last):
      col=int(cols[k]);assert col==cols[k]and 0<=col<n;assert values[k]!=0;reconstructed[y*n+col]=values[k]
    assert reconstructed==list(dense);cases.append({'m':m,'n':n,'nnz':nnz,'all_dense_values_reconstructed_exactly':m*n,'all_row_offsets_independently_counted':m+1,'column_value_pairing_preserved':True,'empty_rows':sum(prefix[y+1]==prefix[y]for y in range(m)),'full_rows':sum(prefix[y+1]-prefix[y]==n for y in range(m))})
   assert off==len(a)==5118;reports.append({'model':model,'trial':trial['trial'],'artifact_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':cases})
 return {'benchmark':'spd2s','result':str(Path(path).resolve()),'captured_trials_checked':len(reports),'verification_contract':'Complete pairedper-rowcanonicalCSR reconstructs everyinputdensevalue, independentoffsetcounts, exactzero-basedInt64indices.','checker_weakness_witness':{'native_checker':'Globally sortcolumnindices andvalues independently; pairinglost.','dense_row':[1,2],'correct_csr_columns':[0,1],'correct_values':[1,2],'wrong_values':[2,1],'independent_sorted_column_value_multisets_same':True,'paired_reconstruction_detects_error':True,'scope':'Minimalcheckerblindspot, not observedfailure ofnativecuSPARSEconversion.'},'scope':'cuSPARSEanalysis outsideROI,convert-only timedstage; +/-0 bothexcluded fromnonzeros andfullprefixbyteschecked; notallformats/densities/CLI performance.','reports':reports}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--out');args=p.parse_args();out=audit(args.result);text=json.dumps(out,indent=2)+'\n';print(text)
 if args.out:Path(args.out).write_text(text)

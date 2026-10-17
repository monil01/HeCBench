"""All saved sequence/scan/reduction values checked using exact integer identities."""
import argparse,array,json,hashlib
from pathlib import Path

def audit(path):
 d=json.loads(Path(path).read_text());reports=[]
 for model,m in d['models'].items():
  for t in m.get('trials',[]):
   source=Path(t['output_artifact']['saved_path']);a=array.array('i');a.frombytes(source.read_bytes());assert a.itemsize==4;off=0
   for n in (1025,65535):
    assert all(x==0 for x in a[off:off+2*n]);off+=2*n;assert list(a[off:off+4])==[n,3,0,1];off+=4
   assert off==133128;cases=[]
   for n in (1025,65535):
    vec=a[off:off+n];off+=n;scan=a[off:off+n];off+=n;scalar=a[off:off+3];off+=3
    assert all(vec[i]==i for i in range(n));assert all(scan[i]==i*(i+1)//2 for i in range(n));assert scan[0]==vec[0]and all(scan[i]-scan[i-1]==vec[i]for i in range(1,n));serialsum=sum(vec);assert list(scalar)==[serialsum,serialsum,0];assert serialsum<2**31
    cases.append({'n':n,'full_sequence_checked':n,'full_scan_checked':n,'all_prefix_first_differences_equal_sequence':True,'independent_int64_serial_sum':serialsum,'reduced_last_difference':list(scalar),'sum_fits_Int32':True})
   assert off==len(a)==266254;reports.append({'model':model,'trial':t['trial'],'artifact_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases':cases})
 return {'benchmark':'nosync','result':str(Path(path).resolve()),'captured_trials_checked':len(reports),'verification_contract':'Fullactualsequence,scan,reducedsum,lastscananddifference; exactintegeroracle/firstdifferenceidentity.','scope':'SourceThrust hostfuture/explicitstream calls retained viaCABI. This proves output/ordering correctness fornamedblockingstream/legacydefaultstream sizes, not actualkerneloverlap/asynchrony ofpar_nosync hint. Noauthoredframeworkdevicekernels.','reports':reports}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('--out');args=p.parse_args();out=audit(args.result);text=json.dumps(out,indent=2)+'\n';print(text)
 if args.out:Path(args.out).write_text(text)

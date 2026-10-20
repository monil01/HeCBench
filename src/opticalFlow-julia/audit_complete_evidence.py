#!/usr/bin/env python3
"""Complete optical-flow scientific arrays, actual input/guard bytes and mask joins."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(path):
 raw=path.read_bytes();h=struct.unpack_from('<8I',raw);alpha=struct.unpack_from('<f',raw,32)[0];assert h[0]==0x4f463537 and len(raw)==36+h[7]*4
 return h,alpha,np.frombuffer(raw,dtype='<f4',offset=36)
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--model',required=True);p.add_argument('--folder',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(__file__).resolve().parent;result=Path(a.result);r=json.loads(result.read_text());folder=Path(a.folder);expected=[];caseinfo=[]
 for c,name in enumerate(('original_pair','identity_partial')):
  h,alpha,values=fixture(root/f'{name}_fixture.bin');w,height,stride,levels,warps,solver=h[1:7];expected.extend((c,w,height,stride,alpha,levels,warps,solver));expected.extend(values);caseinfo.append((name,w,height,stride,values))
 raw=(root/'primitive_fixture.bin').read_bytes();assert struct.unpack_from('<2I',raw)==(0x4f465052,9);o=8;probes=[]
 for i in range(9):
  id,w,h,s=struct.unpack_from('<4I',raw,o);o+=16;assert id==i;n=h*s;values=np.frombuffer(raw,dtype='<f4',offset=o,count=n);o+=n*4;probes.append(values);expected.extend((i,w,h,s,n));expected.extend(values)
 assert o==len(raw);expected.append(np.float32(.2));expected.extend(np.zeros(2*256*13,dtype=np.float32));expected=np.array(expected,dtype='<f4');assert len(expected)==818494
 po=(root/'primitive_oracle.bin').read_bytes();magic,narr,nout=struct.unpack_from('<3I',po);assert(magic,narr,nout)==(0x4f464f52,11,28468);sizes=np.frombuffer(po,dtype='<u4',offset=12,count=11);ref=np.frombuffer(po,dtype='<f8',offset=56,count=nout);budgets=np.frombuffer(po,dtype='<f8',offset=56+nout*8,count=nout);assert len(po)==56+nout*16
 rows=[]
 for t in r['models'][a.model]['trials']:
  trial=t['trial'];path=Path(t['output_artifact']['saved_path']);assert sha(path)==t['output_artifact']['sha256'];x=np.fromfile(path,dtype='<f4');assert len(x)==1467732 and np.all(np.isfinite(x)) and np.array_equal(x[:818494].view(np.uint32),expected.view(np.uint32));cursor=818494;case_metrics=[];hostraw={}
  for c,(name,w,h,s,values)in enumerate(caseinfo):
   count=w*h*2;actual=x[cursor:cursor+count];cursor+=count;oracle=(root/f'{name}_oracle.bin').read_bytes();assert struct.unpack_from('<2I',oracle)==(w,h);reference=np.frombuffer(oracle,dtype='<f8',offset=8);assert len(reference)==count;error=np.abs(actual.astype(np.float64)-reference);bound=.05+.001*np.abs(reference);assert np.all(error<=bound)
   hostfile=folder/f'{a.model}_trial{trial}_case{c}.host.bin';hostbytes=hostfile.read_bytes();hv=np.frombuffer(hostbytes,dtype='<f4').reshape(2,h,s);assert np.array_equal(hv[:,:,:w].reshape(-1).view(np.uint32),actual.view(np.uint32));hostraw[c]=hostbytes
   legacy=(root/f'{name}_legacyCPU.bin').read_bytes();assert struct.unpack_from('<2I',legacy)==(w,h);old=np.frombuffer(legacy,dtype='<f4',offset=8);oldL1=float(np.abs(actual.astype(np.float64)-old).sum()/(w*h));case_metrics.append({'case':c,'w':w,'h':h,'full_active_values':count,'max_abs_error_vs_independent_Float64':float(error.max()),'max_budget_fraction':float((error/bound).max()),'legacy_CPU_meanL1':oldL1,'legacy_source_threshold_pass':oldL1<.05,'host_full_stride_bytes':len(hostbytes),'host_raw_sha256':sha(hostfile),'active_primary_to_full_host_exact':True,'padding_identity_claim':False})
   if c==1:assert float(np.max(np.abs(actual)))<=.05
  actual=x[cursor:];assert len(actual)==28468;pe=np.abs(actual.astype(np.float64)-ref);assert np.all(pe<=budgets);fraction=np.divide(pe,budgets,out=np.zeros_like(pe),where=budgets>0)
  physical=folder/f'{a.model}_trial{trial}.physical.bin';layout=folder/f'{a.model}_trial{trial}.layout.jsonl';raw=physical.read_bytes();records=[json.loads(s)for s in layout.read_text().splitlines()];assert len(records)==55;seen=set();guardbytes=0;source_inputs={};source_outputs={}
  for row in records:
   c,id,n=struct.unpack_from('<3Q',raw,row['offset_bytes']);assert(c,id,n)==(row['case'],row['id'],row['payload_bytes']);key=(c,id);assert key not in seen;seen.add(key);start=row['offset_bytes']+24;block=raw[start:start+n+512];assert len(block)==n+512 and block[:256]==bytes([0xa7])*256 and block[-256:]==bytes([0xa7])*256;guardbytes+=512;payload=block[256:-256]
   if c<2 and id in(8,9):source_outputs[(c,id)]=payload
   if c<2 and id in(12,13):source_inputs[(c,id)]=payload
  assert guardbytes==28160 and len([k for k in seen if k[0]==0])==22 and len([k for k in seen if k[0]==1])==18 and len([k for k in seen if k[0]==2])==15
  for c,(_,w,h,s,values)in enumerate(caseinfo):
   assert source_outputs[(c,8)]+source_outputs[(c,9)]==hostraw[c]
   assert source_inputs[(c,12)]+source_inputs[(c,13)]==values.tobytes()
  rows.append({'trial':trial,'primary_sha256':sha(path),'exact_actual_input_prefix_values':818494,'scientific_output_values':649238,'cases':case_metrics,'primitive_values':28468,'primitive_max_abs_error':float(pe.max()),'primitive_max_budget_fraction':float(fraction.max()),'actual_physical_guard_bytes':guardbytes,'physical_raw_sha256':sha(physical),'physical_raw_bytes':len(raw),'layout_sha256':sha(layout),'physical_allocation_records':55,'actual_GPU_physical_output_to_host_return_complete_bytes_exact':True,'actual_finest_GPU_images_to_shared_fixture_complete_bytes_exact':True})
 assert len(rows)==5;out={'benchmark':'opticalFlow','model':a.model,'correct':True,'result':str(result),'result_sha256':sha(result),'primary_values_per_trial':1467732,'input_prefix_values':818494,'independent_scientific_output_comparisons_all5':649238*5,'actual_physical_guards_all5':28160*5,'raw_output_identity_claim':False,'mask_scope':'Completeactive(u,v)fields+all11primitivearrays, rawdeadpadding/physicalscratchretainedwithoutnumericidentityclaim','trials':rows};Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'correct':True,'all5_scientific_values':649238*5,'actual_guard_bytes_all5':28160*5}))
if __name__=='__main__':main()

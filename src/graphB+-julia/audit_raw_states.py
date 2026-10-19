#!/usr/bin/env python3
"""CPU-only exact audit of retained complete graph states and physical order bijections."""
from pathlib import Path
import argparse,json,struct,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1]
def rd(p):
 b=p.read_bytes();assert len(b)%8==0;return list(struct.unpack('<'+'q'*(len(b)//8),b))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--raw-root',type=Path);args=ap.parse_args();result=json.loads(args.result.read_text());cs=json.loads((P/'fixtures.json').read_text())['cases'];raw=args.raw_root or R/'.porting-state/portable_julia/graphB+_raw_v1';receipts=[];guard_receipts=[];values=0;slots=0
 for model,m in result['models'].items():
  # Every accepted Julia process has five completed full cases. Failed processes have no trial5 files.
  candidates=list(raw.glob(f'{model if model!="cuda"else"native"}_*_trial5_n129_timed.raw.i64'))
  assert len(candidates)==(0 if model=='cuda'and not args.raw_root else 1),(model,candidates)
  stem='native'if not candidates else candidates[0].name.split('_trial5')[0]
  for trial in m['trials']:
   f=Path(trial['output_artifact']['saved_path']);assert sha(f)==trial['output_artifact']['sha256'];a=rd(f);values+=len(a);p=0
   for c in cs:
    expect=c['nindex']+[v*2 for v in c['nlist']]+c['eweight']+c['origID']+c['roots'];assert a[p:p+len(expect)]==expect;p+=len(expect)
   for c in cs:
    n=c['n'];e=len(c['nlist']);nt=a[p];p+=1;assert nt==c['iterations']+1;final=None
    for k in range(nt):
     length=a[p];p+=1;s=a[p:p+length];p+=length;it=c['iterations']-1 if k==0 else k-1;o=c['traces'][it];tag='timed'if k==0 else f'it{it}'
     if k==0:final=s
     if k==nt-1:assert s==final
     assert s[:n]==o['encoded_parent'];dfs=s[n:2*n];assert sorted(dfs)==list(range(0,2*n,2));assert dfs[o['root']]==0
     assert s[2*n:3*n]==o['labels'];assert s[3*n:4*n]==o['count'];assert s[4*n:5*n]==o['inCC']
     rows=[s[j:j+6]for j in range(5*n,5*n+6*e,6)];q=5*n+6*e;assert s[q]==o['hi'];w=s[q+1];assert [s[j:j+2]for j in range(q+2,q+2+2*w,2)]==o['pairs'];assert s[-2:]==[o['depth'],0]
     path=raw/f'{stem}_trial{trial["trial"]}_n{n}_{tag}.raw.i64';z=rd(path);assert len(z)==5*e
     physical=z[:e];infos=z[e:3*e];w1=z[3*e:4*e];w2=z[4*e:];assert sorted(zip(w1[:w],w2[:w]))==[tuple(t)for t in o['pairs']]
     perm=[]
     for u in range(n):
      lo,hi=c['nindex'][u:u+2];order=sorted(range(lo,hi),key=lambda j:physical[j]>>1);assert [physical[j]>>1 for j in order]==c['nlist'][lo:hi];perm+=order
      for j,actual in zip(range(lo,hi),order):
       v=c['nlist'][j];row=rows[j];tree=o['tree'][j];assert row==[physical[actual],infos[2*actual],infos[2*actual+1],o['inTree'][j],o['negCnt'][j],o['minus'][j]];assert row[0]==2*v+tree;assert row[2]&1==int(c['eweight'][j]==-1)
       if tree:
        child=v!=o['root'] and o['parent'][v]==u;node=v if child else u;assert row[1]==(dfs[node]if child else dfs[node]|1);assert row[2]==((dfs[node]+2*(o['subtree'][node]-1))|int(c['eweight'][j]==-1))
     assert sorted(perm)==list(range(e));slots+=e
     gp=Path(str(path)+'.guards.u8');lp=Path(str(path)+'.guards.json')
     if args.raw_root:
      gb=gp.read_bytes();layout=json.loads(lp.read_text());assert layout['model']==model;assert len(layout['buffers'])==(17 if model=='cuda'else 19);offset=0
      expected_names={'idx','list','weight'if model=='cuda'else'wei','orig','roots','parent','queue','label','tail','inCC','inTree','neg','ws1','ws2','info','minus','changed'}|({'hi','wsize'}if model!='cuda'else set())
      assert {b['name']for b in layout['buffers']}==expected_names
      for buf in layout['buffers']:
       assert buf['offset_bytes']==offset;size=buf['count_bytes'];assert size==64*buf['element_bytes'];assert buf['guard_elements_per_side']==32;assert gb[offset:offset+size]==bytes([165])*size;offset+=size
      assert offset==len(gb)
      guard_receipts.append(dict(model=model,trial=trial['trial'],n=n,stage=tag,path=str(gp.relative_to(R)),sha256=sha(gp),bytes=len(gb),layout_path=str(lp.relative_to(R)),layout_sha256=sha(lp),buffer_count=len(layout['buffers']),all_actual_guardbytes_equal_165=True))
     receipts.append(dict(model=model,trial=trial['trial'],n=n,stage=tag,path=str(path.relative_to(R)),bytes=path.stat().st_size,sha256=sha(path),physical_directed_slots=e,worklist_active_pairs=w,worklist_tail_values_retained=2*(e-w),full_CSR_bijection_verified=True,tree_ranges_and_original_signs_verified=True))
   assert p==len(a)
 out=dict(status='pass',source_result=str(args.result),source_result_sha256=sha(args.result),models=4,checked_trials=20,checked_primary_values=values,raw_files=len(receipts),raw_directed_slots_checked=slots,raw_bytes=sum(x['bytes']for x in receipts),raw_receipts=receipts,guard_evidence='All guard bytes checked by original capture process; primary retains badbytecount0, not individual raw guard bytes. CPU audit checks0 receipt and source coverage, no claim independently rereading missing raw guard bytes.',raw_order_contract='Complete physical CSR/einfo/worklist buffers retained. Per-node neighbor bijection checked, active worklist sorted with multiplicities; no literal queue/worklist order identity. Dead EdgeInfo/worklist-tail bytes retained and hashed but no mathematical value claimed.',timed_dfs_scope='DFS labels are supplemental tagged untimed replay labels; actual timed final node/edge/component arrays are retained and equal replay final. No timed intermediate label capture claim.')
 if args.raw_root:
  out['actual_guard_bytes_retained']=True;out['guard_evidence']='Actual per-buffer prefix/suffixUInt8bytes retained for every timed/replay stage/model/trial; source code performs actualDtoHcopies, CPU audit independently checks every retained byte165 and complete layout bounds.';out['guard_files']=len(guard_receipts);out['guard_bytes']=sum(x['bytes']for x in guard_receipts);out['guard_buffer_frames']=sum(x['buffer_count']for x in guard_receipts);out['guard_receipts']=guard_receipts
 args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k not in ['raw_receipts','guard_receipts']}))
if __name__=='__main__':main()

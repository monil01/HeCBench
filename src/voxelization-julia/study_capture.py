"""Retain all physical states; canonicalize actual complete keyed numerical values."""
from pathlib import Path
import sys,json,hashlib,shutil,time,os,struct
sys.path.insert(0,'/tmp/hecbench_resnet_numpy')
import numpy as np
R=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent
rawpath=Path(sys.argv[1]);outpath=Path(sys.argv[2]);model=sys.argv[3];trial=int(sys.argv[4]);store=R/'.porting-state/portable_julia/voxelization_raw_states';store.mkdir(exist_ok=True);tag=f'{model}_{os.getpid()}_{time.time_ns()}_trial{trial}';saved=store/(tag+'.bin');shutil.copyfile(rawpath,saved);h=hashlib.sha256()
with saved.open('rb')as f:
 for part in iter(lambda:f.read(1048576),b''):h.update(part)
index=store/(tag+'.json');base={'implementation_model':model,'trial':trial,'raw_path':str(saved),'raw_sha256':h.hexdigest(),'raw_bytes':saved.stat().st_size,'numpy_version':np.__version__,'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'raw_output_identity_claim':False,'complete_physical_state_retained':True};index.write_text(json.dumps(base,indent=2)+'\n');outpath.with_suffix(outpath.suffix+'.index').write_text(str(index)+'\n')
meta=json.loads((p/'independent_oracle.json').read_text());data=np.memmap(saved,dtype='<u4',mode='r');pos=0;G=64;GUARD=meta['guard'];EMPTY=0xffffffff;prefix=[4];suffix=[];reviews=[]
def take(n):
 global pos
 assert pos+n<=len(data),'short rawcapture';a=data[pos:pos+n];pos+=n;return a

def guarded(n):
 a=take(n+128);assert np.all(a[:64]==GUARD)and np.all(a[-64:]==GUARD),'rawguard';return a[64:-64],np.r_[a[:64],a[-64:]].tolist()
def h64(k):
 k^=k>>16;k=(k*0x85ebca6b)&0xffffffffffffffff;k^=k>>13;k=(k*0xc2b2ae35)&0xffffffffffffffff;k^=k>>16;return k
assert int(take(1)[0])==4
for c in meta['cases']:
 assert int(take(1)[0])==c['n'];n=c['n'];nv=c['valid_voxels'];points,pg=guarded(5*n);table,hg=guarded(1200000);counts,cg=guarded(160000);temp,tg=guarded(8000000);indices,ig=guarded(640000);real,rg=guarded(1);postcounts,pcg=guarded(160000);posttemp,ptg=guarded(8000000);features,fg=guarded(4000000)
 assert points.tolist()==c['input_bits'];prefix +=[n]+points.tolist();assert int(real[0])==nv;halfsize=2*n;keys=table[:halfsize];values=table[halfsize:2*halfsize];slots=np.flatnonzero(keys!=EMPTY);assert len(slots)==nv;assert np.all(table[2*halfsize:]==EMPTY);assert np.all(values[keys==EMPTY]==EMPTY);ids=values[slots];assert sorted(map(int,ids))==list(range(nv));mapping={int(keys[s]):(int(values[s]),int(s))for s in slots};assert len(mapping)==nv;assert set(mapping)=={g['key']for g in c['groups']}
 collision_probes=0
 for key,(vid,slot)in mapping.items():
  q=h64(key)%halfsize;steps=0
  while q!=slot:
   assert int(keys[q])!=EMPTY,'linearprobe keyunreachable';q=(q+1)%halfsize;steps+=1;assert steps<halfsize
  collision_probes+=steps
 assert np.all(counts[nv:]==0)and np.all(postcounts[nv:]==0);assert np.all(indices[4*nv:]==0);assert np.all(temp[50*nv:]==EMPTY);assert np.all(features.view('<u2')[5*nv:]==0)
 expectedtemp=temp.copy();expectedcounts=counts.copy();mean_by_id=np.zeros((nv,5),dtype='<u4');actualgroups=[];suffix.append(int(real[0]));order_report=[]
 for g in c['groups']:
  key=g['key'];vid,slot=mapping[key];cp=int(counts[vid]);after=int(postcounts[vid]);keep=min(cp,10);actual_index=indices[vid*4:vid*4+4].tolist();assert cp==g['count_pre']and after==g['count_post']and actual_index==g['index'];expectedcounts[vid]=keep;rawpoints=temp[vid*50:vid*50+keep*5].reshape(keep,5);assert sorted(rawpoints.tolist())==g['points'];assert np.all(temp[vid*50+keep*5:(vid+1)*50]==EMPTY)
  means=[]
  for f in range(5):
   acc=np.float32(rawpoints.view('<f4')[0,f])
   for j in range(1,keep):acc=np.float32(acc+rawpoints.view('<f4')[j,f])
   v=np.float32(acc/np.float32(keep));means.append(int(np.array([v],dtype='<f4').view('<u4')[0]))
  assert means==g['mean_bits'];actualmean=posttemp[vid*50:vid*50+5].tolist();actualhalf=features.view('<u2')[vid*5:vid*5+5].tolist();assert actualmean==means and actualhalf==g['half_bits'];assert np.array(means,dtype='<u4').view('<f4').astype('<f2').view('<u2').tolist()==actualhalf;expectedtemp[vid*50:vid*50+5]=means;mean_by_id[vid]=means
  suffix +=[key]+actual_index+[cp,after]+sum(sorted(rawpoints.tolist()),[])+actualmean+actualhalf
  order_report.append({'key':key,'physical_voxel_id':vid,'physical_hash_slot':slot,'retained_physical_point_order_rawbits':rawpoints.tolist(),'count_before':cp,'count_after':after})
 assert np.array_equal(postcounts,expectedcounts);assert np.array_equal(posttemp,expectedtemp),'sourceinplacefirstpointmeanonly'
 for guards in [pg,hg,cg,tg,ig,rg,pcg,ptg,fg]:suffix +=guards
 reviews.append({'case':c['case'],'n':n,'complete_voxels':nv,'raw_hash_ID_permutation_valid':True,'total_linear_probe_steps':collision_probes,'full_inactive_source_initializers_valid':True,'all_pre_post_scratch_values_source_transition_valid':True,'all_keyed_actual_points_counts_means_half_features_valid':True,'actual_mapping_and_point_order':order_report})
assert pos==len(data),'trailingrawcapture';canonical=np.array(prefix+suffix,dtype='<u4');gold=R/meta['golden_path'];assert hashlib.sha256(gold.read_bytes()).hexdigest()==meta['golden_sha256'];assert canonical.tobytes()==gold.read_bytes(),'canonicalactualvaluesdifferfromindependentoracle';canonical.tofile(outpath);base.update(all_checks_pass=True,canonical_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),canonical_count=len(canonical),input_prefix_count=len(prefix),cases=reviews,scope='Completekeyednumericsemanticagreement; everyactualphysicalhash/slotstate retained independentlyvalidated, no physicalorderidentityclaim.');index.write_text(json.dumps(base,indent=2)+'\n');print('VOXEL_RAW_INDEX '+str(index))

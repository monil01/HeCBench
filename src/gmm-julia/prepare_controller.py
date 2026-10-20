"""Reproducible source-preserving host launch adapter; no acceptance data."""
from pathlib import Path
import re,json,hashlib
B=Path(__file__).resolve().parent;src=B/'reference_history/cluster.cu';s=src.read_text()
# Each of the seven actually-used original GPU launch sites is replaced by the
# same dispatch ABI; all nonlaunch CPU controller statements remain verbatim.
names={'seed_clusters_kernel':1,'constants_kernel':2,'estep1':3,'estep2':4,'mstep_N':5,'mstep_means':6,'mstep_covariance2':7};counts={}
pattern=r'(seed_clusters_kernel|constants_kernel|estep1|estep2|mstep_N|mstep_means|mstep_covariance2)\s*<<<(.*?)>>>\s*\((.*?)\);'
def replace(m):
 name=m[1];counts[name]=counts.get(name,0)+1;stage=names[name]
 data='d_fcs_data_by_event'if stage==1 else'd_fcs_data_by_dimension'
 clusters='original_num_clusters'if stage==1 or (stage==2 and counts[name]==1)else'num_clusters'
 gx=clusters;gy='1'
 if stage==1:gx='1'
 if stage==3:gy='NUM_BLOCKS'
 if stage==4:gx='NUM_BLOCKS'
 if stage==6:gy='num_dimensions'
 if stage==7:gx='(num_clusters+NUM_CLUSTERS_PER_BLOCK-1)/NUM_CLUSTERS_PER_BLOCK';gy='num_dimensions*(num_dimensions+1)/2'
 like='d_likelihoods'if stage==4 else'nullptr'
 return f'gmm_dispatch({stage},{gx},{gy},256,{data},d_clusters,num_dimensions,{clusters},num_events,{like});'
s,n=re.subn(pattern,replace,s,flags=re.S);assert n==10 and set(counts)==set(names),(n,counts)
(B/'controller_adapted.inc').write_text(s)
x={'original_source':str(src.relative_to(B)),'original_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'output':'controller_adapted.inc','output_sha256':hashlib.sha256(s.encode()).hexdigest(),'replaced_launch_sites':counts,'total_sites':n,'nonlaunch_CPU_statements_unchanged':True,'actual_GPU_entrypoints':7,'compiled_but_not_selected_source_GPU_entrypoints':['mstep_covariance1']};(B/'controller_adapter_provenance.json').write_text(json.dumps(x,indent=2)+'\n');print(x)

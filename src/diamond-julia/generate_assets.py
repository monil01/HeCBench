#!/usr/bin/env python3
"""Independent explicit 51-state HMM oracle and fixed33 protein-masking fixtures."""
from pathlib import Path
import json,array,math,re,hashlib,struct,sys
P=Path(__file__).resolve().parent;ROOT=P.parent.parent;L=33;D=50;RP=.005;RE=.05;DECAY=.9;LAM=.267;MASK=23
f32=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
def blosum():
 f=ROOT/'src/diamond-sycl/src/blast/sm_blosum62.c';text=f.read_text();text=text.split('s_Blosum62PSM[25 * 25] = {',1)[1].split('};',1)[0];text=re.sub(r'/\*.*?\*/','',text,flags=re.S);scores=list(map(int,re.findall(r'-?\d+',text)));assert len(scores)==625;return scores

def likelihood():
 scores=blosum();return [math.exp(LAM*scores[i*25+j])if i<25 and j<25 else 1.0 for i in range(L)for j in range(L)]
def zero_padded_source64_prefix():
 scores=blosum();full=[math.exp(LAM*scores[i*25+j])if i<25 and j<25 else 0.0 for i in range(64)for j in range(64)];return full[:L*L]
def posterior(seq,mat):
 """Unscaled state-space forward/backward; no source scaling/ring-pointer algorithm."""
 n=len(seq);b2b=1-RP;f2f=1-RE;prior=[RP*(1-DECAY)/(1-DECAY**D)*DECAY**(d-1)for d in range(1,D+1)]
 alpha=[[1.]+[0.]*D]
 emissions=[]
 for k in range(n):
  emission=[mat[seq[k]*L+seq[k-d]]if k>=d else 0. for d in range(1,D+1)];emissions.append(emission);old=alpha[-1];new=[old[0]*b2b+math.fsum(old[1:])*RE]
  new +=[emission[d]*(old[0]*prior[d]+old[d+1]*f2f)for d in range(D)];alpha.append(new)
 terminal=[b2b]+[RE]*D;Z=math.fsum(a*b for a,b in zip(alpha[-1],terminal));assert Z>0 and math.isfinite(Z)
 beta=[None]*(n+1);beta[n]=terminal
 for k in range(n-1,-1,-1):
  nxt=beta[k+1];emission=emissions[k];beta[k]=[b2b*nxt[0]+math.fsum(prior[d]*emission[d]*nxt[d+1]for d in range(D))]+[RE*nxt[0]+f2f*emission[d]*nxt[d+1]for d in range(D)]
 assert abs(beta[0][0]/Z-1)<3e-13
 probs=[1-alpha[k+1][0]*beta[k+1][0]/Z for k in range(n)];assert all(-1e-12<=x<=1+1e-12 for x in probs)
 return probs

def sequences(n,case):
 seqs=[]
 for j in range(n):
  mode=(j+case)%5
  if mode==0:a=[0]*L
  elif mode==1:a=[17]*L
  elif mode==2:a=[(k%3)for k in range(L)]
  elif mode==3:a=[(k*7+j*11+case)%20 for k in range(L)]
  else:a=[(k*k*7+k*13+j*3+case)%20 for k in range(L)]
  seqs.extend(a)
 return seqs
if __name__=='__main__':
 assert sys.byteorder=='little';mat=likelihood();table=[MASK]*L;cases=[];prefix=[L,D,RP,RE,DECAY,0,0,.5,128,3];prefix +=[-777.]*16+mat+[-777.]*16;prefix +=[211.]*16+table+[211.]*16;tail=[];proball=[];errall=[];files={}
 for case,n in enumerate([1,128,257]):
  seq=sequences(n,case);probs=[];masked=[]
  for j in range(n):
   one=seq[j*L:(j+1)*L];pr=posterior(one,mat);probs.extend(pr);masked.extend([MASK if p>=.5 else x for x,p in zip(one,pr)])
  name=f'case{case}_input_u8.bin';(P/name).write_bytes(bytes(seq));prefix +=[case,n,(n+128)//128];prefix +=[211.]*16+seq+[211.]*16;tail +=[211.]*16+masked+[211.]*16;proball+=probs;errall +=[0]*n
  cases.append({'case':case,'sequences':n,'sequence_length':L,'blocks':(n+128)//128,'threads':128,'bytes':len(seq),'input':name,'masked_changes':sum(x!=y for x,y in zip(seq,masked)),'minimum_threshold_margin':min(abs(p-.5)for p in probs),'posterior_min':min(probs),'posterior_max':max(probs)})
 (P/'likelihood_f64.bin').write_bytes(array.array('d',mat).tobytes());(P/'mask_table_u8.bin').write_bytes(bytes(table));(P/'oracle_primary_f64.bin').write_bytes(array.array('d',prefix+tail).tobytes());(P/'oracle_posterior_f64.bin').write_bytes(array.array('d',proball).tobytes())
 h=f'#pragma once\nconstexpr int L=33,D=50,CASES=3,PREFIX={len(prefix)},COUNT={len(prefix+tail)},POSTERIOR_COUNT={len(proball)};\nconst int NS[3]={{1,128,257}};\n';(P/'study_params.h').write_text(h)
 jl=f'const L=33\nconst D=50\nconst PREFIX={len(prefix)}\nconst COUNT={len(prefix+tail)}\nconst POSTERIOR_COUNT={len(proball)}\nconst NS=(1,128,257)\n';(P/'study_params.jl').write_text(jl)
 names=[x['input']for x in cases]+['likelihood_f64.bin','mask_table_u8.bin','oracle_primary_f64.bin','oracle_posterior_f64.bin','study_params.h','study_params.jl'];files={name:{'bytes':(P/name).stat().st_size,'sha256':hashlib.sha256((P/name).read_bytes()).hexdigest()}for name in names}
 plan={'benchmark':'diamond','component':'Tantan repeat-HMM proteinmasking','cases':cases,'sequence_length':L,'max_repeat_offset':D,'repeat_prob':RP,'repeat_end_prob':RE,'offset_decay':DECAY,'lambda':LAM,'mask_char':MASK,'input_prefix_count':len(prefix),'artifact_count':len(prefix+tail),'posterior_count':len(proball),'independent_oracle':'Explicit51stateunscaledforward/backwardtransitionmodel, separatefromsource-scaledring/localarrayrecurrence; noindels original0gapparameters.','numerical_contract':{'primary_mask_bytes':'exact','actual_input_prefix':'exact','guards':'exact','supplemental_GPU_posterior_atol':3e-7,'supplemental_GPU_posterior_rtol':3e-7,'categorical_threshold':.5},'primary_native_reference_repaired':True,'native_terminal_guard_repair':'Skip sourceinactivek0lookup[50] from50-elementprivatearray; no posterior/mask arithmetic change, sourceobservedruntimefaultnotclaimed.','host_repair':'Proper33x33compactlikelihoodtablefromtrue25x25BLOSUM62/λ.267, notfirst1089flatentriesof64stridebuffer; feedoriginalunmaskedproteininput, notCPUalready-maskedcopy. Immutabledevicebodies unchanged.','ROI':'Namedresident3batchmaskingkernels/source128threadgrid +wholedevicecompletion; originalCLIalloc/H2D/D2H/free excluded all4.','source_default_parameters_unmeasured':['othersequence lengths','nonzero gap parameters','different minMaskProb','maxRepeatOffset<sequence_length','fullalignmentquality'],'files':files,'python':sys.version}
 (P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({'cases':cases,'prefix':len(prefix),'count':len(prefix+tail),'posterior_count':len(proball)}))

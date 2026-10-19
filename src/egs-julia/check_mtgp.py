"""Independent scalar MTGP recurrence and source seed/table preparation check."""
from pathlib import Path
import argparse,struct,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--capture-dir',type=Path,required=True);p.add_argument('--parameter-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();D=a.capture_dir;U=(1<<32)-1;N=101
u=lambda n:list(struct.unpack('<'+'I'*(n.stat().st_size//4),n.read_bytes()))
raw=a.parameter_file.read_bytes();param=list(struct.unpack_from('<15I',raw));assert param[0]==3217
_,bits,ident,M,sh1,sh2,*_=param;mask=param[14];rec=[];temp=[]
for i in range(16):
 r=t=0
 for k in range(4):
  if i&(1<<k):r^=param[6+k];t^=param[10+k]
 rec.append(r);temp.append((t>>9)|0x3f800001)
hidden=rec[4]^((rec[8]<<16)&U);tmp=(hidden+(hidden>>16))&U;tmp=(tmp+(tmp>>8))&U
state=[(tmp&255)*0x01010101]*N;state[0]=1325631759;state[1]=hidden
for i in range(1,N):state[i]^=(0x6c078965*(state[i-1]^(state[i-1]>>30))+i)&U
actualinit=u(D/'mtgp_initial_status.bin');actualparams=u(D/'mtgp_actual_params.bin');actualtables=u(D/'mtgp_actual_tables.bin');assert actualparams==[M,sh1,sh2,mask];assert actualtables==rec+temp;assert actualinit[:N]==state
assert M>0 and N-M>=32,'Parameter doesnot supportsource32laneparallel recurrence'

def generate():
 global state
 random=[0.]*96
 # Independent canonical scalar recurrence. WithN-M>=32, eachparallelbatch
 # depends onlyonoldfuturevalues oralreadycompletedpreviousbatches.
 for i in range(N):
  x=(state[i]&mask)^state[(i+1)%N];x^=(x<<sh1)&U;x^=state[(i+M)%N]>>sh2;x^=rec[x&15];x&=U
  if i<96:
   t=state[(i+M-1)%N];t^=t>>16;t^=t>>8;word=((x>>9)^temp[t&15])|1
   value=struct.unpack('<f',struct.pack('<I',word))[0]-1;assert 0<value<1;random[i]=value
  state[i]=x
 return random
array=generate();index=0;expected=[]
for draw in range(12):
 index+=1
 if index>=3:array=generate();index=0
 expected.extend(array[index*32:(index+1)*32])
got=list(struct.unpack('<384f',(D/'mtgp_uniforms.bin').read_bytes()));final=u(D/'mtgp_final_status.bin');assert len(got)==len(expected)
mismatches=[{'index':i,'got':g,'expected':e}for i,(g,e)in enumerate(zip(got,expected))if g!=e];statebad=[i for i,(g,e)in enumerate(zip(final[:N],state))if g!=e]
report={'kind':'native_readiness_diagnostic_not_benchmark_translation_or_performance','parameter_file_sha256':hashlib.sha256(raw).hexdigest(),'M':M,'sh1':sh1,'sh2':sh2,'mask':mask,'parameter_and_table_preparation_exact':True,'initial101active_state_words_match_source_seed_math':True,'draws':12,'lanes':32,'full_uniform_values':384,'uniform_mismatch_count':len(mismatches),'uniform_mismatches':mismatches[:20],'active_final_state_mismatch_count':len(statebad),'unused27wordpadding_unchanged':actualinit[N:]==final[N:],'all_pass':not mismatches and not statebad and actualinit[N:]==final[N:],'capture_hashes':{q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in D.glob('mtgp*.bin')},'scope':'FirstMTGPparameteronly; source32lanefullwarp/functionsunchanged. Scalarrecurrenceand384draws notstatisticalquality/fulltransportcertification; originaluninitializedunusedpadding retained/readback, no fabricatedzeropaddingclaim.'}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if report['all_pass']else 1)

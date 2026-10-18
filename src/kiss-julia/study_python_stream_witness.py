"""Arbitrary-precision selected-stream controls for the independently widened full oracle."""
from pathlib import Path
import json,struct,hashlib
p=Path(__file__).resolve().parent;mask=(1<<64)-1;mask32=(1<<32)-1;G=1<<20

def hash32(v):
    v^=v>>16;v=v*0x85ebca6b&mask32;v^=v>>13;v=v*0xc2b2ae35&mask32;return v^(v>>16)

def hash64(v):
    v^=v>>33;v=v*0xff51afd7ed558ccd&mask;v^=v>>33;v=v*0xc4ceb9fe1a85ec53&mask;return v^(v>>33)

raw=(p/'oracle_u64.bin').read_bytes();base=0;rows=[]
for n,seed in [(33,0),(G+33,42),(2*G+17,mask32)]:
    for tid in [0,1,16,17,32,255,256,G-1]:
        if tid>=min(n,G):continue
        local=hash32((seed+tid)&mask32);w=local or mask32
        for _ in range(8):x=hash64(w);y=hash64(x);z=hash64(y);w=hash64(z)
        assert y and z and w
        initial=[x,y,z,w];draws=[]
        for index in range(tid,n,G):
            x=(6906969069*x+1234567)&mask;y^=(y<<13)&mask;y^=y>>17;y^=(y<<43)&mask
            t=((z&63)*(1<<58)+w)&mask;wide=z+t;w=(z//64)+(wide//(1<<64));z=wide&mask;value=(x+y+z)&mask;expected=struct.unpack_from('<Q',raw,8*(base+index))[0];assert value==expected;draws.append({'output_index':index,'value':value,'carry':wide>>64})
        rows.append({'N':n,'seed':seed,'tid':tid,'local_seed':local,'fallback_used':local==0,'initial_state':initial,'draws':draws,'all_full_oracle_samples_exact':True})
    base+=n
proof={'method':__doc__,'all_selected_streams_exact':True,'samples':rows,'nonzero_constructor_proof':'Each xor-right shift is invertible; each Murmur multiplier is odd and hence invertible modulo2^64; composition maps0onlyto0. Constructor starts with nonzero seed or UInt64(2^32-1) fallback, so all32hashsteps produce nonzero and sourceassert conditions hold for every possibleUInt32local seed. Portable invalidstatecheck has no changed valid-domain outcome.','odd64_multipliers_modular_inverse':[{'multiplier':m,'inverse_mod2p64':pow(m,-1,1<<64)}for m in [0xff51afd7ed558ccd,0xc4ceb9fe1a85ec53]],'full_oracle_sha256':hashlib.sha256(raw).hexdigest(),'scope':'Finite bitstreams/recurrences/seed mapping, not statistical distribution quality or period certificate.'};(p/'python_stream_witness.json').write_text(json.dumps(proof,indent=2)+'\n');print({'streams':len(rows),'draws':sum(len(x['draws'])for x in rows),'all_exact':True})

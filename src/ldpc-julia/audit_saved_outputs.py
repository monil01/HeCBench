from pathlib import Path
import json,struct,math,hashlib,sys,re
p=Path(__file__).resolve().parent;m=json.loads((p/'fixture_description.json').read_text());load=lambda f:list(struct.unpack('<'+'d'*(Path(f).stat().st_size//8),Path(f).read_bytes()));prefix=load(p/'fixture.bin');gold=load(p/'oracle.bin');H=[list(map(int,prefix[1+r*24:1+(r+1)*24])) for r in range(12)];checks=[[c*96+(z+s)%96 for c,s in enumerate(row) if s>=0] for row in H for z in range(96)];known=[];off=697
for c in m['cases']:
 n=c['codewords']*2304;off+=1+n;known.append(list(map(int,prefix[off:off+n])));off+=n
assert off==len(prefix);rows=[]
for arg in sys.argv[1:]:
 d=json.loads((Path(arg)/'result.json').read_text())
 for model,v in d['models'].items():
  for t in v.get('trials',[]):
   f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert len(raw)==m['artifact_values']*8;assert raw[:len(prefix)*8]==(p/'fixture.bin').read_bytes();a=load(f)[len(prefix):];assert struct.pack('<'+'d'*len(a),*a)==(p/'oracle.bin').read_bytes();pos=guards=inactive=0;casechecks=[]
   for ci,c in enumerate(m['cases']):
    outs=[]
    for k,n in enumerate(c['output_sizes']):
     z=a[pos:pos+n+16];pos+=n+16;assert z[:8]==[-12345.]*8 and z[-8:]==[-12345.]*8;guards+=16;vals=z[8:-8];assert all(map(math.isfinite,vals));outs.append(vals)
     if k in [1,2]:
      for cw in range(c['codewords']):
       for bc in range(24):
        for br in range(12):
         if H[br][bc]<0:
          assert vals[cw*27648+bc*1152+br*96:cw*27648+bc*1152+(br+1)*96]==[0.]*96;inactive+=96
    hd=list(map(int,outs[3]));assert all(x in [0.,1.] for x in outs[3]);sy=[];er=[]
    for cw in range(c['codewords']):
     bits=hd[cw*2304:(cw+1)*2304];sy.append(sum(sum(bits[j] for j in row)%2 for row in checks));er.append(sum(x!=y for x,y in zip(bits,known[ci][cw*2304:(cw+1)*2304])))
    assert sy==c['syndrome_weights'] and er==c['decoded_bit_errors_vs_transmitted']
    if c['known_codeword_claim']:assert not any(sy) and not any(er)
    else:assert outs[0]==[0.]*(c['codewords']*2304) and hd==[1]*(c['codewords']*2304);assert sy==[384,384]
    for n,sentinel in [(c['codewords']*2304,-12345.),(12,90),(24,90),(84,1515870810),(288,1515870810)]:
     z=a[pos:pos+n+16];pos+=n+16;assert z[:8]==[sentinel]*8 and z[-8:]==[sentinel]*8;guards+=16
    casechecks.append(dict(case=ci,syndrome_weights=sy,decoded_bit_errors=er,physical_decoding_validation=c['known_codeword_claim'],zero_information_control_not_successful_decoding=not c['known_codeword_claim']))
   assert pos==len(gold);assert t['decoder_iterations']==10 and t['kernel_launches']==60
   rows.append(dict(model=model,trial=t['trial'],all_checks_pass=True,artifact=str(f),sha256=hashlib.sha256(raw).hexdigest(),exact_full_artifact_word_mismatches=0,guard_checks=guards,inactive_R_dt_positions_zero=inactive,cases=casechecks))
assert rows
print(json.dumps(dict(all_pass=all(x['all_checks_pass'] for x in rows),oracle=m['oracle'],graph=m['graph'],rows=rows,zero_LLR_scope='NativeexplicitAPP>0hard0else1. AllzeroLLRhasnochannelinformation, sourcealloneshardoutputsyndrome384, notvaliddecodingorframeworkaccuracyfailure; do notpoolitserrorswithclean/noisysixcodewordscientificcontrols'),indent=2))

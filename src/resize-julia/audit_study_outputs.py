"""Full saved typed image replay against independent rational oracle."""
from pathlib import Path
import json,hashlib,struct,sys,math
p=Path(__file__).resolve().parent;bundle=Path(sys.argv[1]);d=json.loads((bundle/'result.json').read_text());meta=json.loads((p/'fixture_description.json').read_text());prefix=(p/'fixture.bin').read_bytes();pre=struct.unpack('<28417d',prefix);gold=struct.unpack('<105568d',(p/'oracle.bin').read_bytes());inputs=[];pos=1
for c in meta['paths']:
 assert list(pre[pos:pos+8])==[c['bits'],int(c['mode']=='bilinear'),c['iw'],c['ih'],c['ow'],c['oh'],c['channels'],c['repeat']];pos+=8;inputs.append(pre[pos:pos+c['input_values']]);pos+=c['input_values']
assert pos==len(pre)
rows=[]
for model,m in d['models'].items():
 for trial in m['trials']:
  path=Path(trial['output_artifact']['saved_path']);raw=path.read_bytes();v=struct.unpack('<133985d',raw);assert raw[:len(prefix)]==prefix;assert tuple(v[len(pre):])==gold;pos=len(pre);paths=[]
  for c,h in zip(meta['paths'],inputs):
   no,ni,ng=c['output_values'],c['input_values'],c['guard_fields'];out=v[pos:pos+no];pos+=no;inp=v[pos:pos+ni];pos+=ni;guards=v[pos:pos+ng];pos+=ng;bits=c['bits'];mask=(1<<bits)-1;fmt={8:'B',16:'H',32:'I'}[bits];guard={8:0xa5,16:0xa55a,32:0xa55aa55a}[bits]
   assert all(math.isfinite(x) and int(x)==x and 0<=x<=mask for x in (*out,*inp,*guards));assert struct.pack('<'+str(ni)+fmt,*map(int,inp))==struct.pack('<'+str(ni)+fmt,*map(int,h));assert all(x==guard for x in guards)
   if c['mode']=='nearest':assert 0 in out and mask in out
   else:
    inplane=c['iw']*c['ih'];outplane=c['ow']*c['oh']
    for ch in range(c['channels']):assert min(h[ch*inplane:(ch+1)*inplane])<=min(out[ch*outplane:(ch+1)*outplane])<=max(out[ch*outplane:(ch+1)*outplane])<=max(h[ch*inplane:(ch+1)*inplane])
    if c['shape']==4:assert tuple(out)==tuple(h)
   paths.append(dict(path=c['path'],bits=bits,mode=c['mode'],exact_rational_oracle=True,all_actual_input_bytes_exact=True,all_guards_exact=True,full_unsigned_range_output_observed=c['mode']=='nearest',bilinear_identity_exact=c['mode']=='bilinear' and c['shape']==4,max_input=max(h),max_output=max(out)))
  assert pos==len(v);rows.append(dict(model=model,trial=trial['trial'],artifact_sha256=hashlib.sha256(raw).hexdigest(),values=len(v),paths=paths,pass_=True))
assert len(rows)==20
for row in rows:row['pass']=row.pop('pass_')
a=dict(all_pass=True,rows=rows,all20_bitwise_identical=len({x['artifact_sha256'] for x in rows})==1,conditional_reference_repaired=True,untouched_native_agreement_claim=False,oracle_domain='Exactdyadicratios, positiveawaytiesnearestcopy/fullUInt32range; boundedFloat32bilinear/UIntnarrowing,Fractionconvexoracle, nohighUInt32bilinearpromise.',float_exactness_proof='Maxbilinearpixel3520; coordinatefractionsdenominator<=8, productdenominator<=64. Expandedpolynomialintermediatenumeratorbound<2^21<2^24, soallFloat32arithmetic/contractionexactinthisdomainbeforefinalUInttruncation.',primary_acceptance='Exactalloutputs/inputecho/guardsvsdeclaredFractionoracleandbothcorrectedGPUrefs. Read-onlyconvexrange/identity/fullUIntcopychecksaresecondaryinvariants, notrelaxations.')
(p/'output_audit.json').write_text(json.dumps(a,indent=2)+'\n');print(json.dumps({'all_pass':True,'all20_bitwise_identical':a['all20_bitwise_identical']}))

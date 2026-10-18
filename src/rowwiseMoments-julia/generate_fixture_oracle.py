from pathlib import Path
from decimal import Decimal,localcontext
import struct,json,hashlib,sys
B=Path(__file__).resolve().parent
specs=[(1,1,'single',0),(17,3,'bounded',0),(255,3,'bounded',0),(256,3,'constant',0),(257,3,'bounded',0),(1024,3,'bounded',0),(4099,3,'offset',1024),(524288,4,'source_per_row_length_batch1',0)]
inputs=[];oracles=[];cases=[]
eps=struct.unpack('<f',struct.pack('<f',1e-6))[0]
with localcontext()as ctx:
 ctx.prec=80
 for n,rows,name,offset in specs:
  start=len(inputs);go=len(oracles)
  for row in range(rows):
   coeff=[(4096 if name in ['single','constant'] else offset*1024+((i*37+row*97)%1024)) for i in range(n)]
   vals=[x/1024 for x in coeff];inputs.extend(vals)
   s=sum(coeff);ss=sum(x*x for x in coeff);mean=Decimal(s)/Decimal(n*1024);var=Decimal(n*ss-s*s)/Decimal(n*n*1024*1024);rstd=1/(var+Decimal.from_float(eps)).sqrt();oracles.extend([float(mean),float(rstd)])
  cases.append(dict(n=n,rows=rows,name=name,input_offset=start,oracle_offset=go,eps=eps,exact_mean_constant_control=name in ['single','constant']))
 for name,fmt,vals in [('inputs_f32.bin','f',inputs),('oracle_f64.bin','d',oracles)]:
  (B/name).write_bytes(struct.pack('<'+fmt*len(vals),*vals))
 files={n:dict(bytes=(B/n).stat().st_size,sha256=hashlib.sha256((B/n).read_bytes()).hexdigest())for n in['inputs_f32.bin','oracle_f64.bin']}
 (B/'fixture_provenance.json').write_text(json.dumps(dict(cases=cases,files=files,oracle='ExactPythoninteger sums/sumsquares ofdyadicFloat32inputs, variance=(n*sum(x²)-sum(x)²)/n² thenDecimal80 reciprocal sqrt withactualFloat32eps; independentofWelford/reductiontree',dual_atol=1e-5,dual_rtol=2e-6,independent_mean_atol=5e-4,independent_mean_rtol=2e-6,independent_rstd_atol=1e-3,independent_rstd_rtol=2e-6,original_source_cli='1281281281284R100 has512rows/524288elements; namedlargestcasebatch1 retains4rows/sourceperrowlength, notfull1GiB128batchclaim',python=sys.version,generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')

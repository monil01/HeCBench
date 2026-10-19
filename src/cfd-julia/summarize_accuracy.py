"""Summarize physical final-state errors from already checked complete artifacts."""
from array import array
from pathlib import Path
import argparse,hashlib,json,math,sys
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert sys.byteorder=='little';root=Path(__file__).resolve().parent

def read(q):
 z=array('d')
 with q.open('rb')as f:z.fromfile(f,q.stat().st_size//8)
 return z
j=json.loads(a.result.read_text());gold=read(root/'oracle.bin');n=97152;physical=97046;g=132*192+116*n;oracle=gold[g:g+5*n];start=3213929+(149*192+2321)+116*n+1536+32
rows={};cache={}
for model,spec in j['models'].items():
 art=spec['trials'][0]['output_artifact'];file=Path(art['saved_path']);digest=hashlib.file_digest(file.open('rb'),'sha256').hexdigest();assert digest==art['sha256']
 if digest not in cache:
  z=read(file);u=z[start:start+5*n];old=z[start+5*n+64:start+10*n+64];flux=z[start+10*n+128:start+15*n+128];fields={};pressure_error=0
  for k,name in enumerate(['density','momentum_x','momentum_y','momentum_z','energy']):
   e=max(abs(u[k*n+i]-oracle[k*n+i])for i in range(physical));fields[name]={'max_abs_error_vs_F64_oracle':e,'final_flux_RMS_diagnostic':math.sqrt(sum(flux[k*n+i]**2 for i in range(physical))/physical),'last_full_iteration_change_RMS_diagnostic':math.sqrt(sum((u[k*n+i]-old[k*n+i])**2 for i in range(physical))/physical)}
  for i in range(physical):
   pg=(1.3999999761581421-1)*(u[4*n+i]-.5*sum(u[k*n+i]**2 for k in [1,2,3])/u[i]);pr=(1.3999999761581421-1)*(oracle[4*n+i]-.5*sum(oracle[k*n+i]**2 for k in [1,2,3])/oracle[i]);pressure_error=max(pressure_error,abs(pg-pr))
  cache[digest]={'final_conserved_fields':fields,'final_pressure_max_abs_error_vs_F64_oracle':pressure_error}
 rows[model]=cache[digest]
a.output.write_text(json.dumps({'run_id':j['run_id'],'physical_cells':physical,'full_iterations':2000,'all_five_acceptance':'output_audit.json; thissummarydescribesfirstcheckedrecord/model, deterministicidenticaltrialhashesverifiedthere','models':rows,'scope':'FinalsourcefluxRMS andlastiterationchange aredescriptivediagnostics, NOTresidualconvergencethreshold orphysicalmodelvalidation. No globalmassconservationclaim withlocalstepcoefficients.'},indent=2)+'\n')

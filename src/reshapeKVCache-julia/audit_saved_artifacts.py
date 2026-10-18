from pathlib import Path
import json,hashlib,sys,numpy as np
R=Path('/home/imo/HeCBench_julia');p=R/'src/reshapeKVCache-julia';S=R/'.porting-state/portable_julia';kind=sys.argv[1]if len(sys.argv)>1 else'final';rows=[];results={};Q=4194304;prefix=8389641
for B in (16,32):
 for x in (1,2,4,8):
  tag=f'B{B}_x{x}';lp=S/f'reshapeKVCache_{kind}_{tag}.log';rp=Path(json.loads(lp.read_text().splitlines()[-1])['result']);r=json.load(open(rp));assert r['reference_agreement'];results[tag]={'result':str(rp),'means_seconds':{m:v.get('mean_seconds')for m,v in r['models'].items()if v.get('mean_seconds')},'configuration_id':r['configuration_id']};e=np.fromfile(p/f'study_expected_{tag}_u8.bin',dtype=np.uint8)
  inp=np.concatenate([np.fromfile(p/'study_key_f32.bin',dtype=np.uint32),np.fromfile(p/'study_value_f32.bin',dtype=np.uint32),np.fromfile(p/'study_slot_i64.bin',dtype=np.int64).astype(np.uint32),np.fromfile(p/'study_scales_f32.bin',dtype=np.uint32),np.array([1024,32,128,B,x,1000,512],np.uint32)]);assert len(inp)==prefix
  for m,v in r['models'].items():
   if v.get('status')=='missing':continue
   assert v['status']=='native_check_pass';assert len(v['trials'])==5
   for t in v['trials']:
    art=t['output_artifact'];fn=Path(art.get('saved_path',art['path']));a=np.fromfile(fn,dtype=np.uint32);ok=len(a)==16778249 and np.array_equal(a[:prefix],inp) and np.array_equal(a[prefix:],e) and np.all((a[prefix:]&127)!=127);assert ok,(tag,m,fn)
    rows.append({'configuration':tag,'model':m,'trial':t['trial'],'artifact':str(fn),'sha256':hashlib.sha256(fn.read_bytes()).hexdigest(),'exact_prefix':True,'full_finite_FP8_codes_exact_oracle':True,'checked_outputs':8388608})
summary={'benchmark':'reshapeKVCache','numpy':np.__version__,'scope':'8originalFP32subcases; BF16hardwareguardexcluded','all_artifacts_pass':True,'artifact_count':len(rows),'results':results,'artifacts':rows};out=p/('independent_saved_output_audit.json'if kind=='final'else f'{kind}_saved_output_audit.json');out.write_text(json.dumps(summary,indent=2)+'\n');print(out,len(rows),'PASS')

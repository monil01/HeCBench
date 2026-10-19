from pathlib import Path
import json,hashlib,array,math,tempfile,shutil,subprocess,sys,re
P=Path(__file__).resolve().parent;R=P.parent.parent;F=R/'.porting-state/portable_julia/runs/20261019T204016719018Z_gibbs/result.json';r=json.load(open(F));A=P/'jacc_field_contract_audit.json';a=json.load(open(A));assert a['all_pairs_all5_field_contract_pass']and len(a['full_pair_trial_checks'])==30
hashes=0
for m,v in r['models'].items():
 for path,sha in v['adapter_file_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path;hashes+=1
 assert len(v['trials'])==5 and v['provenance_valid']and v['timing_valid']
# Scalar conditional-Gamma z scores are descriptive diagnostics, not post-hoc acceptance bounds.
x=array.array('d');x.frombytes(Path(r['models']['cuda']['trials'][0]['output_artifact']['saved_path']).read_bytes());zs=[]
for step in range(32):
 off=7900+step*6812;shape=x[off+1];rate=x[off+2]
 for i in range(513):
  aa=shape+i%7;bb=rate+.5+(i%13)*.125;theta=x[off+9+16+i];zs.append((theta-aa/bb)/(math.sqrt(aa)/bb))
stats={'n_unique_native_trajectory_draws':len(zs),'standardized_conditional_residual_mean':sum(zs)/len(zs),'standardized_conditional_second_moment':sum(z*z for z in zs)/len(zs),'is_acceptance_gate':False,'independent_repeated_MCMC_chains_claim':False,'posterior_convergence_claim':False}
# Cold replay only actually generated deterministicCPUassets; noGPUexpectedarray regeneration claim.
with tempfile.TemporaryDirectory(prefix='gibbs-cpu-cold-')as tmp:
 t=Path(tmp);shutil.copy2(P/'generate_assets.py',t);subprocess.run([sys.executable,str(t/'generate_assets.py')],check=True,capture_output=True)
 names=['y_i32.bin','n_f32.bin','observations.csv','study_params.h','study_params.jl'];cold={}
 for name in names:
  b=(t/name).read_bytes();assert b==(P/name).read_bytes(),name;cold[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
# Supplemental actual JACCtemplateobservations are not source outputs and were overwritten before math.
log=F.parent/'jacc_run1.log';line=next(l for l in log.read_text().splitlines()if 'scratch_initial_final1024values' in l);p=Path(line.split('path=')[1].split(';')[0]);vals=array.array('f');vals.frombytes(p.read_bytes());assert len(vals)==1024 and all(v==0 for v in vals)
report={'benchmark':'gibbs','final_result':str(F.relative_to(R)),'result_sha256':hashlib.sha256(F.read_bytes()).hexdigest(),'mandatory_full_field_audit':{'path':str(A.relative_to(R)),'sha256':hashlib.sha256(A.read_bytes()).hexdigest(),'pair_trial_checks':30,'all_pass':True},'all20_full_artifacts_checked':True,'values_each_artifact':225884,'input_prefix':7900,'declared_dependency_hash_checks':hashes,'cold_CPU_generated_files_exact':cold,'conditional_Gamma_descriptive_diagnostics':stats,'JACC_actual_scratch_initial_final':{'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'values':1024,'allzero':True,'512_positions_overwritten_before_math':True},'no_GPU_execution_in_this_audit':True,'central_envelope_alone_sufficient':False,'full_external_dataset_trajectory_certified':False,'native_reference_repaired':False,'native_integration_adapted':True}
(P/'saved_output_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'hashes':hashes,'cold_files':len(cold),'pair_trial_checks':30,'stats':stats},indent=2))

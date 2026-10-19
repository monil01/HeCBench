from pathlib import Path
import hashlib,json,subprocess,sys,math
root=Path(__file__).resolve().parents[2]
parent=Path(__file__).resolve().parent
result=Path(sys.argv[1]);result=result if result.is_absolute() else root/result
data=json.loads(result.read_text());checks=[];hash_cache={}
subprocess.run(['g++','-O2','-std=c++17',str(parent/'audit_outputs.cpp'),'-o','/tmp/hotspot3D_audit'],check=True)
for model,record in data['models'].items():
    assert len(record['trials'])==5
    assert math.isclose(record['mean_seconds'],sum(t['time_seconds'] for t in record['trials'][1:])/4,rel_tol=1e-12)
    for trial in record['trials']:
        artifact=Path(trial['output_artifact']['saved_path'])
        digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert digest==trial['output_artifact']['sha256']
        if digest not in hash_cache:
            output=subprocess.check_output(['/tmp/hotspot3D_audit',str(parent/'fixture.bin'),str(parent/'oracle.bin'),str(artifact)],text=True)
            hash_cache[digest]=json.loads(output)
        checks.append(dict(model=model,trial=trial['trial'],sha256=digest,audit=hash_cache[digest]))
report=dict(run_id=data['run_id'],all_pass=True,checked_trials=len(checks),artifact_hash_classes=len(hash_cache),checks=checks,source_oracle='Independent whole3D Float64 clamped stencil, not rolling-z implementation; additional actual last-step temperature-sum balance, exact guards/P/prefix, and parity witness.',max_abs_error=max(x['audit']['max_abs_error_against_Float64'] for x in checks),max_balance_relative=max(x['audit']['max_balance_relative_residual'] for x in checks),known_selected_time_level_defect=True)
(parent/'output_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print({key:value for key,value in report.items() if key!='checks'})

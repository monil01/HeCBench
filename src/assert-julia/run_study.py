import subprocess,sys,os,json,hashlib,pathlib
command=sys.argv[1:];out=pathlib.Path(os.environ.get('STUDY_OUTPUT_DIR','.'));out.mkdir(parents=True,exist_ok=True)
observations=[]
for trial in range(1,6):
 child=subprocess.run(command+['--negative'],capture_output=True,text=True)
 log=out/f'assert_negative_{trial}.log';log.write_text(child.stdout+child.stderr)
 statuses=[int(s.split()[1]) for s in child.stdout.splitlines() if s.startswith('ASSERT_STATUS ')]
 if child.returncode or statuses!=[710]:
  print(child.stdout,child.stderr,file=sys.stderr);raise SystemExit('actual isolated assertion status mismatch')
 observations.append(dict(trial=trial,process_returncode=child.returncode,observed_status=710,log=str(log.resolve()),sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
child=subprocess.run(command,capture_output=True,text=True)
(out/'positive_process.log').write_text(child.stdout+child.stderr)
if child.returncode:print(child.stdout,child.stderr,file=sys.stderr);raise SystemExit(child.returncode)
rows=[json.loads(s[len('PERF_TRIAL '):]) for s in child.stdout.splitlines() if s.startswith('PERF_TRIAL ')]
assert len(rows)==5
for row,obs in zip(rows,observations):
 row.update(correct=True,checked_values=0,source_empty_kernel=False,source_no_observable_numerical_outputs=True,no_output_reason='source_device_assertion_only',device_assertions_present=True,successful_assertion_kernels_checked=True,expected_device_assertion_failure_checked=True,expected_device_assertion_status=710,observed_device_assertion_status=710,assertion_failure_context_isolated=True,device_completion_checked=True,execution_checked=True,numerical_agreement_kind='no_numerical_outputs_execution_only',max_abs_error=None,max_rel_error=None,timing_region='sum_two_original_positive_assertion_kernels_1000x256_launch_device_completion',jit_excluded=True,internal_repeats=1,negative_diagnostic=obs)
 print('STUDY_TRIAL '+json.dumps(row),flush=True)

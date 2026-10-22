from pathlib import Path
import json,hashlib,subprocess,datetime,shutil,fcntl
P=Path('/tmp/roofline-completion-20261021');D=P/'sparkler071_recovery';R=Path('/home/imo/HeCBench_julia');receipt=P/'root071_execution_receipt.json';assert not receipt.exists()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(D/'DELIVERY.json')=='91c56f9a007250bd3922c156ac2d6acde1902b4a58addc622bd27dfb9ebc25d1';assert sha(P/'sparkler071_independent_review/DELIVERY.json')=='68321049b5785189cf1ce883af92f9d069759bef66e77a819f8522e4666be317'
delivery=json.loads((D/'DELIVERY.json').read_text())
for row in delivery['files']:
 p=Path(row['path']);assert sha(p)==row['sha256']and p.stat().st_size==row['bytes'],str(p)
for row in json.loads((D/'tool_bindings.json').read_text())['files']:assert sha(row['path'])==row['sha256']
with Path('/tmp/hecbench-roofline-metadata.lock').open()as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 for row in json.loads((D/'plans.json').read_text())['rows']:
  p=Path(row['plan']['path']);assert sha(p)==row['plan']['sha256'];plan=json.loads(p.read_text());assert plan['model']==row['model']and plan['benchmark']=='sparkler';assert sha(plan['reference_result'])==plan['reference_result_sha256']
  ledger=json.loads((R/'ontology/studies/v100_roofline_537/benchmarks/sparkler.json').read_text());ct=ledger['accepted_named_contract'];assert ct['configuration_id']==plan['configuration_id'];assert ct['raw_config_sha256']==plan['raw_config_sha256']
  for path,h in plan['execution_bindings'].items():assert sha(path)==h,path
command=json.loads((D/'future_command.json').read_text());assert command['maximum_collector_invocations']==1;assert '--retain-auxiliary'not in command['argv'];minimum=command['prospective_peak_free_requirement_bytes'];free=shutil.disk_usage('/tmp').free
state={'status':'root_source_review_and_independent_review_passed','time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'free_bytes':free,'required_free_bytes':minimum,'maximum_collector_invocations':1,'GPU':'6only','qualification':'Explicit2025measurementstratum; MPIprivate-path hypothesis unvalidated until actualchecks. SameMPI4/comms/vendor/kernel/check/timer; exactUUIDguardoutsideROI. All5checksmandatory; partialsamplesnotwholeROI. Canonicalnewpostmanifestcleanup, no laterbinaryrecertification.'};receipt.write_text(json.dumps(state,indent=2)+'\n')
if free<minimum:state['status']='storage_hold_zero_invocations';receipt.write_text(json.dumps(state,indent=2)+'\n');raise SystemExit(0)
with(P/'root071_collector.log').open('xb')as out:r=subprocess.run(command['argv'],cwd=R,stdout=out,stderr=subprocess.STDOUT)
state.update(status='finite_one_call_returned_root_scientific_review_required',collector_returncode=r.returncode,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());receipt.write_text(json.dumps(state,indent=2)+'\n')

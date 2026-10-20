#!/usr/bin/env python3
"""Owned protocol launcher: new captures on /tmp; canonical symlink, unchanged verifier."""
import argparse,pathlib,subprocess,os,json,hashlib
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def main():
 a=argparse.ArgumentParser();a.add_argument('--config',type=pathlib.Path,required=True);z=a.parse_args();c=z.config.resolve();assert json.load(open(c))['benchmark']=='slu';state=pathlib.Path('/tmp/hecbench-stencil-ports-state');state.mkdir(exist_ok=True);env=os.environ.copy();env['STATE']=str(state);tool=R/'tools/verify_portable_julia.py';p=subprocess.run(['python3',str(tool),'slu','--config',str(c)],env=env,text=True,capture_output=True,cwd=R);print(p.stdout,end='');print(p.stderr,end='')
 for line in p.stdout.splitlines():
  try:r=json.loads(line)
  except ValueError:continue
  if 'result'not in r:continue
  path=pathlib.Path(r['result']);out=path.parent;link=R/'.porting-state/portable_julia/runs'/out.name
  if link.exists()or link.is_symlink():assert link.resolve()==out.resolve()
  else:link.symlink_to(out,target_is_directory=True)
  record=dict(benchmark='slu',configuration_id=json.load(open(c))['configuration_id'],run_id=out.name,actual_state_root=str(state),canonical_run_alias=str(link),actual_result=str(path),result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),verifier_sha256=hashlib.sha256(tool.read_bytes()).hexdigest(),launcher_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),GPU_lock='unchangedcentral/tmp/hecbench-julia-study-gpu.lock',reason='Prospectiveownnewartifactsonlocalfilesystemavoidnearlyfullhome; nooldcapturerelocation/deletion/noacceptedreruns. STATEexistingverifieroption, noexecution/verdict/tolerancechange.',ephemeral_storage_caveat='Local/tmpcapturepathsrequirepreservationinresearchartifactbeforemachinecleanup/reboot; notsourceGitassets. Logicalcontentandcanonicalreceiptaliasesretained.',global_database_note='ThisrunSQLite/CSV areunderexplicitalternateSTATE; canonicalrunaliasallowsresult-discovery. Noattempttomutate/mergeglobalcoordinatordatabase.')
  folder=P/'storage_provenance';folder.mkdir(exist_ok=True);(folder/(out.name+'.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'canonical_result_alias':str(link/'result.json')}))
 return p.returncode
if __name__=='__main__':raise SystemExit(main())

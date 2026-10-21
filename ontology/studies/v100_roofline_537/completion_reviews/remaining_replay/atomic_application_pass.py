#!/usr/bin/env python3
"""Host-only per-application-pass output and exact property artifact receipt.
Runs the original complete child. Does not change computation, reset order,
launches, CPU participant, synchronization or checks. NCU owns app replay.
"""
import hashlib,json,os,pathlib,struct,subprocess,sys,uuid
P=pathlib.Path
def dump(p,a):p.write_text(json.dumps(a,indent=2)+'\n')
def main():
 if len(sys.argv)<2:raise SystemExit('Original child argv required')
 root=P(os.environ['STUDY_OUTPUT_DIR']).resolve();passes=root/'application_passes';passes.mkdir(parents=True,exist_ok=True)
 directory=passes/('pass_'+str(os.getpid())+'_'+uuid.uuid4().hex);directory.mkdir()
 env=os.environ.copy();env['STUDY_OUTPUT_DIR']=str(directory)
 command=sys.argv[1:];receipt={'wrapper_pid':os.getpid(),'child_command':command,'STUDY_OUTPUT_DIR':str(directory),'replay_pass_index':None,'qualification':'Unique process receipt; actualNCUpassmatching andhardwarecountercontext remainprofilerprovenance, notinferredfromPID.','status':'started'}
 dump(directory/'pass_receipt.json',receipt);rows=[]
 with (directory/'program.log').open('w')as log:
  child=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  receipt['child_pid']=child.pid;dump(directory/'pass_receipt.json',receipt)
  for line in child.stdout:
   log.write(line);log.flush();sys.stdout.write(line);sys.stdout.flush()
   if line.startswith('STUDY_TRIAL '):rows.append(json.loads(line[len('STUDY_TRIAL '):]))
  code=child.wait()
 receipt['child_returncode']=code;receipt['STUDY_TRIAL_rows']=rows;receipt['status']='child_completed'
 # Preserve exact child failure, including signal-bearing returncode in receipt.
 if code:
  dump(directory/'pass_receipt.json',receipt);return code
 try:
  if [x['trial']for x in rows]!=[1,2,3,4,5]:raise ValueError('Require all5 original checked trials eachpass')
  initial=[0]*10;initial[7]=initial[9]=255
  expected=[8,8,1000]+initial+[160000,15,0,0,0,1,65535,255]+[1,1]+[1515870810]*32
  artifacts=[]
  for row in rows:
   if row.get('correct')is not True or row.get('checked_values')!=42 or row.get('max_abs_error')!=0:raise ValueError('Original propertycheck failed')
   raw=row.get('raw_state')
   if not isinstance(raw,list)or len(raw)!=10:raise ValueError('Full rawproperty state log required')
   if [raw[i]for i in [0,2,3,4,5,7,8,9]]!=[160000,15,0,0,0,1,65535,255]or not(0<=raw[1]<16 and 0<=raw[6]<16):raise ValueError('Raw atomic state outside original property contract')
   artifact=row['output_artifact'];path=P(artifact['path']).resolve()
   if not path.is_relative_to(directory)or artifact.get('dtype')!='int32'or artifact.get('count')!=55:raise ValueError('Private complete artifact required')
   data=path.read_bytes()
   if len(data)!=220 or list(struct.unpack('<55i',data))!=expected:raise ValueError('Complete original property artifact/guards mismatch')
   artifacts.append({'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'trial':row['trial']})
  receipt.update(status='all5_property_checks_completed',artifacts=artifacts,property_scores_and_all32_guards_checked=True,source_warmup_plus5_scientific_launches_expected=6,numerical_agreement_claim=False)
 except Exception as error:
  receipt.update(status='per_pass_check_failed',error=repr(error));dump(directory/'pass_receipt.json',receipt);raise
 dump(directory/'pass_receipt.json',receipt);print('ATOMIC_APPLICATION_PASS '+json.dumps({'receipt':str(directory/'pass_receipt.json'),'all5_checked':True,'child_returncode':0}),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())

"""Independent coordinate permutation and dynamic queue completion-state witness."""
import array
import json
import pathlib
import struct
import sys
A,B=13,17
expected=array.array('I')
for payload in (1,32,65,129,256):
 for output_col in range(B):
  for output_row in range(A):
   for lane in range(payload):
    source=(output_row*B+output_col)*payload+lane
    value=((73*source+19)%65521-32760)/64
    expected.append(struct.unpack('<I',struct.pack('<f',value))[0])
 for index in range(A*B):
  flag=0 if index==A*B-1 or (index*A)%(A*B-1)==index else 1
  expected.append(flag)
 expected.append(A*B-1+16)
raw=expected.tobytes();assert len(expected)==107853
result=json.loads(pathlib.Path(sys.argv[1]).read_text());rows=[]
for model,record in result['models'].items():
 for trial in record['trials']:
  assert pathlib.Path(trial['output_artifact']['saved_path']).read_bytes()==raw
  assert trial['all_internal_repeats_checked'] is True
  rows.append({'model':model,'trial':trial['trial'],'all_payload_bits_completion_flags_terminal_head_match':True,'all100internal_repeats_local_checks_pass':True})
assert len(rows)==20
print(json.dumps({'benchmark':'asta','trials':rows,'oracle':'Directcoordinatepermutation,notGPUcyclefollowing;flagsfromfixedpointpredicate,headfrommactiveindicesplusonefinalfetchper16blocks','retained_artifact_scope':'Finalinternalrepeatperpayloadonly;all100internalrepeatfullCPUchecksperformed,butallotherpayloadoutputsnotretained','scope':'A13,B17,payload<=4*64slots,nolarge-caseormemorymodeluniversalityclaim'},indent=2))

"""Independent dyadic vector-add oracle for all mapped-memory result artifacts."""
import array
import json
import pathlib
import sys
N=1048579
expected=array.array('f',((2*((37*i+11)%1024-512)+((53*i+7)%2048-1024))/8192 for i in range(N))).tobytes()*2
result=json.loads(pathlib.Path(sys.argv[1]).read_text())
rows=[]
for model,record in result['models'].items():
    for trial in record['trials']:
        assert trial['mapped_host_verified'] is True
        data=pathlib.Path(trial['output_artifact']['saved_path']).read_bytes()
        assert data==expected
        rows.append({'model':model,'trial':trial['trial'],'all2097158_float32_values_match_integer_dyadic_oracle':True})
assert len(rows)==20
print(json.dumps({'benchmark':'zerocopy','trials':rows,'oracle':'Independent integer numerator exact dyadic addition; noGPUkernelreuse','memory_placement_evidence':'EachmodelassertsCUDApointertypeHostaftercudaHostGetDevicePointeronbothalloc/registerpaths;actualhostoutputreadwithoutD2Hcopy','scope':'NamedN1048579;notfullsizesweeporNUMA/interconnectcharacterization'},indent=2))

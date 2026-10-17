"""Independent closed-form memtest test5 full-buffer oracle. Usage: python3 scientific_memory_witness.py RESULT_JSON."""
import pathlib,json,array,sys
r=pathlib.Path(__file__).resolve().parents[2];b=pathlib.Path(__file__).resolve().parent;result=json.loads(pathlib.Path(sys.argv[1]).read_text());words=524288;per=262144;half=131072;expected=array.array('I')
for i in range(words):
 block,pos=divmod(i,per);src=(pos-8)%half if pos<half else pos-half;absolute=block*per+src;pattern=1<<((absolute//16)%32);expected.append(pattern if absolute%4<2 else pattern^0xffffffff)
expected.append(0);raw=expected.tobytes();rows=[]
for m,v in result['models'].items():
 for t in v['trials']:
  got=pathlib.Path(t['output_artifact']['saved_path']).read_bytes();assert got==raw;rows.append({'model':m,'trial':t['trial'],'all524288words_and_zero_error_count_match_independent_closed_form':True})
assert len(rows)==20
e={'benchmark':'memtest','independent_method':'Closedform sourceindex of two-phase blockmove; walkingbit mask derived from original64bytechunk index, noGPUkernelreuse','trials':rows,'fault_witness':'Eachmodeluntimedcheckrequiresinjectedfirstwordfaultcount1,actualdeviceaddress,expectedpairedword,currentandsecondread;native/CUDAjl/KA/JACClocalasserts','scope':'Namedtest5only;notphysicalmemoryhealthcertification'}
print(json.dumps(e,indent=2))

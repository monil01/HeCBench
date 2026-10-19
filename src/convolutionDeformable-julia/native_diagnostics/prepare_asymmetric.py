from pathlib import Path
import importlib.util,array,json,sys,hashlib
P=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1]);D.mkdir(parents=True,exist_ok=True);s=importlib.util.spec_from_file_location('shape_oracle',P/'generate_assets.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
g=o.dims(dict(id=4,sample=0,C=2,G=1,H=5,W=6,KH=3,KW=3,PH=1,PW=0,SH=1,SW=1,DH=1,DW=1));inputs=o.data(g);outputs=o.oracle(g,inputs)
for name in ['image','offset','mask','colgrad']:(D/f'case4_{name}_f32.bin').write_bytes(array.array('f',inputs[name]).tobytes())
a=[]
for name in ['columns','grad_image','grad_offset','grad_mask']:a+=o.wrap(outputs[name])
(D/'asymmetric_expected_f64.bin').write_bytes(array.array('d',a).tobytes());(D/'fixture.json').write_text(json.dumps({'geometry':g,'oracle_values':len(a),'expected_sha256':hashlib.sha256((D/'asymmetric_expected_f64.bin').read_bytes()).hexdigest(),'source_suspicion':'Originalcol2imlauncherpassespad_h,pad_h todevicekernel; asymmetricPH1/PW0 mayproducewronginputgradient; originalsourceunchanged. Actualdiagnosticrequired before observedfailureclaim.'},indent=2)+'\n')

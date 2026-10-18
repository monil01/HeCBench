
from pathlib import Path
import numpy as np,subprocess,json,hashlib
p=Path(__file__).resolve().parent
tmp=Path("/tmp/quantAQLM_source_codes_generator")
subprocess.run(["g++","-O2",str(p/"generate_codes.cpp"),"-o",str(tmp)],check=True)
subprocess.run([str(tmp),str(p/"study_codes1_u16.bin")],check=True)
codes=np.fromfile(p/"study_codes1_u16.bin",dtype="<u2").reshape(12288,512)
metadata=[]
for case,(m,k,b)in enumerate([(12288,4096,4),(96,512,4)],1):
 if case==1:
  code=codes
  inp=(np.arange(k*b,dtype="f4")/np.float32(k*b)).astype("<f2")
  book=(np.arange(3*65536*8,dtype="f4")/np.float32(3*65536*8)).astype("<f2")
 else:
  rr=np.arange(m,dtype="u4")[:,None];cc=np.arange(k//8,dtype="u4")[None,:]
  code=((rr*7919+cc*65521+32768)%65536).astype("<u2")
  inp=(((np.arange(k*b,dtype="i4")*17)%257-128)/256).astype("<f2")
  idx=np.arange(3*65536*8,dtype="i4");book=(((idx*13)%257-128)/512).astype("<f2")
 data=np.r_[code.reshape(-1),inp.view("<u2"),book.view("<u2")].astype("<u2")
 out=p/f"study_case{case}_u16.bin";out.write_bytes(data.tobytes())
 metadata.append({"case":case,"m":m,"k":k,"b":b,"partitions":[m//3,2*m//3,m],"source_codebook_entries":65536,"code_count":int(code.size),"input_count":int(inp.size),"book_half_count":int(book.size),"fixture":out.name,"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"bytes":out.stat().st_size,"CPU_generated":True,"scope":"OriginalsourceFloat32division→Float16rounding+glibc_srand123randcodes"if case==1 else"Deterministic signed dyadic input/codebook probes allthreebookpartitions/full16bitcodeindices"})
(p/"fixture_provenance.json").write_text(json.dumps({"NumPy":np.__version__,"source":"src/quantAQLM-cuda/main.cu","source_sha256":hashlib.sha256((p.parent/"quantAQLM-cuda/main.cu").read_bytes()).hexdigest(),"cases":metadata,"generator":"generate_fixtures.py+generate_codes.cpp; noGPUfixtures"},indent=2)+"\n")

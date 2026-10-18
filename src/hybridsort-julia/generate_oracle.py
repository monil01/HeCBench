from pathlib import Path
import struct,json,hashlib,sys
P=Path(__file__).resolve().parent;N=16384
def main():
 files={}
 for case in (1,2):
  m=8192 if case==1 else 2048
  v=[(0.25 if case==1 else 0.125)+((i*4051+i//m*37)%m)/m for i in range(N)]
  for name,a in [(f"study_case{case}_f32.bin",v),(f"oracle_case{case}_f32.bin",sorted(v))]:
   b=struct.pack("<"+"f"*N,*a);(P/name).write_bytes(b);files[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 (P/"fixture_provenance.json").write_text(json.dumps(dict(n=N,cases=2,files=files,oracle="Independent Python comparison sort; finite positive dyadic values, duplicate multiplicities2 and8; permutation fixture; original algorithm positive zero-padding assumption scoped.",generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python=sys.version),indent=2)+"\n")
if __name__=="__main__":main()

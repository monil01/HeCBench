"""Independent exact-integer fixtures; no fast-division magic recurrence."""
from pathlib import Path
import struct,json,hashlib
B=Path(__file__).resolve().parent
DIVISORS=[s*d for d in [1,2,3,5,7,11,16,31,127,255,1024,65535,65537,2147483647] for s in [1,-1]]
def quotient(n,d):return (abs(n)//abs(d))*(-1 if (n<0)!=(d<0) else 1)
if __name__=='__main__':
 xs=list(range(-2048,2049))+[-2147483647,-2147483646,-1073741825,-65537,65537,1073741825,2147483646,2147483647]
 data=struct.pack('<'+'i'*len(xs),*xs);(B/'input_i32.bin').write_bytes(data)
 vals=[]
 for d in DIVISORS:
  qs=[quotient(x,d) for x in xs];rs=[x-q*d for x,q in zip(xs,qs)];vals+=qs+rs
 (B/'oracle_i32.bin').write_bytes(struct.pack('<'+'i'*len(vals),*vals))
 (B/'fixture_description.json').write_text(json.dumps({'n':len(xs),'divisors':DIVISORS,'input_sha256':hashlib.sha256(data).hexdigest(),'independent_oracle':'Python arbitrary-precision abs // and sign for truncation toward zero; remainder n-q*d','excluded':'Zero divisor, INT_MIN and INT_MIN/-1 overflow; all results fitInt32. Not exhaustive32-bit certification.','scoped_region':'Observable quotient+remainder capture maps for plain and unchanged fastdiv helper; not original unobservable throughput/latency CLI.','capture':'Actual input+2guards;28divisors*2methods*[guard,actual divisor argument,allquotients,allremainders,guard];original check kernel4words+2guards perdivisor outside timing','repetitions':5},indent=2)+'\n')

from pathlib import Path
import struct,math,json
p=Path(__file__).resolve().parent
f=list(struct.unpack('<%sf'%(len((p/'fixtures_f32.bin').read_bytes())//4),(p/'fixtures_f32.bin').read_bytes()));o=list(struct.unpack('<%sd'%(len((p/'oracle_f64.bin').read_bytes())//8),(p/'oracle_f64.bin').read_bytes()));i=oi=0;checks=[]
while i<len(f):
 B,C,T,mode=map(int,f[i:i+4]);eps=f[i+4];i+=5;n=B*C*T;w=f[i:i+C*T];i+=C*T;k=f[i:i+n];i+=n;g=f[i:i+n];i+=n;x=o[oi:oi+n];gw=o[oi+n:oi+2*n];gk=o[oi+2*n:oi+3*n];oi+=3*n
 reduced=[math.fsum(gw[(b*C+c)*T+t]for b in range(B))for c in range(C)for t in range(T)]
 lhs=math.fsum(g[j]*(x[j]-eps)for j in range(n));rhs_k=math.fsum(k[j]*gk[j]for j in range(n));rhs_w=math.fsum(w[j]*reduced[j]for j in range(C*T))
 checks.append({'B':B,'C':C,'T':T,'dot_g_forward_without_eps':lhs,'dot_k_gradK':rhs_k,'dot_w_batchsummed_gradW':rhs_w,'exact_adjoint_identity':lhs==rhs_k==rhs_w,'all_batchsummed_gradW_values':reduced})
assert all(c['exact_adjoint_identity']for c in checks)
(p/'derivative_adjoint_witness.json').write_text(json.dumps({'method':'Independent mathematical bilinear adjoint identity with exact dyadic values and math.fsum. Full native perbatch gradW oracle batchsum is host-only, never claimed timed GPU reduction.','checks':checks},indent=2)+'\n')

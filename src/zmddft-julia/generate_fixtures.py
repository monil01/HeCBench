"""Cold regeneration: NumPy2.5.4, deterministic full256³complex inputs/oracles."""
from pathlib import Path
import numpy as np,re,json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[1];N=256;shape=(N,N,N);source=R/'src/zmddft-cuda/main.cu';s=source.read_text();text=re.search(r'D3\[512\]\s*=\s*\{(.*?)\};',s,re.S).group(1);tw=np.array([float(t.strip().strip('()'))for t in text.split(',')if t.strip()],dtype='<f8');assert len(tw)==512;(P/'twiddles_f64.bin').write_bytes(tw.tobytes())
files=[]
def save(name,a):
 f=P/name;np.ascontiguousarray(a,dtype='<c16'if np.iscomplexobj(a)else'<f8').tofile(f);files.append(dict(path=name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
i=np.arange(N,dtype='i8')[:,None,None];j=np.arange(N,dtype='i8')[None,:,None];k=np.arange(N,dtype='i8')[None,None,:]
x=(((i*17+j*13+k*7+(i*j+k*j)%251)%257-128)/128).astype('c16');x+=1j*((i*11-j*3+k*19+(i*k)%127)%251-125)/256;save('input_complex_f64.bin',x);input_energy=float(np.sum(np.abs(x)**2));input_sum=complex(x.sum())
a=x
for stage in range(3):a=np.ascontiguousarray(np.moveaxis(np.fft.fft(a,axis=-1),-1,0));save(f'oracle_stage{stage}_f64.bin',a)
fftn=np.fft.fftn(x);final_difference=float(np.max(np.abs(a-fftn)));assert final_difference<1e-7
# Independent analytic impulse: location(3,5,7), amplitude .75-.5i.
phase=lambda r:np.exp(-2j*np.pi*r*np.arange(256)/256);amp=.75-.5j
p=np.zeros(shape,dtype='c16');p[:,3,5]=amp*phase(7);save('impulse_stage0_f64.bin',p)
p.fill(0);p[:,:,3]=amp*phase(5)[:,None]*phase(7)[None,:];save('impulse_stage1_f64.bin',p)
p=amp*phase(3)[:,None,None]*phase(5)[None,:,None]*phase(7)[None,None,:];save('impulse_stage2_f64.bin',p)
files.append(dict(path='twiddles_f64.bin',bytes=4096,sha256=hashlib.sha256((P/'twiddles_f64.bin').read_bytes()).hexdigest()))
plan=dict(shape=list(shape),complex_values=256**3,doubles=2*256**3,repeats=100,threads=64,blocks=16384,twiddle_doubles=512,input_prefix_count=2*256**3+512+8,output_values=3*2*256**3,guard_values=4*16,artifact_count=4*2*256**3+512+8+4*16,atol=1e-9,rtol=1e-12,input_formula_real='((17i+13j+7k+(ij+kj)%251)%257-128)/128',input_formula_imag='((11i-3j+19k+(ik)%127)%251-125)/256',input_energy=input_energy,input_sum=[input_sum.real,input_sum.imag],source_input_rng_not_reproduced=True,source_full_shape_and_R100_preserved=True,oracle_stage_permutation='move last FFTaxis to first, repeated3times; P1axes(cfreq,a,b),P2(bfreq,cfreq,a),Y(afreq,bfreq,cfreq)',separate_fftn_final_oracle_max_complex_difference=final_difference,analytic_impulse=dict(location=[3,5,7],amplitude=[.75,-.5],expected_final='(.75-.5i)exp(-2pi*i*(3a+5b+7c)/256)'),files=files,NumPy=np.__version__,large_CPU_files_local_regenerable_not_Git=True,cold_command='PYTHONPATH=/tmp/pasc-julia-plot-tools OPENBLAS_NUM_THREADS=1 python3 src/zmddft-julia/generate_fixtures.py')
(P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({k:plan[k]for k in ['shape','doubles','input_prefix_count','artifact_count']}))

"""Independent exact IEEE32 leftfold and dyadic-count witnesses; no GPU."""
from pathlib import Path
import struct,json,hashlib
R=Path('/home/imo/HeCBench_julia');p=R/'src/simpleMultiDevice-julia';state=R/'.porting-state/portable_julia/simpleMultiDevice_reference_review';state.mkdir(exist_ok=True)
CASES=[(3,1),(8193,1),(131079,1),(33554432,0)];G=64;PART=8192;SENT=-777.
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0];u32=lambda x:struct.unpack('<I',struct.pack('<f',x))[0];u64=lambda x:struct.unpack('<Q',struct.pack('<d',x))[0]
def value(i,flavor):
 if flavor==0:return (i&255)/512.
 x=(i*1664525+1013904223)&0xffffffff;x ^=x>>16;x=(x*2246822519)&0xffffffff;x ^=x>>13
 return f32(f32(x)*2**-32)
def fold(a):
 total=0.
 for x in a:total=f32(total+x)
 return total
prefix=1+sum(2+n for n,flavor in CASES);reports={}
for ngpu in [1,2]:
 golden=state/f'golden_g{ngpu}_u64.bin';out=open(golden,'wb');out.write(struct.pack('<Q',u32(len(CASES))));cpu_inputs=[];suffix=[];report=[]
 for n,flavor in CASES:
  out.write(struct.pack('<2Q',u32(n),u32(flavor)))
  if flavor==0:
   pattern=struct.pack('<256Q',*[u32(i/512.)for i in range(256)]);out.write(pattern*(n//256)+pattern[:(n%256)*8]);cpu=n//256*sum(range(256))/512.;a=None
  else:
   a=[value(i,flavor)for i in range(n)];out.write(struct.pack('<'+'Q'*n,*map(u32,a)));cpu=0.
   for x in a:cpu+=x
  counts=[n//ngpu+(g<n%ngpu)for g in range(ngpu)];offset=0;local=[];rows=[]
  for count in counts:
   partial=[]
   for tid in range(PART):
    if flavor==0:
     steps=max(0,(count-1-tid)//PART+1);acc=steps*((offset+tid)&255)/512.;assert acc==f32(acc)
    else:
     acc=0.
     for i in range(tid,count,PART):acc=f32(acc+a[offset+i])
    partial.append(acc)
   total=fold(partial);local.append(total);suffix +=[u32(v)for v in partial]*2+[u32(total)]+[u32(SENT)]*(8*G)
   rows.append({'n':count,'partial_exact_oracle':'SequentialIEEE32roundaftereveryadd; largedyadicbound32M/8192<=4096terms,numeratormax4096*255<2^24, sums/512representable','source_local_FP32_fold':total});offset+=count
  gpu=fold(local);diff=abs(cpu-gpu)/abs(cpu);assert diff<1e-5,(n,ngpu,diff)
  suffix +=[u32(gpu),u64(cpu),u64(diff)];report.append({'n':n,'flavor':flavor,'devices':rows,'source_GPU_FP32_final':gpu,'source_CPU_FP64_final':cpu,'relative_difference':diff,'source_1e_5_pass':True})
 out.write(struct.pack('<'+'Q'*len(suffix),*suffix));out.close();(p/f'oracle_g{ngpu}_u64.bin').write_bytes(struct.pack('<'+'Q'*len(suffix),*suffix));reports[str(ngpu)]={'cases':report,'input_prefix_count':prefix,'output_count':prefix+len(suffix),'oracle_suffix_count':len(suffix),'golden_path':str(golden.relative_to(R)),'golden_sha256':hashlib.sha256(golden.read_bytes()).hexdigest(),'oracle_sha256':hashlib.sha256((p/f'oracle_g{ngpu}_u64.bin').read_bytes()).hexdigest()}
(p/'independent_oracle.json').write_text(json.dumps({'protocol':'ExactIEEEinput/partialvectors/hostsequentialFP32fold/sourceCPUFP64sum/guards; no pairwiseBase.sumsubstitution','cases':CASES,'input_generation':'Large32M: (global_i&255)/512 exactly; smallercontrols: UInt32LCG/xor/multiplyhash thencorrectFloat32conversion*2^-32; sourcehostrandgeneratorunmeasured.','proof':'Largekernelpartialswithexactdyadicboundednumerators; smallhashpartials independentlysequentialIEEE32CPUrounding; nativeCPUFP64sumexplicitglobalorder, GPUhostFP32foldexplicit8192order; source1e-5criterioncheckedbeforeGPU.','artifact_encoding':'UInt64zeroextendedrawIEEE32words forinputs/partials/Float32hostfolds/guards; sourceCPUFP64sum/reldiffrawIEEE64. Numericuint64exactnotrealerrorinterpretation.','configurations':reports},indent=2)+'\n');print(json.dumps({g:{k:v for k,v in d.items()if k!='cases'}for g,d in reports.items()},indent=2))

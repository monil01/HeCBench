from pathlib import Path
import numpy as np,json,hashlib
P=Path(__file__).resolve().parent;inputs=[];outputs=[];cases=[]
settings=[(0,16,16,16,'source'),(0,16,16,64,'source'),(0,32,32,64,'source'),(0,256,256,256,'source'),(1,256,256,256,'source'),(0,512,512,512,'source'),(1,512,512,512,'source'),(1,64,128,32,'source'),(0,64,64,64,'dyadic'),(1,64,64,64,'dyadic'),(0,16,16,16,'identity')]
for impl,m,n,k,mode in settings:
 def source(count):
  a=np.arange(count,dtype='i8')%13;return np.where(a%3!=0,-a,a)
 A=source(m*k).astype('f2');B=source(k*n).astype('f2');C=source(m*n).astype('f4')
 if mode=='dyadic':
  A=(((np.arange(m*k)*17+3)%31-15)/32).astype('f2');B=(((np.arange(k*n)*11+7)%29-14)/16).astype('f2');C=(((np.arange(m*n)*5+1)%17-8)/8).astype('f4')
 elif mode=='identity':A=np.eye(16,dtype='f2').ravel();B=((np.arange(256)%17-8)/8).astype('f2');C=np.zeros(256,dtype='f4')
 expected=(.5*(A.reshape(m,k).astype('f8')@B.reshape(n,k).T.astype('f8'))+2*C.reshape(m,n)).astype('f4').ravel()
 # All products/sums are exactly representable for these integer/dyadic witnesses.
 cases.append(dict(impl=impl,m=m,n=n,k=k,io=len(inputs),oo=len(outputs),mode=mode,repeats=100))
 inputs.extend(A.view('u2').astype('u4'));inputs.extend(B.view('u2').astype('u4'));inputs.extend(C.view('u4'));outputs.extend(expected)
a=np.array(inputs,dtype='<u4');o=np.array(outputs,dtype='<f4');(P/'input_u32.bin').write_bytes(a.tobytes());(P/'oracle_f32.bin').write_bytes(o.tobytes())
fields=['impl','m','n','k','io','oo','repeats'];(P/'study_cases.h').write_text('struct Case{int '+','.join(fields)+';};\nconst Case cases[]={\n'+''.join('{'+','.join(str(c[f])for f in fields)+'},\n'for c in cases)+'};\n')
(P/'study_cases.jl').write_text('const CASES=[\n'+''.join('('+','.join(f'{f}={c[f]}'for f in fields)+',),\n'for c in cases)+']\n')
plan=dict(cases=cases,input_words=len(a),output_words=len(o),exact_input_prefix_count=len(a)+11*10,guard_words_eachcase=128,artifact_count=len(a)+110+len(o)+11*128,fixture_dtype='UInt16FP16bitpatterns promotedUInt32/F32rawbitwords',oracle='IndependentNumPy2.5.4FP64GEMM→exactF32 integer/dyadic outputs, sourcealpha.5beta2',original_unmeasured=['1024³','4096³','8192³','impl1non64multiplegeometries'],files={n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in ['input_u32.bin','oracle_f32.bin']},NumPy=np.__version__);(P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps({k:plan[k]for k in ['input_words','output_words','exact_input_prefix_count','artifact_count']}))

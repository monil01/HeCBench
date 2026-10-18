"""Source network/MNIST fixtures and independent dense inference quality baseline."""
from pathlib import Path
import numpy as np,json,hashlib,struct
P=Path(__file__).resolve().parent;R=P.parents[1];DATA=R/'src/snicit-cuda/dataset/beyond';M=256
raw=(DATA/'MNIST/t10k-images-idx3-ubyte').read_bytes();magic,count,h,w=struct.unpack('>4I',raw[:16]);assert(magic,count,h,w)==(2051,10000,28,28);pixels=np.frombuffer(raw[16:],dtype='u1').reshape(count,784)[:M];MNIST=pixels.astype('f4')/np.float32(255);lraw=(DATA/'MNIST/t10k-labels-idx1-ubyte').read_bytes();assert struct.unpack('>2I',lraw[:8])==(2049,10000);labels=np.frombuffer(lraw[8:],dtype='u1')[:M].astype('i4');words=[];cases=[];sources={};dense=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def add(a,typ):
 a=np.ascontiguousarray(a,dtype=typ).ravel();offset=len(words);words.extend(a.view('<u4'));return offset

def load(path,typ='f4'):
 sources[str(path.relative_to(R))]=sha(path);return np.loadtxt(path,dtype=typ)
for ci,(name,H,L,T,density,cifar)in enumerate([('n128-l18-acc94.94',128,18,8,.6,False),('n256-l18-acc96.88',256,18,8,.6,False),('n256-l12-acc95.61',256,12,6,.5,False),('n256-l12-acc75.86',256,12,6,.5,True)]):
 WP=DATA/'networks/tsv_weights'/name;BP=DATA/'networks/tsv_biases'/name;N=H if cifar else 784;offset=1 if cifar else 2
 X=(((np.arange(M)[:,None]*17+np.arange(H)[None,:]*11+(np.arange(M)[:,None]*np.arange(H)[None,:])%31)%257)/256).astype('f4')if cifar else MNIST.copy();lab=(np.arange(M)%10).astype('i4')if cifar else labels.copy();c=dict(case=ci,name='ABCD'[ci],network=name,M=M,N=N,H=H,L=L,T=T,cifar=cifar,seed_size=128,io=len(words),x=add(X,'f4'),label=add(lab,'i4'),layers=[],real_data=not cifar)
 if not cifar:
  W=load(WP/'l1-dense.tsv').reshape(N,H);b=load(BP/'l1-dense.tsv');assert b.shape==(H,);c['win']=add(W,'f4');c['bin']=add(b,'f4');current=np.clip(X.astype('f8')@W.astype('f8')+b,0,1)
 else:c['win']=c['bin']=-1;current=X.astype('f8')
 for layer in range(L):
  a=load(WP/f'l{layer+offset}-sparse.tsv','f8');ri=a[:,0].astype('i4');co=a[:,1].astype('i4');va=a[:,2].astype('f4');assert np.all(np.diff(ri)>=0)and ri.min()>=0 and ri.max()<H and co.min()>=0 and co.max()<H;nnz=len(va);assert nnz==int(np.floor(np.float32(H*H)*np.float32(density)+.5));rp=np.r_[0,np.cumsum(np.bincount(ri,minlength=H))].astype('i4');b=load(BP/f'l{layer+offset}-sparse.tsv');assert b.shape==(H,)
  entry=dict(layer=layer,nnz=nnz,rp=add(rp,'i4'),col=add(co,'i4'),val=add(va,'f4'),bias=add(b,'f4'));c['layers'].append(entry);W=np.zeros((H,H),dtype='f8');np.add.at(W,(ri,co),va);current=np.clip(current@W+b,0,1)
 W=load(WP/f'l{L+offset}-dense.tsv').reshape(H,10);b=load(BP/f'l{L+offset}-dense.tsv');assert b.shape==(10,);c['wout']=add(W,'f4');c['bout']=add(b,'f4');exact=current@W.astype('f8')+b;c['dense_oracle_offset']=len(dense);dense.extend(exact.ravel());c['dense_reference_correct_labels']=int(np.count_nonzero(exact.argmax(axis=1)==lab));c['dense_reference_accuracy_valid_for_real_data']=not cifar;c['end']=len(words);cases.append(c)
a=np.array(words,dtype='<u4');a.tofile(P/'input_u32.bin');np.array(dense,dtype='<f8').tofile(P/'dense_oracle_f64.bin')
for p in [DATA/'MNIST/t10k-images-idx3-ubyte',DATA/'MNIST/t10k-labels-idx1-ubyte']:sources[str(p.relative_to(R))]=sha(p)
plan=dict(cases=cases,input_words=len(a),input_bytes=a.nbytes,input_sha256=sha(P/'input_u32.bin'),dense_oracle_sha256=sha(P/'dense_oracle_f64.bin'),dense_oracle_count=len(dense),batch=M,source_seed_size=128,source_original_sample_count=10000,source_metadata_initialization_repair='centroidLUT=0/nonempty=false/rows=0 beforeeachindependentrun, centroidsmarked-1 fromactualseedCPUcompaction',source_small_n_loader_bounds_witness='native_diagnostics/loader_n1.cu',CIFAR_input_and_labels_available=False,CIFAR_named_fixture='deterministicdyadicfeaturevectorsandlabels_i_mod10, originaltrainedDweights; notCIFARaccuracyclaim',source_files=sources,NumPy=np.__version__,independent_dense_baseline='FP64denseNNalllayers/clamp0..1, originalnetworkweights; qualitycomparisonnottranslationacceptanceoracle forapproxalgorithm')
(P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
# C-compatible metadata lines, each sparse layer row is separately indexed.
fields=['case','M','N','H','L','T','cifar','io','x','label','win','bin','wout','bout','dense_oracle_offset','end']
(P/'study_cases.h').write_text('struct Case{int '+','.join('id' if f=='case' else f for f in fields)+';};\nconst Case cases[]={\n'+''.join('{'+','.join(str(int(c[f]))for f in fields)+'},\n'for c in cases)+'};\nstruct Layer{int nnz,rp,col,val,bias;};\nconst Layer layers[4][18]={\n'+''.join('{'+','.join('{'+','.join(str(e[f])for f in ['nnz','rp','col','val','bias'])+'}'for e in c['layers'])+'},\n'for c in cases)+'};\n')
(P/'study_cases.jl').write_text('const CASES=[\n'+''.join('('+','.join(f'{"last" if f=="end" else f}={int(c[f])}'for f in fields)+',layers=['+','.join('('+','.join(f'{f}={e[f]}'for f in ['nnz','rp','col','val','bias'])+',)'for e in c['layers'])+'],),\n'for c in cases)+']\n')
print('SNICITinput',len(a),a.nbytes,'sources',len(sources),'densebaseline',[c['dense_reference_correct_labels']for c in cases])

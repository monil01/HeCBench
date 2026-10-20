#!/usr/bin/env python3
from pathlib import Path
import argparse,array,hashlib,json,math,struct
ap=argparse.ArgumentParser();ap.add_argument('fixture');ap.add_argument('--output',required=True);a=ap.parse_args();path=Path(a.fixture);f=path.open('rb');magic,version=struct.unpack('<2I',f.read(8));assert magic==0x4d465331 and version==1;m,n,k,nr=struct.unpack('<4i',f.read(16));nnz,nt=struct.unpack('<2q',f.read(16));parameters=struct.unpack('<9f',f.read(36));scale,ave,std,b,alpha,beta,lrate,lp,lq=parameters
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
def rd(T,N):
 x=array.array(T);x.frombytes(f.read(N*x.itemsize));assert len(x)==N;return x
pm,qm,ip,iq=rd('i',m),rd('i',n),rd('i',m),rd('i',n);assert sorted(pm)==list(range(m))and sorted(qm)==list(range(n));assert all(ip[pm[i]]==i for i in range(m))and all(iq[qm[i]]==i for i in range(n));R=[struct.unpack('<iif',f.read(12))for _ in range(nnz)];test=[struct.unpack('<iif',f.read(12))for _ in range(nt)];P=rd('H',m*k);Q=rd('H',n*k);rates=rd('f',nr);assert not f.read(1);f.close();base=Path(__file__).resolve().parent
original=list(struct.iter_unpack('<iif',(base/'data/movielens_u1_train.bin').read_bytes()));original_test=list(struct.iter_unpack('<iif',(base/'data/movielens_u1_test.bin').read_bytes()));assert test==original_test;assert len(original)==nnz
mean=sum(x[2]for x in original)/nnz;variance=sum(float(x[2])**2 for x in original)/nnz-mean*mean;assert ave==f32(mean)and std==f32(math.sqrt(variance))and scale==max(f32(1e-4),std)and b==f32(ave/std)
assert lp==lq==f32(f32(.05)/scale);invscale=f32(1./scale)
for (u,v,r),(actual_u,actual_v,actual_r)in zip(original,R):assert actual_u==pm[u]and actual_v==qm[v]and actual_r==f32(r*invscale)
assert all(rate==f32(alpha/(1.+beta*math.pow(i,1.5))+lrate)for i,rate in enumerate(rates));assert(m,n,k,nnz,nt,nr)==(943,1682,128,80000,20000,34)
features=[]
for name,words in [('P',P),('Q',Q)]:
 values=[struct.unpack('<e',struct.pack('<H',w))[0]for w in words];assert all(math.isfinite(x)and 0<=x<=.09 for x in values);features.append({'name':name,'half_count':len(words),'half_bits_sha256':hashlib.sha256(words.tobytes()).hexdigest(),'min':min(values),'max':max(values)})
out={'pass':True,'scope':'FreshCPUindependentauditofactualnativepreparedproblem, permuted/scaledall80kratings+20kheldout+allinitialhalfbitfeatures+34rates. NoSGDtraining/Juliaportacceptance. FeaturevaluesareactualsourceGPUreadbacks, notclaimedseedreconstructible.','fixture':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dimensions':{'m':m,'n':n,'k':k,'train':nnz,'test':nt},'parameters':dict(zip(['scale','ave','std','b','alpha','beta','lrate','lambda_p','lambda_q'],parameters)),'permutations_exact_inverse':True,'every_rating_mapping_scaling_checked':True,'every_dynamic_rate_exact_fp32_checked':True,'features':features};Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items()if k in ['pass','bytes','sha256','dimensions']})

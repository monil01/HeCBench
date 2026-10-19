#!/usr/bin/env python3
"""Independent coordinate Canny-derived source-policy oracle; no GPU required."""
import pathlib,struct,array,math,json,sys,hashlib
P=pathlib.Path(__file__).resolve().parent
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
weights=[1,2,1,2,4,2,1,2,1];kx=[-1,0,1,-2,0,2,-1,0,1];ky=[-1,-2,-1,0,0,0,1,2,1]
coeff=[struct.unpack('<i',struct.pack('<f',v/16))[0]for v in weights]+kx+ky
PI=f32(3.14159265);thresholds=[f32(f32(i*PI)/8)for i in [1,3,5,7,9,11,13,15]]
def oracle(data,rows,cols):
 g=[0]*len(data);mag=list(data);theta=[0]*len(data);nms=[0]*len(data);edges=list(data);min_angle_margin=float('inf');min_noninteger_hypot_margin=float('inf');canonical_gauss=[]
 for r in range(1,rows-1):
  for c in range(1,cols-1):
   vals=[data[(r+dy)*cols+c+dx]for dy in [-1,0,1]for dx in [-1,0,1]]
   # Source compound-int-add truncates after each exact dyadic term.
   g[r*cols+c]=sum(v*w//16 for v,w in zip(vals,weights));canonical_gauss.append(sum(v*w for v,w in zip(vals,weights))/16-g[r*cols+c])
 for r in range(1,rows-1):
  for c in range(1,cols-1):
   vals=[g[(r+dy)*cols+c+dx]for dy in [-1,0,1]for dx in [-1,0,1]];gx=sum(v*w for v,w in zip(vals,kx));gy=sum(v*w for v,w in zip(vals,ky));norm=math.hypot(gx,gy);mag[r*cols+c]=min(255,max(0,int(f32(norm))))
   frac=norm-math.floor(norm)
   if frac:min_noninteger_hypot_margin=min(min_noninteger_hypot_margin,frac,1-frac)
   angle=f32(math.atan2(gy,gx))
   if angle<0:angle=f32(math.fmod(f32(angle+f32(2*PI)),f32(2*PI)))
   min_angle_margin=min(min_angle_margin,min(abs(angle-t)for t in thresholds));direction=0
   for t,d in zip(thresholds,[0,45,90,135,0,45,90,135]):
    if angle<=t:direction=d;break
   theta[r*cols+c]=direction
 for r in range(1,rows-1):
  for c in range(1,cols-1):
   p=r*cols+c;d=theta[p];offset={0:[-1,1],45:[-cols+1,cols-1],90:[-cols,cols],135:[-cols-1,cols+1]}[d];nms[p]=mag[p]if all(mag[p]>mag[p+q]for q in offset)else 0;edges[p]=255 if nms[p]>=40 else 0
 return [g,mag,theta,nms,edges],dict(min_angle_threshold_margin=min_angle_margin,min_noninteger_hypot_floor_margin=min_noninteger_hypot_margin,per_term_truncation_vs_exact_gaussian_max=max(canonical_gauss),per_term_truncation_vs_exact_gaussian_mean=sum(canonical_gauss)/len(canonical_gauss),theta_categories=sorted(set(theta)),edge_count=sum(v==255 for v in edges),initial_boundary_policy='workinginput retainsimageborder; intermediate andtheta zero-defined beforepipeline',CPU_budget='Gaussian/angle/finaledge/guards exact; hypotmagnitude andNMSmagnitude absolute1 allowed for Float32libdevice rounding; dualGPU comparison exactallvalues.')
def image(rows,cols,frame):
 if frame==0:return [128]*(rows*cols)
 if frame==1:return [16+64*(r>=rows//2)+128*(c>=cols//2)for r in range(rows)for c in range(cols)]
 if frame==2:return [32+80*((r+c)//8%2)+96*((r-c)//11%2)for r in range(rows)for c in range(cols)]
 return [(r*13+c*7+(r//9)*(c//7)*11)%256 for r in range(rows)for c in range(cols)]
if __name__=='__main__':
 wrap=lambda a:[211]*16+list(a)+[211]*16
 cases=[];prefix=list(coeff);tail=[];trace=[];cursors=[];id=0
 for case,(rows,cols)in enumerate([(34,50),(66,98),(130,194)]):
  for frame in range(4):
   inp=image(rows,cols,frame);stages,review=oracle(inp,rows,cols);n=len(inp);name=f'case{id:02d}_input_u8.bin';(P/name).write_bytes(bytes(inp));prefix.extend([case,frame,rows,cols,16]);prefix.extend(wrap(inp));prefix.extend(wrap([0]*n));prefix.extend(wrap([0]*n));tail.extend(wrap(stages[4]));tail.extend(wrap(stages[3]));tail.extend(wrap(stages[2]));start=len(trace)
   for stage in stages:trace.extend(wrap(stage))
   cases.append(dict(id=id,case=case,frame=frame,rows=rows,cols=cols,n=n,input=name,input_sha256=hashlib.sha256(bytes(inp)).hexdigest(),review=review,trace_begin=start,trace_count=len(trace)-start));id+=1
 prefix.extend(coeff);expected=prefix+tail;(P/'oracle_primary_i32.bin').write_bytes(array.array('i',expected).tobytes());(P/'oracle_trace_i32.bin').write_bytes(array.array('i',trace).tobytes())
 h='#pragma once\nstruct CaseSpec{int id,caseid,frame,rows,cols,n;const char* input;};\n'
 h+=f'constexpr int CASES={len(cases)},PREFIX={len(prefix)},COUNT={len(expected)},TRACE_COUNT={len(trace)},TILE=16,SHMEM=18*18*4;\n'
 h+='const CaseSpec SPECS[]={'+','.join('{'+','.join(map(str,[c['id'],c['case'],c['frame'],c['rows'],c['cols'],c['n']]))+',"'+c['input']+'"}'for c in cases)+'};\n';(P/'study_params.h').write_text(h)
 jl=f'const TILE=16\nconst SHMEM=1296\nconst PREFIX={len(prefix)}\nconst COUNT={len(expected)}\nconst TRACE_COUNT={len(trace)}\nconst CASES=[\n'
 jl+='\n'.join('(id='+str(c['id'])+',caseid='+str(c['case'])+',frame='+str(c['frame'])+',rows='+str(c['rows'])+',cols='+str(c['cols'])+',n='+str(c['n'])+',input="'+c['input']+'"),'for c in cases)+']\n';jl+='const GAUSS=Float32['+','.join(str(w/16)for w in weights)+']\nconst SOBX=Int32['+','.join(map(str,kx))+']\nconst SOBY=Int32['+','.join(map(str,ky))+']\n';(P/'study_params.jl').write_text(jl)
 plan=dict(benchmark='ced',source_method='Canny-derivedGaussian(per-termInt32truncation),SobelFloat32hypot/atan2,quantizedorientation,NMS,pointthreshold40 (notconnectivityhysteresis).',cases=cases,input_prefix_count=len(prefix),artifact_count=len(expected),trace_count=len(trace),primary_ROI='Residentinput12framefour-source-kernelpipeline48launches+wholedevicecompletion; allocations/H2D/D2H/hostthread/partitiondriver excluded all4.',native_reference_repaired=True,repair='Define originallyuninitializedintermediate/angleborderszero; sourcekernelsunchanged. UnsafehostparseroriginalASANpreserved; namedbinaryloaderdefined.',CPU_output_budgets='Exactallprefix,guards,theta/finalbinaryinterior; NMSmagnitudeabs1, Sobelmagabs1 inuntimedtrace; Gaussian0. AllcrossGPUvaluesexact.',alpha_scope='GPUonlysourceMakefilealpha0; CPU/mixed/dynamicpartitionmodesunmeasured.',precision='UInt8storage; GaussianFloat32dyadicproduct+pertermInt32conversion; SobelgradientexactintegersrepresentedF32,norm/angleF32; NMSbytecomparisons.',source_coefficient_memory='Native__constant__ symbols; targetexplicitglobalGPUreadonlybycodecoefficientarrays (mappingdifferencemustbedisclosed).',resident_buffers_bytes=3*sum(c['n']+32 for c in cases),generator_python=sys.version,files={name:dict(bytes=(P/name).stat().st_size,sha256=hashlib.sha256((P/name).read_bytes()).hexdigest())for name in [c['input']for c in cases]+['study_params.h','study_params.jl','oracle_primary_i32.bin','oracle_trace_i32.bin']});(P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(cases=len(cases),input_prefix_count=len(prefix),artifact_count=len(expected),trace_count=len(trace),resident_buffer_bytes=plan['resident_buffers_bytes'],min_angle_margin=min(c['review']['min_angle_threshold_margin']for c in cases))))

#!/usr/bin/env python3
"""Independent integer-dyadic oracle for every saved managed input/output byte."""
from pathlib import Path
from array import array
import argparse,hashlib,json,sys,statistics
p=argparse.ArgumentParser();p.add_argument('result',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.result.read_text());assert sys.byteorder=='little';prefix=array('f');out=array('f');descriptors=[]
for n in (4099,1048579):
 A=array('f',(1. if n==4099 else (i%17-8)/16 for i in range(n)));B=array('f',(2. if n==4099 else ((i*5)%23-11)/32 for i in range(n)));G=array('f',(9. if n==4099 else (((i*5)%23-11)+14*(i%17-8))/32 for i in range(n)));prefix.extend((n,7));prefix.extend(A);prefix.extend(B)
 for _ in range(2):out.extend(A);out.extend(G)
 descriptors.append({'n':n,'array_bytes_each':4*n,'two_managed_array_bytes':8*n,'blocks':(n+255)//256,'threads':256,'inactive_tail_lanes':((n+255)//256)*256-n,'prefetch_calls_per_iteration':2,'iterations':7,'logical_kernel_bytes_per_iteration':12*n,'kernel_additions_per_iteration':n,'actual_migration_bytes':'notmeasured','actual_dram_bytes':'notmeasured'})
assert len(prefix)==2105360 and len(out)==4210712;expected=prefix.tobytes()+out.tobytes();assert len(expected)==6316072*4;expectedsha=hashlib.sha256(expected).hexdigest();rows=[];components={}
for m,md in d['models'].items():
 for t in md['trials']:
  raw=Path(t['output_artifact']['saved_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==t['output_artifact']['sha256']==expectedsha;assert raw==expected;rows.append({'model':m,'trial':t['trial'],'sha256':expectedsha,'exact_prefix_values':2105360,'exact_actual_output_values':4210712,'all_input_A_unchanged':True,'all_B_dyadic_math_exact':True})
 components[m]=[statistics.mean(t['component_seconds'][j] for t in md['trials'][1:]) for j in range(4)]
assert len(rows)==20
result={'benchmark':'prefetch','result':str(a.result.resolve()),'all20_exact':True,'oracle':'Arbitraryprecisionintegernumerators/32; integerbounds<=123 forpatterned outputs, so bothinput/outputexactly representableFP32. Sourceonesfinal9 alsoexact. NotusingGPUgridindexing/CPUfloataccumulation.','expected_artifact_sha256':expectedsha,'artifact_count':6316072,'rows':rows,'component_order':['4099_prefetch','4099_naive','1048579_prefetch','1048579_naive'],'component_means_last_four_seconds':components,'descriptors':descriptors,'scope':'NamedCPUfirsttouchmanagedmigration+add+periterationcompletion,2policies2sizes. No actualpagefault/migration/trafficcounter/oversubscription/crossvendorclaim; totaltime iscomposite, component samples retained.','source_binding':'ActualCuArrayUnifiedMemoryowners andhostviews/rawdeviceviews, copiesallcanonicalCPUvalues beforeeverytrial, ownersheldthroughsynchronization; bypassinstalledKernelAdaptorauto-prefetchso naive policyremains explicit. Prefixbytesalone are notdevice-inputreadback.'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'all20_exact':True,'artifact_bytes':len(expected),'component_means':components}))

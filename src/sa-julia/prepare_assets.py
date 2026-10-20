#!/usr/bin/env python3
from pathlib import Path
import argparse,array,json,hashlib,sys
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--genome',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);genome=a.genome.read_bytes();assert len(genome)>=1000000;guard=0x13579bdf
cases=[('primary',1000000,100,genome[:1000000])]
for n in [10,11,12]:cases.append(('unique'+str(n),n,1,bytes(range(65,65+n))))
for n in [31,32,33]:cases.append(('uniform'+str(n),n,1,b'A'*n))
for n in [34,35,36]:cases.append(('periodic'+str(n),n,1,(b'ACGT'*((n+3)//4))[:n]))
for n in [1021,1022,1023,1024,1025]:cases.append(('real'+str(n),n,1,genome[:n]))
records=[]
for name,n,repeats,b in cases:
 assert len(b)==n and min(b)>0 and max(b)<128
 v=array.array('i',list(b)+[0,0,0]+[guard]*17)
 if sys.byteorder!='little':v.byteswap()
 f=out/(name+'_input_i32.bin');f.write_bytes(v.tobytes());records.append({'name':name,'n':n,'internal_repeats':repeats,'input_file':f.name,'input_words':n+20,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size})
for which,subset in [('primary',records[:1]),('controls',records[1:])]:
 (out/(which+'_cases.txt')).write_text(''.join(f"{r['n']} {r['internal_repeats']} {r['input_file']}\n"for r in subset))
plan={'benchmark':'sa','algorithm':'DC3_sample_mod1_mod2_recursive_triple_rank_and_mod0_binarysearch_merge','source_custom_GPU_kernel_definitions':9,'source_device_lex_helpers':2,'cases':records,'primary':'fullsourceadvertised1Mgenomeprefix/R100default','controls':'14sourcepositiveASCIIcasesexerciseallnmod3,unique/uniform/periodic/realprefixranks;verification-onlyperformancenotaggregated','timing':'sourceH2Dinput_and_previousSA_assignment,completeDC3_withThrustalloc_sort_scan_D2D_GPUrankreadback_hostrecursion_and_workspacefree,D2HSA; averageoverR100forprimary; full-devicecompletionbeforetimerend; oracle/inputreadback/serializationexcluded','guard_scope':'Externalinput/output17tailInt32guards beyondoriginal3zero-sentinels; privateThrustworkspacesretainoriginalsourceextents/zeroinitialization, no extra guardsclaimed.','native_reference_repaired':False,'native_integration_adapted':True,'exclude_untouched_native_aggregates':True,'library_boundary':'Targetsreimplementall9customGPUkernelsanddata-dependenthostDC3controller; sharedCABIexposesoriginalThrustcontainer/sort_by_key/exclusive_scanonly, noDC3solver/customkerneldelegation','source_stream_policy':'singlelegacyqueue; JuliaCUDA.stream!(CUDA.legacy_stream()), KAcurrentstream/JACCspeclegacy; originalnative/runtime0samelegacysemantics; wholedevicecompletionexplicit','verification':'ExactInt32alloutputs/inputbits/guards; independentcompletepermutation+lex(first_symbol,inverse_rank_of_tail)certificatebyfinite-tailinduction, uniqueSA—notweakchecksum/ambiguouspropertyequivalence','atol':0,'rtol':0,'original_full1M_witness':'native_diagnostics/original_full1M_certificate.json','source_default_R100_targeted':True,'source_file_bytes':len(genome),'source_genome_sha256':hashlib.sha256(genome).hexdigest(),'fixed_seed_or_rng':'none','backend_architecture_planning_scope':'OpaqueThrustvector/CABIinterop/legacy-streammappingplannedinsharednativeprep beforeanytargetdraft; inheritedcontextmustbedisclosedforcalibration, notcleanroomsynthesis'};(out/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print('cases',len(records),'primaryR100','fixturebytes',sum(x['bytes']for x in records))

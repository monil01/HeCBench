from pathlib import Path
P=Path(__file__).resolve().parent
h=P/'profile_window.h';text=h.read_text();needle='extern "C" void profile_set_trial'
helper=r'''static void prof_attest_device(int rank,const char*stage){
 prof_need(rank==0||rank==1,"attestrank");int device=-1;char pci[64]={},uuid[64]={};cudaDeviceProp props{};
 prof_cuda(cudaGetDevice(&device));prof_need(device==rank,"attestlogicaldevicerank");prof_cuda(cudaDeviceGetPCIBusId(pci,sizeof(pci),device));
 prof_need(prof_same_pci(pci,getenv(rank?"STUDY_EXPECTED_GPU6_PCI":"STUDY_EXPECTED_GPU5_PCI")),"attestphysicalPCI");
 prof_cuda(cudaGetDeviceProperties(&props,device));const unsigned char*u=reinterpret_cast<const unsigned char*>(props.uuid.bytes);
 snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
 const char*expected=getenv(rank?"STUDY_EXPECTED_GPU6_UUID":"STUDY_EXPECTED_GPU5_UUID");prof_need(expected&&strcmp(uuid,expected)==0,"attestphysicalUUID");
 printf("ALLREDUCE_DEVICE_ATTEST {\"rank\":%d,\"logical_device\":%d,\"pid\":%d,\"pci\":\"%s\",\"uuid\":\"%s\",\"phase\":\"%s\"}\n",rank,device,int(getpid()),pci,uuid,stage);fflush(stdout);
}
'''
text=text.replace(needle,helper+needle);text=text.replace('++prof_calls;','++prof_calls;if(prof_calls==1)prof_attest_device(rank,"before_first_warmup_add");');text=text.replace('extern "C" void profile_finish(int rank){','extern "C" void profile_finish(int rank){\n prof_attest_device(rank,"after_all5_full_checks");');h.write_text(text)
r=P/'run_proposal.py';text=r.read_text();text=text.replace("env['STUDY_EXPECTED_GPU6_PCI']=snap['devices']['6']['pci']", "env.update({f'STUDY_EXPECTED_GPU{i}_{key.upper()}':snap['devices'][i][key]for i in ('5','6')for key in ('pci','uuid')})")
text=text.replace("trials=[];begins=[];ends=[];done=[]", "trials=[];begins=[];ends=[];done=[];attest=[]")
text=text.replace("('ALLREDUCE_PROFILE_DONE ',done)]", "('ALLREDUCE_PROFILE_DONE ',done),('ALLREDUCE_DEVICE_ATTEST ',attest)]")
needle=" begin=begins[0];end=ends[0]"
addition=''' if len(attest)!=4:raise RuntimeError('bothrankprewarmup_and_final_deviceattestation_required')
 for rank in(0,1):
  for phase in('before_first_warmup_add','after_all5_full_checks'):
   matches=[x for x in attest if x.get('rank')==rank and x.get('phase')==phase]
   if len(matches)!=1:raise RuntimeError('exactrankattestationphase')
   x=matches[0];env=rs[rank]['effective_environment'];physical=str(rank+5)
   if(x.get('pid'),x.get('logical_device'),x.get('uuid'))!=(rs[rank]['pid'],rank,env[f'STUDY_EXPECTED_GPU{physical}_UUID']):raise RuntimeError('runtimephysicalUUIDrankattestation')
   def pci_tuple(value):return tuple(int(v,16)for v in re.split('[:.]',value))
   if pci_tuple(x['pci'])!=pci_tuple(env[f'STUDY_EXPECTED_GPU{physical}_PCI']):raise RuntimeError('runtimephysicalPCIrankattestation')
'''
text=text.replace(needle,addition+needle)
text=text.replace("'rank_registry':rs}","'rank_registry':rs,'device_attestations':attest}")
# Bind exact host architecture before authorizing/staging any runtime.
text=text.replace(" if bundle['benchmark']", " if os.uname().machine!='x86_64':raise RuntimeError('exactreviewedNCUhostarchitecture_required')\n if bundle['benchmark']")
r.write_text(text)
# Record effective expected identity env in registry.
w=P/'rank_worker.py';s=w.read_text();print(s)

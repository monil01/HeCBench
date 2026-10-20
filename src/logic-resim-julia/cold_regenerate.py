#!/usr/bin/env python3
"""CPU-only cold fixture/oracle regeneration, exact hashes, no benchmark/GPU calls."""
from pathlib import Path
import argparse,subprocess,sys,shutil,tempfile,hashlib,json
P=Path(__file__).resolve().parent;R=P.parent.parent;a=argparse.ArgumentParser();a.add_argument('--materialize',action='store_true');args=a.parse_args()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
manifest=json.load(open(P/'asset_manifest.json'))['assets']if(P/'asset_manifest.json').exists()else{}
with tempfile.TemporaryDirectory(prefix='logic-cold-')as td:
 root=Path(td);out=root/'src/logic-resim-julia';out.mkdir(parents=True);(root/'src/logic-resim-cuda').symlink_to(R/'src/logic-resim-cuda',target_is_directory=True)
 for f in P.iterdir():
  if f.is_file()and f.suffix in ['.py','.cpp','.h','.cuh','.json']:shutil.copy2(f,out/f.name)
 subprocess.run([sys.executable,str(out/'generate_graph.py')],check=True)
 subprocess.run([sys.executable,str(out/'extract_sources.py')],check=True)
 exe=root/'prepare_inputs';subprocess.run([sys.executable,str(out/'build_prepare_inputs.py'),str(exe)],check=True)
 source=R/'src/logic-resim-cuda/Simulation/toy';subprocess.run([str(exe),str(source/'intermediate.file'),str(source/'NV_NVDLA_partition_o_dc_24x33x55_5x5x55x25_int8_input.vcd'),str(out/'initial_waves_u64.bin')],check=True)
 cpu=root/'oracle';subprocess.run(['g++','-O3','-std=c++17',str(out/'independent_oracle.cpp'),'-o',str(cpu)],check=True);subprocess.run([str(cpu),str(out/'initial_waves_u64.bin'),str(out/'graph_u64.bin'),str(out/'truth_table_u8.bin'),str(out/'oracle_waves_u64.bin'),str(out/'oracle_occupancy_u64.bin')],check=True)
 checks=[]
 for name in ['graph_u64.bin','initial_waves_u64.bin','truth_table_u8.bin','oracle_waves_u64.bin','oracle_occupancy_u64.bin','wire_names.json','wire_aliases.json','graph_audit.json','source_kernel.cuh','source_cpu_fallback.cpp','source_extraction.json','source_interParser_input_decode.cpp']:
  f=out/name;old=P/name;expected=sha(old)if old.exists()else manifest[name]['sha256'];checks.append({'file':name,'sha256':sha(f),'expected_sha256':expected,'bytes':f.stat().st_size,'same':sha(f)==expected})
 assert all(x['same']for x in checks),[x for x in checks if not x['same']]
 if args.materialize:
  for f in out.glob('*.bin'):
   if not(P/f.name).exists():shutil.copy2(f,P/f.name)
  for child in P.glob('logic-resim-julia-*'):
   if child.is_dir():
    for f in out.glob('*.bin'):
     if not(child/f.name).exists():(child/f.name).symlink_to(Path('..')/f.name)
 report={'benchmark':'logic-resim','status':'pass','CPU_only':True,'GPU_kernel_invocations':0,'original_source_GPU_dispatch_not_linked_or_executed':True,'integer_oracles_no_NumPy_float_arithmetic':True,'checks':checks,'python':sys.version,'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],'command':sys.argv,'source_assets_already_tracked':['src/logic-resim-cuda/Simulation/toy/intermediate.file','src/logic-resim-cuda/Simulation/toy/NV_NVDLA_partition_o_dc_24x33x55_5x5x55x25_int8_input.vcd']}
 (P/'cold_regeneration_audit.json').write_text(json.dumps(report,indent=2)+'\n');print('COLD PASS',len(checks),'assets; CPUonly/noGPU')

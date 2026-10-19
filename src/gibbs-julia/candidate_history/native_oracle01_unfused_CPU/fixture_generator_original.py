from pathlib import Path
import array,json,hashlib,sys
P=Path(__file__).resolve().parent;N=513;steps=32
y=[i%7 for i in range(N)];n=[.5+(i%13)*.125 for i in range(N)]
(P/'y_i32.bin').write_bytes(array.array('i',y).tobytes());(P/'n_f32.bin').write_bytes(array.array('f',n).tobytes());(P/'observations.csv').write_text(''.join(f'{a} {b:.3f}\n'for a,b in zip(y,n)))
files={name:{'sha256':hashlib.sha256((P/name).read_bytes()).hexdigest(),'bytes':(P/name).stat().st_size}for name in ['y_i32.bin','n_f32.bin','observations.csv']}
plan={'benchmark':'gibbs','N':N,'steps':steps,'seed':123,'initial_a':20,'initial_b':1,'blocks':3,'threads':256,'original_steps_default':1000,'prefix_count':7900,'artifact_count':225884,'CPU_conditional_gamma_atol':3e-5,'CPU_reduction_atol':2e-4,'dual_atol':3e-5,'dual_rtol':0,'state_integer_bitfields_exact':True,'files':files,'python':sys.version,'original_dataset':'DVCdata.tar.gz absent; syntheticpositiveexposures/nonnegativecounts','ROI':'Sum32separate source2kernel launches+wholedevicecompletion, excludesactualCPUfeedback+fullstepcaptures; notfullMCMCwalltime','native_reference_repaired':False,'native_integration_adapted':True,'exclude_untouched_native_aggregates':True,'original_setup_kernel_interop_outside_ROI':True}
(P/'study_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(P/'study_params.h').write_text('#define N 513\n#define STEPS 32\n#define BLOCKS 3\n#define PREFIX 7900\n#define COUNT 225884\n')
(P/'study_params.jl').write_text('const N=513\nconst STEPS=32\nconst BLOCKS=3\nconst PREFIX=7900\nconst COUNT=225884\n')

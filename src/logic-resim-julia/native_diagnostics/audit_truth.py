#!/usr/bin/env python3
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;R=P.parents[2];D=R/'.porting-state/diagnostics/logic-resim';source=R/'src/logic-resim-cuda/Simulation/toy/intermediate.file';raw=(D/'metadata_capture/actual_truth_bytes.bin').read_bytes()
with source.open()as f:
 h=list(map(int,next(f).split()));truth=[next(f).split()for _ in range(h[4])]
used=0;wrong=[];poison_wrong=[];packed=bytearray(1025*h[4])
for func,(degree,values)in enumerate(truth):
 n=int(degree);assert len(values)==4**n
 for k,ch in enumerate(values):packed[func*1025+k//4]|=int(ch)<<((k%4)*2)
 packed[func*1025+1024]=n
 for k in range(len(values)//4):
  pos=func*1025+k;used+=1
  if raw[pos]!=packed[pos]:wrong.append({'func':func,'byte':k,'actual':raw[pos],'expected':packed[pos]})
  poisoned=255
  # The exact original |= updates cannot clear any initialized-one bit.
  for z in range(4):poisoned|=int(values[4*k+z])<<(2*z)
  if poisoned!=packed[pos]:poison_wrong.append({'func':func,'byte':k,'poisoned_packed':poisoned,'expected':packed[pos]})
r={'benchmark':'logic-resim','source_parser_static_uninitialized_read':'h_eTable=newchar[EvalC*TRUTH_SIZE], parseEval uses|=beforeinitialization','actual_original_used_truth_bytes':used,'actual_original_used_truth_byte_mismatches':len(wrong),'actual_original_mismatch_examples':wrong[:8],'deterministic_poison_initialization_witness_mismatches':len(poison_wrong),'poison_examples':poison_wrong[:8],'poison_is_controlled_CPU_allocation_state_witness_not_observed_original_GPU_wrongwave':True,'actual_original_unused_padding_not_scientific_output':True,'source_truth_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'actual_GPU_truth_buffer_sha256':hashlib.sha256(raw).hexdigest(),'defined_zero_packed_truth_sha256':hashlib.sha256(packed).hexdigest(),'observed_original_GPU_fault':False}
(P/'truth_initialization_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(r)

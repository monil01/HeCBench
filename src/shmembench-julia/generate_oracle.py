"""Exact defined FP32 output of the source float4 swap/reduce kernel."""
from pathlib import Path
import array,struct,json,hashlib
P=Path(__file__).resolve().parent;N=1048576;offsets=[0,1,3,7,13,17];components=[0,11,19,23]
def main():
    order=list(range(6))
    for _ in range(1024):
        for a,b in [(0,1),(2,3),(4,5),(1,2),(3,4)]:order[a],order[b]=order[b],order[a]
    output=array.array('f')
    for gid in range(N//4):
        tid=gid%256
        output.extend(sum(tid+offsets[k]+c for k in order)for c in components)
    assert len(output)==N
    # The kernel writes only half of its source size*sizeof(double) allocation.
    defined=output.tobytes();checksum=0.
    for (v,) in struct.iter_unpack('<d',defined):checksum+=v
    original_constant=21256458760384741137729978368.00
    oldwrong=sum((6*(gid%256)+41+6*c)!=(6*(gid%256)+41+6*components[c])for gid in range(N//4)for c in range(4))
    output.extend([16777215.]*N);data=output.tobytes();(P/'expected_f32_guards.bin').write_bytes(data)
    metadata={'VECTOR_SIZE':N,'physical_blocks':1024,'threads':256,'iterations_per_kernel':1024,'primary_repeat':1000,'shared_vectors':1536,'shared_bytes':24576,'vector_components':'Float32x4/nativefloat4 and JuliaNTuple4VecElementFloat32 (alignmentcheckedbyactualcompiler)','defined_output_float32_values':N,'allocated_output_bytes':N*8,'unwritten_tail_float32_guard_values':N,'guard_float32':16777215.,'component_sums':[41,107,155,179],'sixplane_order_after1024_swaps':order,'oracle':'Independent six-slot permutation simulation followed by exact integer per-component sums, encodedFP32; staticparameters/sourcehashes, noGPUinputbuffer','swap_sequence_not_identified_by_final_sum':True,'historical_scalar_component_init_wrong_outputs':oldwrong,'native_host_checksum_defined_half_reinterpreted_as_Float64_sum':checksum,'native_host_checksum_constant':original_constant,'native_host_checksum_matches_if_undefined_tail_zero':checksum==original_constant,'source_host_checksum_caveat':'Sourceallocates8MiB butkernelwrites4MiB, copiesall8MiB/readsasdouble; unwrittendevicehalfundefined. DefinedlogicalFP32outputandunwritten-tailguardcheckedinstead. Kernelunchanged; originalhostchecksum/CLI notcertified. GuardinitializationoutsideROIneverreadbykernel.','source_intended_shared_access_bytes_per_launch':(6+4*5*1024+6)*N*4,'source_operation_count_caveat':'Sourceassumptiononly. Numericaloutputinvariantunderpermutationscannottestexecutionofswaps. Profileactualsharedinstructions/trafficseparately; nevercallcalculatednumeratordividedbytimemeasuredbandwidth.','expected_sha256':hashlib.sha256(data).hexdigest()};(P/'fixture_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n');print(metadata)
if __name__=='__main__':main()

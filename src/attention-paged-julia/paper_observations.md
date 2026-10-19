# Paged-attention observation checkpoint

Single-query decoder attention over a paged key/value cache. FP32 block16/equal query-KV heads, three named cases D32/64/128; lengths1/15/16/33/129/257, ALiBi/tails/repeated pages/constantV. Original BF16 actualsm70 deviceassert; no BF16target baseline. Sparse, partitioned, GQA, other D/defaultCLI sizes not evaluated. Original GPU math unchanged; faulty CPU packed-key checker independently replaced.

RQ1: All20fullrecords pass independent densepacked-key attention anddual GPUreferences. OriginalCPUchecker misindexespackedkeys: minimal witness differs by0.6088587 whileunchangednative matchesindependentmath within2.27e-8. Source epsilon normalization retained.

RQ2/RQ4: CUDAjl4programchecks (3deviceJITprocesses), KA2, JACC1; parse, APIbroadcast, tupleboxing andKAindexrematerialization feedback retained separately. KAactualdevicewait requiredboundedinterrupt; CPUmacroexpansion+repairedoutcome identify index lifetime. These are orderedworkflow observations, notagent efficacy orcoldstart rankings. Real usage andmodel context bound inpermodel manifests/calibration.

RQ3: Meanlast4 of5checkedtrials, sum3cases microseconds: {'cuda': 54.7975, 'cuda_jl': 214.39700000000002, 'ka': 266.9275, 'jacc': 343.366}. Complete percase times retained; nooutlierdrop orattributedbottleneck.

RQ5: AllJulia mappings useownframeworkdispatch/shared/barriers plus CUDAwarp/fastmath extensions. No crossvendor claim; BF16native hardwareboundary remainsseparate.

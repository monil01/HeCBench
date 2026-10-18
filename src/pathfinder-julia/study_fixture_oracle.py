from pathlib import Path
import json,struct,shutil,hashlib
R=Path('/home/imo/HeCBench_julia');p=R/'src/pathfinder-julia';cases=[(1,251,5),(33,11,5),(257,13,5),(499,8,3),(1001,11,5)];f=[len(cases)];states=[];sent=-12345
for cols,rows,h in cases:
 a=[[ (r*17+c*13+(r*c)%7)%10 for c in range(cols)]for r in range(rows)];f +=[cols,rows,h]+[x for row in a for x in row];dp=[a[0]]
 for row in a[1:]:dp.append([row[c]+min(dp[-1][max(0,c-1):min(cols,c+2)])for c in range(cols)])
 debug=[0]*16384;previous=0;grid=rows*cols//250;launches=0
 for t in range(0,rows-1,h):
  it=min(h,rows-1-t);small=250-2*it
  for bx in range(grid):
   x=small*bx-h+11
   if 0<=x<cols:debug[dp[t][x]]=1
  previous=t;launches+=1
 states += [sent]+[x for row in a[1:]for x in row]+[sent]+[sent]+dp[-1]+[sent]+[sent]+dp[previous]+[sent]+[sent]+debug+[sent]
raw=lambda x:struct.pack('<%di'%len(x),*x)
(p/'fixtures_i32.bin').write_bytes(raw(f));(p/'oracle_i32.bin').write_bytes(raw(states));(p/'independent_oracle.json').write_text(json.dumps({'cases':cases,'prefix_count':len(f),'output_count':len(f)+len(states),'oracle_suffix_words':len(states),'expected_full_artifact_sha256':hashlib.sha256(raw(f+states)).hexdigest(),'method':'IndependentCPUintegerrow-by-rowclamped3-neighborminimum DP, notsourceghostzonealgorithm. Final/previouspyramidrow bothpingpong; debugmarks sourcefirstiterationtx11 mapping; fullreadonlywall/canaries.','source_grid':'floor(rows*cols/250), includingidleblocks, sameall4','scope':'0..9costs,5namedrectangles, temporalpartialbatch2/1, edgecols1/33/257/499/1001, original100000x1000h5unmeasured','debug_policy':'Initialize0outsideROI; sourceidempotentnon-atomicconcurrentstores1 preserved, noformalCUDAdata-race-freedomclaim.'},indent=2)+'\n')
(p/'SCOPE.md').write_text('''# Declared Pathfinder scope\n\nGenuine source250-thread shared-memory ghost-zone dynamicprogramming, sourcegridfloor(rows*cols/250) including idleblocks, fixedborderheight, source temporalpyramids andpartiallastbatch, two250Int32sharedarrays/2000bytes. Nativekernel unchanged. Named(1cols,251rows,h5),(33,11,5),(257,13,5),(499,8,3),(1001,11,5), source0..9 deterministiccosts. Original100000x1000h5 unmeasured. Compilewarmup before5trials; kernelpipeline+finalcontextcompletionROI only, allocations/copies/reset/captureoutside, sum5pipelineintervals60kernels/trial. NativeCLI offloadinclusiveROI separateunmeasured.\n\nInitializebothpingpongarraysand16384debugwords deterministicallyoutsideROI. Nativeuninitializeddebuguntouchedslots become0namedfixture; no devicealgorithmrepair. Preserve actualsource tx11/i0 diagnostic globalstore1 (possiblysame-indexidempotentstores; noformalracefreeclaim). Exacteveryfinalandpreviouspyramidrow/bothbuffers, fulldebugarray, readonlywall andguards, fullactualhostinputprefix vsindependentintegerCPUrow-DP+separatediagnosticindexoracle; all5exactdualreferences. No tracebackpath/crossbackendproof/globalroutingapplicationqualityclaim.\n\nHistoricalJulia row-by-row recurrence mathematicallymatchesDP butdifferent launch/temporaltiling/no sharedmapping andomitsunusednativeinstrumentation. P1mappingalignment, not establishedscientificS1. Sourcecost generationnamedcommonfixture; sourceRNGCLI sequenceunmeasured.\n''')
print(len(f),len(f)+len(states))

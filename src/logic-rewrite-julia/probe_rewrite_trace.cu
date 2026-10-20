// Source subsystem unit only: explicit defined initialized fixture, not CLI claim.
#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>
#include "reference_history/robin_hood.h"
#define private public
#include "reference_history/rewrite.h"
#undef private
static void ck(cudaError_t e){if(e)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>void capture_device_array(FILE*f,const T*p,size_t n){std::vector<T>x(n);ck(cudaMemcpy(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));fwrite(x.data(),sizeof(T),n,f);}
#include "reference_history/rewrite.cu"
__global__ void EvaluateNodeTrace(int sz, int *bestout, int *fanin0, int *fanin1, int *isC0, int *isC1, int *nodeLevels, Cut *cuts, Cut* selectedCuts, int *nRef, 
                             Library *lib, TableNode* hashTable, int fUseZeros) {
    if(blockIdx.x * blockDim.x + threadIdx.x >= sz) return;
    int id = 1 + blockIdx.x * blockDim.x + threadIdx.x, reduction = -1, bestLevel = 99999999, bestCut = -1, bestOut;
    int match[54], tableSize, tableId[TABLE_SIZE], tableNum[TABLE_SIZE];
    int matchLevel[54];
    for(int i = 0; i < CUT_SET_SIZE; i++) {
        Cut *cut = cuts + ID(id, i);
        if(cut->used == 0 || cut->nLeaves < 3) continue;
        int nleaves = cut->nLeaves;
        if(nleaves == 3)
            cut->leaves[cut->nLeaves++] = 0;
        tableSize = 0;
        int saved = CalcMFFC(id, cut, fanin0, fanin1, &tableSize, tableId, tableNum, nRef, id);
        int uPhase = lib->pPhases[cut->truthtable];
        int Class = lib->pMap[cut->truthtable];
        int *pPerm = lib->pPerms4[lib->pPerms[cut->truthtable]];
        for(int j = 0; j < 54; j++)
            match[j] = matchLevel[j] = -1;
	      uint64_t isC = 0;
        for(int j = 0; j < 4; j++) {
            match[j] = cut->leaves[pPerm[j]];
            matchLevel[j] = nodeLevels[match[j]];
            if(uPhase >> j & 1)
		            isC |= 1LL << j;
        }
        for(int j = 0; j < lib->nNodes[Class]; j++) {
            int num = j + 4;
            int in0 = lib->fanin0[Class][j], in1 = lib->fanin1[Class][j];
            assert(matchLevel[in0] != -1 && matchLevel[in1] != -1);
            matchLevel[num] = 1 + (matchLevel[in0] > matchLevel[in1] ? matchLevel[in0] : matchLevel[in1]);

            if(match[in0] == -1 || match[in1] == -1 || match[in0] == id || match[in1] == id) continue;
            int nodeId = TableLookup(match[in0], match[in1], (isC >> in0 & 1) ^ lib->isC0[Class][j], (isC >> in1 & 1) ^ lib->isC1[Class][j], hashTable, fanin0, fanin1, isC0, isC1);
            if(nodeId != -1 && !IsDeleted(nodeId, &tableSize, tableId, tableNum)) {
                match[num] = nodeId;
                matchLevel[num] = nodeLevels[nodeId];
            }
        }
        for(int out = 0; out < lib->nSubgr[Class]; out++) {
            int rt = lib->pSubgr[Class][out];
            if(id==10 || id==13) printf("TRACE0 %d %d %d %d %d %d %d\n",id,i,out,saved,Class,rt,match[rt]);
            if(match[rt] == id) continue;
            int nodesAdded = Eval(rt, match, Class, lib, -out - 2);
            int rtLevel = matchLevel[rt];
            if(id==10 || id==13) printf("TRACE1 %d %d %d %d %d %d\n",id,i,out,saved,nodesAdded,rtLevel);
            assert(rtLevel != -1);
            // if(saved - nodesAdded > reduction || (fUseZeros && saved == nodesAdded && bestCut == -1)) {
            // if(saved - nodesAdded > reduction || (fUseZeros && saved == nodesAdded)) {
            //     reduction = saved - nodesAdded;
            //     bestCut = i;
            //     bestOut = out;
            // }
            if (saved - nodesAdded < 0 || (saved - nodesAdded == 0 && !fUseZeros))
                continue;
            if (saved - nodesAdded < reduction || (saved - nodesAdded == reduction && rtLevel >= bestLevel))
                continue;
            reduction = saved - nodesAdded;
            bestLevel = rtLevel;
            bestCut = i;
            bestOut = out;
        }
        cut->nLeaves = nleaves;
    }
    if(bestCut != -1) {
	// atomicAdd(&GPUexpected, 1);
        selectedCuts[id] = cuts[ID(id, bestCut)];
        selectedCuts[id].used = 1;
        bestout[id] = bestOut;
    } else
        selectedCuts[id].used = 0;
}



int main(int argc,char**argv){if(argc!=2)return 1;try{
 rw::CPUSolver cpu;memset(&cpu.lib,0,sizeof(cpu.lib));cpu.ReadLibrary();int n=13,pi=4;
 // Level-sorted equivalent of the13-node earlier unit; literal flags retained.
 std::vector<int>a={0,0,0,0,0,2,6,3,2,3,10,17,22,24};
 std::vector<int>b={0,0,0,0,0,4,8,5,7,6,12,19,21,8};
 std::vector<int>levels={0,0,0,0,0,1,1,1,1,1,2,2,3,4},refs(n+1),phase(n+1);std::vector<int>counts={4,9,11,12,13};
 for(int j=pi+1;j<=n;j++){refs[a[j]/2]++;refs[b[j]/2]++;phase[j]=(phase[a[j]/2]^(a[j]&1))&(phase[b[j]/2]^(b[j]&1));}refs[13]++;
 rw::GPUSolver gpu(n);gpu.CopyLib(cpu.lib);ck(cudaMemset(gpu.selectedCuts,0,(n+1)*sizeof(rw::Cut)));ck(cudaMemset(gpu.bestSubgraph,0,(n+1)*4));
 FILE*f=fopen(argv[1],"wb");fwrite(&n,4,1,f);fwrite(&pi,4,1,f);int len=counts.size();fwrite(&len,4,1,f);fwrite(counts.data(),4,len,f);fwrite(a.data(),4,n+1,f);fwrite(b.data(),4,n+1,f);fwrite(levels.data(),4,n+1,f);fwrite(refs.data(),4,n+1,f);fwrite(&cpu.lib,1,sizeof(cpu.lib),f);
 gpu.EnumerateAndPreEvaluate(levels.data(),counts,n,a.data(),b.data(),refs.data(),false);ck(cudaDeviceSynchronize());
 capture_device_array(f,gpu.fanin0,n+1);capture_device_array(f,gpu.fanin1,n+1);capture_device_array(f,gpu.isComplement0,n+1);capture_device_array(f,gpu.isComplement1,n+1);capture_device_array(f,gpu.cuts,(n+1)*8);capture_device_array(f,gpu.selectedCuts,n+1);capture_device_array(f,gpu.bestSubgraph,n+1);fclose(f);EvaluateNodeTrace<<<1,512>>>(n,gpu.bestSubgraph,gpu.fanin0,gpu.fanin1,gpu.isComplement0,gpu.isComplement1,gpu.phase,gpu.cuts,gpu.selectedCuts,gpu.nRef,gpu.lib,gpu.hashTable,0);ck(cudaDeviceSynchronize());printf("NATIVE_SOURCE_REWRITE_PRE_EVALUATION_CAPTURE n=%d\n",n);return 0;
}catch(std::exception&e){fprintf(stderr,"%s\n",e.what());return 2;}}

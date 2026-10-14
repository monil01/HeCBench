#include <cuda_runtime.h>
#include "layout.h"
extern "C" __global__ void simulateParallel(bool *overflow, const FlatGate *gateMgr,
                                 Event *output, const size_t *offsets, const Event *input, const size_t *oHisMaxSize,
                                 size_t *oHisSize, const char *eTableMgr,
                                 size_t SimLimit, tUnit dumpOff) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid > SimLimit - 1)
    return;

  // Gate info
  size_t maxSize = oHisMaxSize[gid], currSize;
  const char *eTable = eTableMgr + gateMgr[gid].funcSer * TRUTH_SIZE;
  int8_t inLen = *(eTable + TRUTH_SIZE - 1);

  // Simulation cache
  const Event *iPort[MAX_INPUT_PORT];
  Event *oPort = output + offsets[gid];

  tUnit currTime = -1, iTnext[MAX_INPUT_PORT] = {0};
  bool isInit = true;

  dUnit delay = -1, d;

  int16_t iVcurr = 0, iVprev = 0;

  int8_t oVcurr = 0, oVprev, i, iFrom, iTo, delayIdx;

  overflow[gid] = false;
  // Init first input pattern and first trans time
  for (i = 0; i < inLen; i++) {
    iPort[i] = input + gateMgr[gid].iPort[i];
    iVprev = iVprev << 2;
    iVprev |= iPort[i]->v;
    iTnext[i] = (iPort[i] + 1)->t;

    isInit = (iPort[i]->t > 0) ? false : isInit;
    currTime = (iTnext[i] < currTime) ? iTnext[i] : currTime;
  }

  // Init not simulated gate
  if (isInit) {
    currSize = 1;
    oPort[0] = {0, valueX};
    oVprev = valueX;
  } else {
    currSize = oHisSize[gid];
    oVprev = oPort[currSize - 1].v;
  }

  // Drop from full simulated gate
  if (currTime == (tUnit)-1 || currTime > dumpOff) {
    if (oPort[currSize - 1].t != (tUnit)-1) {
      oPort[currSize] = {(tUnit)-1, valueZ};
      oHisSize[gid] = currSize + 1;
    }
    return;
  }

  // Simulation loop
  while (currTime != (tUnit)-1 && currTime <= dumpOff) {
    // Update newly coming input pattern and corresponding output
    for (i = 0; i < inLen; ++i) {
      if (iTnext[i] == currTime) {
        ++iPort[i];
        iTnext[i] = (iPort[i] + 1)->t;
      }
      iVcurr = iVcurr << 2;
      iVcurr |= iPort[i]->v;
    }

    // Get new output value
    oVcurr = (eTable[iVcurr >> 2] >> ((iVcurr & 3) * 2)) & 3;

    // Trigger edge at output
    if (oVcurr != oVprev) {
      // Get minimum delay and next output trans time
      switch (oVprev << 2 | oVcurr) {
      case 1:
        delayIdx = 0;
        break; // 01
      case 4:
        delayIdx = 1;
        break; // 10
      case 2:
        delayIdx = 2;
        break; // 0X
      case 9:
        delayIdx = 3;
        break; // X1
      case 6:
        delayIdx = 4;
        break; // 1X
      case 8:
        delayIdx = 5;
        break; // X0
      }

      for (i = 0; i < inLen; ++i) {
        iTo = (iVcurr >> ((inLen - i - 1) << 1)) & 3,
        iFrom = (iVprev >> ((inLen - i - 1) << 1)) & 3;
        if (iTo != iFrom) {
          d = (iFrom == 1 || iTo == 0)
                  ? gateMgr[gid].dTable[MAX_DTUPLE * i + delayIdx + 6]
                  : gateMgr[gid].dTable[MAX_DTUPLE * i + delayIdx];

          delay = (d < delay) ? d : delay;
        }
      }

      currTime += delay;

      // Pop out late transtion
      while (currSize && oPort[currSize - 1].t >= currTime) {
        --currSize;
      }

      // Push back new output transtion
      if (!currSize || oPort[currSize - 1].v != oVcurr) {
        oPort[currSize] = {currTime, oVcurr};
        ++currSize;

        // Overflow
        if (currSize == maxSize) {
          overflow[gid] = true;
          oHisSize[gid] = currSize;
          return;
        }
      }
    }

    // Re-init cache
    iVprev = iVcurr;
    oVprev = oVcurr;
    iVcurr = 0;
    oVcurr = 0;
    currTime = -1;
    delay = -1;

    // Find next trans time
    for (i = 0; i < inLen; ++i) {
      currTime = (iTnext[i] < currTime) ? iTnext[i] : currTime;
    }
  }
  oPort[currSize] = {(tUnit)-1, valueZ};
  oHisSize[gid] = currSize + 1;
}

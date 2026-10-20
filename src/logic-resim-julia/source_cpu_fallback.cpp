#include "Simulator.h"
#include <climits>
#include <thread>
using std::thread;
// CPU-only object: original CPU methods use dumpOff, not GPU members.
Simulator::Simulator(tUnit off):dumpOff(off){}
void Simulator::simulate(Gate *g) {
  // Gate info
  vector<tHistory *> tList = g->getPortTable();
  const tTable *truthtable = g->getTruthTable();
  const dTable *delaytable = g->getDelayTable();
  size_t inLen = g->getInputSize(), delayIdx;

  // Simulation cache
  short iVcurr = 0, iVprev = 0;
  char oVcurr = tList[inLen]->getValue(0), oVprev = valueX, iFrom, iTo;
  size_t i;
  vector<size_t> iSize(inLen, 0), iHead(inLen, 0);
  vector<tUnit> iTnext(inLen, 0);
  tUnit currTime = -1, delay = -1, d;

  // Init input port trans len and first input pattern
  for (i = 0; i < inLen; i++) {
    iVprev = iVprev << 2;
    iVprev |= tList[i]->getValue(0);
    iSize[i] = tList[i]->size();
  }

  // Find first trans time
  for (i = 0; i < inLen; ++i) {
    if (1 < iSize[i]) {
      iTnext[i] = tList[i]->getTime(1);
      if (iTnext[i] < currTime)
        currTime = iTnext[i];
    }
  }

  // Simulation loop
  while (currTime != (tUnit)-1 && currTime <= dumpOff) {
    // Update newly coming input pattern and corresponding output
    for (i = 0; i < inLen; ++i) {
      if (iHead[i] + 1 < iSize[i]) {
        if (iTnext[i] == currTime)
          ++iHead[i];
      }

      iVcurr = iVcurr << 2;
      iVcurr |= tList[i]->getValue(iHead[i]);
    }

    oVcurr = truthtable->get(iVcurr);

    // Push back output port
    if (oVcurr != oVprev) {
      // Get minimum delay and next trans time
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
        if ((iTo) != (iFrom)) {
          d = (iFrom == 1 || iTo == 0)
                  ? (*(delaytable + (MAX_DTUPLE * i) + delayIdx + 6))
                  : // negedge
                  (*(delaytable + (MAX_DTUPLE * i) + delayIdx));

          if (d < delay)
            delay = d;
        }
      }
      currTime += delay;

      tList[inLen]->push_back(currTime, oVcurr);
    }

    // Update cache
    iVprev = iVcurr;
    oVprev = oVcurr;
    iVcurr = 0;
    oVcurr = 0;
    currTime = delay = -1;

    // Find next trans time
    for (i = 0; i < inLen; ++i) {
      if (iHead[i] + 1 < iSize[i]) {
        iTnext[i] = tList[i]->getTime(iHead[i] + 1);
        if (iTnext[i] < currTime)
          currTime = iTnext[i];
      }
    }
  }
}

void Simulator::simulateBlock(vector<Gate *> &gl) {
  vector<thread> threads;

  for (auto g : gl) {
    threads.push_back(thread(&Simulator::simulate, this, g));
  }

  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i].join();
  }
}

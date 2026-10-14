#include "Simulator.h"
#include <climits>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <algorithm>
using std::thread;

static void require_iris(int result) {
  if(result!=IRIS_SUCCESS) throw std::runtime_error("IRIS task failure");
}
void Simulator::addEvalMgr(char *table,size_t count) {
  eval.assign(table,table+TRUTH_SIZE*count);
}
void Simulator::addTrans(tHistory *trans,size_t& serial) {
  trans->push_end();
  histories.emplace(serial,std::vector<Event>(trans->getHis(),trans->getHis()+trans->size()));
}
Event *Simulator::getTrans(size_t& serial) { return histories.at(serial).data(); }
void Simulator::popTrans(size_t& serial) { histories.erase(serial); }
void Simulator::addGateMgr(gpu_GATE *table,size_t& count) {
  parallelSize=count;
  gates.assign(table,table+count);
}
void Simulator::simulateBlock(tHistory **wires,size_t *serial,size_t exp[][3]) {
  if(!parallelSize) return;
  std::vector<Event> input;
  std::unordered_map<Event*,size_t> locations;
  // Flatten all live input histories once. Offsets remain valid through retries.
  for(auto& item:histories) {
    locations.emplace(item.second.data(),input.size());
    input.insert(input.end(),item.second.begin(),item.second.end());
  }
  std::vector<FlatGate> flat(parallelSize);
  for(size_t i=0;i<parallelSize;++i) {
    flat[i].funcSer=gates[i].funcSer;
    std::memcpy(flat[i].dTable,gates[i].dTable,sizeof(flat[i].dTable));
    size_t ports=static_cast<unsigned char>(eval.at(flat[i].funcSer*TRUTH_SIZE+TRUTH_SIZE-1));
    for(size_t j=0;j<ports;++j) flat[i].iPort[j]=locations.at(gates[i].iPort[j]);
  }
  std::vector<size_t> maximum(parallelSize), offsets(parallelSize), sizes(parallelSize,0);
  size_t total=0;
  for(size_t i=0;i<parallelSize;++i) {
    maximum[i]=exp[i][0]; offsets[i]=total; total+=maximum[i];
  }
  std::vector<Event> output(total);
  std::unique_ptr<bool[]> overflow(new bool[parallelSize]);
  iris::DMem mg(flat.data(),flat.size()*sizeof(FlatGate));
  iris::DMem mi(input.data(),input.size()*sizeof(Event));
  iris::DMem me(eval.data(),eval.size());
  auto start=std::chrono::steady_clock::now();
  size_t loop=1;
  bool finish;
  do {
    finish=true;
    iris::DMem mo(output.data(),output.size()*sizeof(Event));
    iris::DMem mf(overflow.get(),parallelSize*sizeof(bool));
    iris::DMem mx(maximum.data(),parallelSize*sizeof(size_t));
    iris::DMem ms(sizes.data(),parallelSize*sizeof(size_t));
    iris::DMem mp(offsets.data(),parallelSize*sizeof(size_t));
    iris::Task task;
    void *params[]={&mf,&mg,&mo,&mp,&mi,&mx,&ms,&me,&parallelSize,&dumpOff};
    int info[]={iris_w,iris_r,iris_rw,iris_r,iris_r,iris_r,iris_rw,iris_r,sizeof(parallelSize),sizeof(dumpOff)};
    size_t global[]={((parallelSize+C_THREAD_LIMIT-1)/C_THREAD_LIMIT)*C_THREAD_LIMIT};
    size_t local[]={C_THREAD_LIMIT};
    require_iris(task.kernel("simulateParallel",1,nullptr,global,local,10,params,info));
    require_iris(task.flush_out(mf));
    require_iris(task.flush_out(ms));
    require_iris(task.flush_out(mo));
    require_iris(task.submit(iris_gpu,nullptr,1));
    std::vector<size_t> nextMax=maximum, nextOffsets(parallelSize);
    size_t nextTotal=0;
    for(size_t i=0;i<parallelSize;++i) {
      if(overflow[i]) { nextMax[i]+=exp[i][loop%3]; finish=false; }
      nextOffsets[i]=nextTotal; nextTotal+=nextMax[i];
    }
    if(!finish) {
      std::vector<Event> grown(nextTotal);
      for(size_t i=0;i<parallelSize;++i)
        std::copy_n(output.data()+offsets[i],maximum[i],grown.data()+nextOffsets[i]);
      output.swap(grown); maximum.swap(nextMax); offsets.swap(nextOffsets);
    }
    ++loop;
  } while(!finish);
  auto end=std::chrono::steady_clock::now();
  printf("Total simulation time of %zu loops: %f (s)\n",loop,std::chrono::duration<double>(end-start).count());
  for(size_t i=0;i<parallelSize;++i) {
    if(sizes[i]>maximum[i]) throw std::runtime_error("Invalid output history size");
    wires[i]->resize(sizes[i]);
    std::copy_n(output.data()+offsets[i],sizes[i],wires[i]->getHis());
    histories.emplace(serial[i],std::vector<Event>(output.data()+offsets[i],output.data()+offsets[i]+sizes[i]));
  }
}

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

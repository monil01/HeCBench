#include "vcdParser.h"
#include "interParser.h"
#include "FileReader.h"
#include "Simulator.h"
#include "VCD.h"
#include "SAIF.h"



using namespace std;

tUnit getDumpRange(char* c) {
   tUnit t = 0;
   while(*c != 0) {
      t = t*10 + (tUnit)(*c-'0'); ++c;
   }
   return t;
}

int
main(int argc, char** argv)
{
   if(argc!=6) { std::cerr<<"Usage: "<<argv[0]<<" intermediate.file input.vcd dumpOn dumpOff output.saif\n"; return 1; }
   iris::Platform platform;
   if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS) return 2;
   tUnit dumpOn = getDumpRange(argv[3]), dumpOff = getDumpRange(argv[4]);
   string outFileName(argv[5]);

   FileReader vcdFile(argv[2]);
   FileReader interFile(argv[1], vcdFile.getFileEnd());

   Simulator simulator(dumpOff);
   interParser iParser(&simulator);
   char* interStopPtr = iParser.parseHeader(interFile.getPtr());

   vcdParser vParser(iParser.getWireMgr(), dumpOff);
   vParser.parseAll(vcdFile.getPtr());
   #ifdef TIME
      cout << "Done vcd Parser!" << endl;
   #endif

   SAIFprinter sprinter(dumpOn, dumpOff, outFileName, vParser.getSAFIHeader(), iParser.getTransC(),vParser.getModCounter());
   iParser.parseInst(interStopPtr, &sprinter);

   if(platform.finalize()!=IRIS_SUCCESS) return 2;
   return 0;
}

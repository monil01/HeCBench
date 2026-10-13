#include "utils.hpp"
#include "driver.hpp"
#include <iris/iris.hpp>
int gpu_device=-1;

void proteinSampleRun(std::string refFile, std::string queFile, std::string out_file) {
  std::vector<std::string> G_sequencesA, G_sequencesB;
  std::string myInLine;
  unsigned long largestA = 0, largestB = 0, totSizeA = 0, totSizeB = 0;
  std::ifstream ref_file(refFile), quer_file(queFile);

  if (ref_file.is_open())
  {
    while (getline(ref_file, myInLine))
    {
      if (myInLine[0] == '>') {
        continue;
      } else {
        std::string seq = myInLine;
        G_sequencesA.push_back(seq);
        //std::cout<<"ref:"<<G_sequencesA.size()<<std::endl;
        totSizeA += seq.size();
        if(seq.size() > largestA)
        {
          largestA = seq.size();
        }
      }
    }
    ref_file.close();
  } else {
    std::cerr << "Error opening the reference file "
              << refFile << std::endl;
    return;
  }


  if (quer_file.is_open())
  {
    while (getline(quer_file, myInLine))
    {
      if (myInLine[0] == '>') {
        continue;
      } else {
        std::string seq = myInLine;
        G_sequencesB.push_back(seq);
        //std::cout<<"qeu:"<<G_sequencesB.size()<<std::endl;
        totSizeB += seq.size();
        if(seq.size() > largestB)
        {
          largestB = seq.size();
        }
      }
    }
    quer_file.close();
  } else {
    std::cerr << "Error opening the query file "
              << queFile << std::endl;
    return;
  }

  short scores_matrix[] = {
    4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0,-2,-1,0,-4,
    -1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3,-1,0,-1,-4,
    -2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3,3,0,-1,-4,
    -2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3,4,1,-1,-4,
    0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4,
    -1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2,0,3,-1,-4,
    -1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2,1,4,-1,-4,
    0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3,-1,-2,-1,-4,
    -2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3,0,0,-1,-4,
    -1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3,-3,-3,-1,-4,
    -1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1,-4,-3,-1,-4,
    -1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2,0,1,-1,-4,
    -1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1,-3,-1,-1,-4,
    -2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1,-3,-3,-1,-4,
    -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2,-2,-1,-2,-4,
    1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2,0,0,0,-4,
    0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0,-1,-1,0,-4,
    -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3,-4,-3,-2,-4,
    -2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1,-3,-2,-1,-4,
    0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4,-3,-2,-1,-4,
    -2,-1,3,4,-3,0,1,-1,0,-3,-4,0,-3,-3,-2,0,-1,-4,-3,-3,4,1,-1,-4,
    -1,0,0,1,-3,3,4,-2,0,-3,-3,1,-1,-3,-1,0,-1,-3,-2,-2,1,4,-1,-4,
    0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,0,0,-2,-1,-1,-1,-1,-1,-4,
    -4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,1}; // blosum 62

  kernel_driver_aa(out_file, G_sequencesB, G_sequencesA, scores_matrix, -6, -1);
}

int main(int argc, char* argv[])
{
  if(argc!=4)return 1;
  iris::Platform platform;
  if(platform.init(&argc,&argv,1)) return 2;
  int count=0,device=-1; iris_device_count(&count);
  for(int i=0;i<count;++i) {
    int backend=0,type=0; char name[256]={};
    iris_device_info(i,iris_backend,&backend,nullptr);
    iris_device_info(i,iris_type,&type,nullptr);
    iris_device_info(i,iris_name,name,nullptr);
    fprintf(stderr,"IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);
    if(backend==3 && type==256) device=i;
  }
  if(device<0) return 2;
  gpu_device=device;
  proteinSampleRun(argv[1],argv[2],argv[3]);
  platform.finalize();
  return 0;
}

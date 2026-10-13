#include "host_qualifiers.h"
#include <iris/iris.hpp>
#include <memory>
#include <cstring>
#include <chrono>
#include "logan_functions.cuh"
using namespace std;
using namespace chrono;
template<class T> int info(T&) {return sizeof(T);}
inline int info(iris::DMem*&) {return iris_rw;}
template<class T> void* ptr(T& a) {return const_cast<void*>(static_cast<const void*>(&a));}
inline void* ptr(iris::DMem*& a) {return a;}
template<class... A> void launch(int gpu,int blocks,int threads,A&... a) {
 void* args[]={ptr(a)...};int flags[]={info(a)...};
 size_t gws[]={size_t(blocks)*size_t(threads)},lws[]={size_t(threads)};
 iris::Task task;
 if(task.kernel("extendSeedLGappedXDropOneDirectionGlobal",1,nullptr,gws,lws,sizeof...(A),args,flags)!=IRIS_SUCCESS ||
 task.submit(gpu,nullptr,1)!=IRIS_SUCCESS) std::exit(1);
}

void extendSeedL(std::vector<SeedL> &seeds,
    ExtensionDirectionL direction,
    std::vector<std::string> &target,
    std::vector<std::string> &query,
    std::vector<ScoringSchemeL> &penalties,
    int const& XDrop,
    int const& kmer_length,
    int *res,
    int numAlignments,
    int ngpus,
    int n_threads
)
{

  if(scoreGapExtend(penalties[0]) >= 0){

    cout<<"Error: Logan does not support gap extension penalty >= 0\n";
    exit(-1);
  }
  if(scoreGapOpen(penalties[0]) >= 0){

    cout<<"Error: Logan does not support gap opening penalty >= 0\n";
    exit(-1);
  }
  int deviceCount = 0;
  if(iris_device_count(&deviceCount)!=IRIS_SUCCESS) std::exit(1);

  if (deviceCount == 0) {
    std::cout << "Error: no device found\n";
    return;
  }

  if (ngpus > deviceCount || ngpus > MAX_GPUS) {
    std::cout << "Error: the maximum number of devices allowed is "
              << std::min(deviceCount, MAX_GPUS) << std::endl;
    return;
  }

  //start measuring time
#ifdef ADAPTABLE
  n_threads = (XDrop/WARP_DIM + 1)* WARP_DIM;
  if(n_threads>1024)
    n_threads=1024;
#endif

  //declare streams
  std::vector<std::unique_ptr<iris::DMem>> owners[MAX_GPUS];
  std::vector<short> ant_host_l[MAX_GPUS],ant_host_r[MAX_GPUS];
  auto create=[&](int gpu,void* host,size_t bytes) {
    owners[gpu].emplace_back(std::make_unique<iris::DMem>(host,bytes));
    return owners[gpu].back().get();
  };

  // NB nSequences is correlated to the number of GPUs that we have
  int nSequences = numAlignments/ngpus;
  int nSequencesLast = nSequences+numAlignments%ngpus;

  //final result of the alignment
  int *scoreLeft = (int *)malloc(numAlignments * sizeof(int));
  int *scoreRight = (int *)malloc(numAlignments * sizeof(int));

  //create two sets of seeds
  //copy seeds
  vector<SeedL> seeds_r;
  vector<SeedL> seeds_l;
  seeds_r.reserve(numAlignments);

  for (size_t i=0; i<seeds.size(); i++){
    seeds_r.push_back(seeds[i]);  
  }

  //sequences offsets       
  vector<int> offsetLeftQ[MAX_GPUS];
  vector<int> offsetLeftT[MAX_GPUS];  
  vector<int> offsetRightQ[MAX_GPUS];  
  vector<int> offsetRightT[MAX_GPUS];

  //shared_mem_size per block per GPU
  int ant_len_left[MAX_GPUS];
  int ant_len_right[MAX_GPUS];

  //antidiag in case shared memory isn't enough
  iris::DMem* ant_l[MAX_GPUS], *ant_r[MAX_GPUS];

  //total lenght of the sequences
  int totalLengthQPref[MAX_GPUS];
  int totalLengthTPref[MAX_GPUS];
  int totalLengthQSuff[MAX_GPUS];
  int totalLengthTSuff[MAX_GPUS];

  //declare and allocate sequences prefixes and suffixes
  char *prefQ[MAX_GPUS], *prefT[MAX_GPUS];
  char *suffQ[MAX_GPUS], *suffT[MAX_GPUS];

  //declare GPU offsets
  iris::DMem* offsetLeftQ_d[MAX_GPUS], *offsetLeftT_d[MAX_GPUS];
  iris::DMem* offsetRightQ_d[MAX_GPUS], *offsetRightT_d[MAX_GPUS];

  //declare GPU results
  iris::DMem* scoreLeft_d[MAX_GPUS], *scoreRight_d[MAX_GPUS];

  //declare GPU seeds
  iris::DMem* seed_d_l[MAX_GPUS], *seed_d_r[MAX_GPUS];

  //declare prefixes and suffixes on the GPU  
  iris::DMem* prefQ_d[MAX_GPUS], *prefT_d[MAX_GPUS];
  iris::DMem* suffQ_d[MAX_GPUS], *suffT_d[MAX_GPUS];

  std::vector<double> pergpustime(ngpus);

  #pragma omp parallel for
  for(int i = 0; i < ngpus; i++){
    int dim = nSequences;
    if(i==ngpus-1)
      dim = nSequencesLast;
    //compute offsets and shared memory per block
    int MYTHREAD = omp_get_thread_num();
    auto start_setup_ithread = NOW;
    ant_len_left[i]=0;
    ant_len_right[i]=0;
    for(int j = 0; j < dim; j++){

      offsetLeftQ[i].push_back(getBeginPositionV(seeds[j+i*nSequences]));
      offsetLeftT[i].push_back(getBeginPositionH(seeds[j+i*nSequences]));
      ant_len_left[i] = std::max(std::min(offsetLeftQ[i][j],offsetLeftT[i][j]), ant_len_left[i]);

      offsetRightQ[i].push_back(query[j+i*nSequences].size()-getEndPositionV(seeds[j+i*nSequences]));
      offsetRightT[i].push_back(target[j+i*nSequences].size()-getEndPositionH(seeds[j+i*nSequences]));
      ant_len_right[i] = std::max(std::min(offsetRightQ[i][j], offsetRightT[i][j]), ant_len_right[i]);
    }

    //compute antidiagonal offsets
    partial_sum(offsetLeftQ[i].begin(),offsetLeftQ[i].end(),offsetLeftQ[i].begin());  
    partial_sum(offsetLeftT[i].begin(),offsetLeftT[i].end(),offsetLeftT[i].begin());
    partial_sum(offsetRightQ[i].begin(),offsetRightQ[i].end(),offsetRightQ[i].begin());
    partial_sum(offsetRightT[i].begin(),offsetRightT[i].end(),offsetRightT[i].begin());
    //set total length of the sequences
    totalLengthQPref[i] = offsetLeftQ[i][dim-1];
    totalLengthTPref[i] = offsetLeftT[i][dim-1];
    totalLengthQSuff[i] = offsetRightQ[i][dim-1];
    totalLengthTSuff[i] = offsetRightT[i][dim-1];
    //allocate sequences prefix and suffix on the CPU
    prefQ[i] = (char*)malloc(sizeof(char)*totalLengthQPref[i]);
    prefT[i] = (char*)malloc(sizeof(char)*totalLengthTPref[i]);
    suffQ[i] = (char*)malloc(sizeof(char)*totalLengthQSuff[i]);
    suffT[i] = (char*)malloc(sizeof(char)*totalLengthTSuff[i]);
    //generate prefix and suffix on the CPU
    reverse_copy(query[0+i*nSequences].c_str(),query[0+i*nSequences].c_str()+offsetLeftQ[i][0],prefQ[i]);

    memcpy(prefT[i], target[0+i*nSequences].c_str(), offsetLeftT[i][0]);
    memcpy(suffQ[i], query[0+i*nSequences].c_str()+getEndPositionV(seeds[0+i*nSequences]), offsetRightQ[i][0]);
    reverse_copy(target[0+i*nSequences].c_str()+getEndPositionH(seeds[0+i*nSequences]),target[0+i*nSequences].c_str()+getEndPositionH(seeds[0+i*nSequences])+offsetRightT[i][0],suffT[i]);

    for(int j = 1; j<dim; j++){
      char *seqptr = prefQ[i] + offsetLeftQ[i][j-1];
      reverse_copy(query[j+i*nSequences].c_str(),query[j+i*nSequences].c_str()+(offsetLeftQ[i][j]-offsetLeftQ[i][j-1]),seqptr);

      seqptr = prefT[i] + offsetLeftT[i][j-1];
      memcpy(seqptr, target[j+i*nSequences].c_str(), offsetLeftT[i][j]-offsetLeftT[i][j-1]);
      seqptr = suffQ[i] + offsetRightQ[i][j-1];
      memcpy(seqptr, query[j+i*nSequences].c_str()+getEndPositionV(seeds[j+i*nSequences]), offsetRightQ[i][j]-offsetRightQ[i][j-1]);
      seqptr = suffT[i] + offsetRightT[i][j-1];
      reverse_copy(target[j+i*nSequences].c_str()+getEndPositionH(seeds[j+i*nSequences]),target[j+i*nSequences].c_str()+getEndPositionH(seeds[j+i*nSequences])+(offsetRightT[i][j]-offsetRightT[i][j-1]),seqptr);

    }
    auto end_setup_ithread = NOW;
    duration<double> setup_ithread = end_setup_ithread - start_setup_ithread;
    pergpustime[MYTHREAD] = setup_ithread.count();
  }

  #pragma omp parallel for
  for(int i = 0; i < ngpus; i++)
  {
    int dim = nSequences;
    if(i==ngpus-1)
      dim = nSequencesLast;
    //set gpu device
    //create streams
    //allocate antidiagonals on the GPU
    ant_host_l[i].resize((sizeof(short)*ant_len_left[i]*3*dim)/sizeof(short));
    ant_l[i]=create(i,ant_host_l[i].data(),sizeof(short)*ant_len_left[i]*3*dim);
    ant_host_r[i].resize((sizeof(short)*ant_len_right[i]*3*dim)/sizeof(short));
    ant_r[i]=create(i,ant_host_r[i].data(),sizeof(short)*ant_len_right[i]*3*dim);
    //allocate offsets on the GPU
    //allocate results on the GPU
    scoreLeft_d[i]=create(i,scoreLeft+i*nSequences,dim*sizeof(int));
    scoreRight_d[i]=create(i,scoreRight+i*nSequences,dim*sizeof(int));
    //allocate seeds on the GPU
    //allocate sequences on the GPU
    //copy seeds to the GPU
    seed_d_l[i]=create(i,&seeds[0]+i*nSequences,dim*sizeof(SeedL));
    seed_d_r[i]=create(i,&seeds_r[0]+i*nSequences,dim*sizeof(SeedL));
    //copy offsets to the GPU
    offsetLeftQ_d[i]=create(i,&offsetLeftQ[i][0],dim*sizeof(int));
    offsetLeftT_d[i]=create(i,&offsetLeftT[i][0],dim*sizeof(int));
    offsetRightQ_d[i]=create(i,&offsetRightQ[i][0],dim*sizeof(int));
    offsetRightT_d[i]=create(i,&offsetRightT[i][0],dim*sizeof(int));
    //copy sequences to the GPU
    prefQ_d[i]=create(i,prefQ[i],totalLengthQPref[i]*sizeof(char));
    prefT_d[i]=create(i,prefT[i],totalLengthTPref[i]*sizeof(char));
    suffQ_d[i]=create(i,suffQ[i],totalLengthQSuff[i]*sizeof(char));
    suffT_d[i]=create(i,suffT[i],totalLengthTSuff[i]*sizeof(char));
  }

  auto start_c = NOW;

  //execute kernels
  #pragma omp parallel for
  for(int i = 0; i<ngpus;i++)
  {

    int dim = nSequences;
    if(i==ngpus-1)
      dim = nSequencesLast;

    ExtensionDirectionL direction_l=EXTEND_LEFTL;
    launch(i,dim,n_threads,seed_d_l[i], prefQ_d[i], prefT_d[i], direction_l, XDrop,
      scoreLeft_d[i], offsetLeftQ_d[i], offsetLeftT_d[i],
      ant_len_left[i], ant_l[i], n_threads);

    ExtensionDirectionL direction_r=EXTEND_RIGHTL;
    launch(i,dim,n_threads,seed_d_r[i], suffQ_d[i], suffT_d[i], direction_r, XDrop,
      scoreRight_d[i], offsetRightQ_d[i], offsetRightT_d[i],
      ant_len_right[i], ant_r[i], n_threads);
  }

  #pragma omp parallel for
  for(int i = 0; i < ngpus; i++)
  {
    int dim = nSequences;
    if(i==ngpus-1)
      dim = nSequencesLast;
    if(scoreLeft_d[i]->fetch(scoreLeft+i*nSequences)!=IRIS_SUCCESS) std::exit(1);
    if(seed_d_l[i]->fetch(&seeds[0]+i*nSequences)!=IRIS_SUCCESS) std::exit(1);
    if(scoreRight_d[i]->fetch(scoreRight+i*nSequences)!=IRIS_SUCCESS) std::exit(1);
    if(seed_d_r[i]->fetch(&seeds_r[0]+i*nSequences)!=IRIS_SUCCESS) std::exit(1);
  }

  #pragma omp parallel for
  for(int i = 0; i < ngpus; i++)
  {
  }

  auto end_c = NOW;
  duration<double> compute = end_c-start_c;
  std::cout << "Device only time [seconds]:\t" << compute.count() << std::endl;


  #pragma omp parallel for
  for(int i = 0; i < ngpus; i++){

    free(prefQ[i]);
    free(prefT[i]);
    free(suffQ[i]);
    free(suffT[i]);
  }

  for(int i = 0; i < numAlignments; i++){
    res[i] = scoreLeft[i]+scoreRight[i]+kmer_length;
    setEndPositionH(seeds[i], getEndPositionH(seeds_r[i]));    
    setEndPositionV(seeds[i], getEndPositionV(seeds_r[i])); 
    std::cout << res[i] << std::endl;
  }

  free(scoreLeft);
  free(scoreRight);
}


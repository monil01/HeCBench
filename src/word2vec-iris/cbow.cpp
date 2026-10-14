#include <iris/iris.hpp>
#include <vector>
#include <mutex>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "cbow.h"
extern real *syn0;
extern int *table, *sen;
extern int vocab_size,layer1_size,layer1_size_aligned,negative,window,table_size;
static iris::DMem *m0,*m1,*ms,*mr,*mt,*me,*mx;
static std::vector<real> weights,negative_weights,exptable,scratch;
static std::vector<unsigned int> random_values;
static int numBlock,scratch_stride;
static std::mutex submission_mutex;
void initializeGPU() {
  weights.assign(syn0,syn0+vocab_size*layer1_size_aligned);
  negative_weights.resize(vocab_size*layer1_size_aligned,0);
  exptable.resize(EXP_TABLE_SIZE);
  for(int i=0;i<EXP_TABLE_SIZE;++i) {
    exptable[i]=exp((i/(real)EXP_TABLE_SIZE*2-1)*MAX_EXP);
    exptable[i]=exptable[i]/(exptable[i]+1);
  }
  sen=(int*)calloc(MAX_SENTENCE_NUM*MAX_SENTENCE_LENGTH+MAX_SENTENCE_NUM,sizeof(int));
  random_values.resize(MAX_SENTENCE_LENGTH);
  srand(123);
  for(int i=0;i<MAX_SENTENCE_LENGTH;++i) random_values[i]=(unsigned int)rand();
  numBlock=MAX_SENTENCE_LENGTH/(BLOCK_SIZE/THREADS_PER_WORD)+1;
  if (const char* bound=getenv("HECBENCH_WORD2VEC_BLOCKS")) {
    int requested=atoi(bound);
    if(requested>0 && requested<numBlock) numBlock=requested;
  }
  scratch_stride=BLOCK_SIZE+(BLOCK_SIZE/THREADS_PER_WORD)*layer1_size_aligned*2;
  scratch.resize(numBlock*scratch_stride);
  m0=new iris::DMem(weights.data(),weights.size()*sizeof(real));
  m1=new iris::DMem(negative_weights.data(),negative_weights.size()*sizeof(real));
  ms=new iris::DMem(sen,(MAX_SENTENCE_NUM*MAX_SENTENCE_LENGTH+MAX_SENTENCE_NUM)*sizeof(int));
  mr=new iris::DMem(random_values.data(),random_values.size()*sizeof(unsigned int));
  mt=new iris::DMem(table,table_size*sizeof(int));
  me=new iris::DMem(exptable.data(),exptable.size()*sizeof(real));
  mx=new iris::DMem(scratch.data(),scratch.size()*sizeof(real));
}
void TrainGPU(int sentence_num) {
  std::lock_guard<std::mutex> guard(submission_mutex);
  ms->update(sen);
  iris::Task task;
  void* params[]={&sentence_num,&layer1_size,&layer1_size_aligned,&window,&negative,&table_size,&vocab_size,ms,mt,m0,m1,mr,me,mx,&scratch_stride};
  int info[]={sizeof(sentence_num),sizeof(layer1_size),sizeof(layer1_size_aligned),sizeof(window),sizeof(negative),sizeof(table_size),sizeof(vocab_size),iris_r,iris_r,iris_rw,iris_rw,iris_rw,iris_r,iris_w,sizeof(scratch_stride)};
  size_t global[]={static_cast<size_t>(numBlock)*BLOCK_SIZE},local[]={BLOCK_SIZE};
  task.kernel("device_cbow",1,nullptr,global,local,15,params,info);
  task.submit(iris_gpu,nullptr,1);
}
void GetResultFromGPU() {
  std::lock_guard<std::mutex> guard(submission_mutex);
  iris::Task task;task.flush_out(*m0);task.submit(iris_gpu,nullptr,1);
  memcpy(syn0,weights.data(),weights.size()*sizeof(real));
}
void cleanUpGPU() {
  delete m0;delete m1;delete ms;delete mr;delete mt;delete me;delete mx;
  free(sen);
}

#!/usr/bin/env python3
"""Generate owned instrumentation copies; preserve original files byte-identically."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;R=P.parents[1];S=R/'word2vec-cuda'
def replace(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def main():
 out=Path('/tmp/w2vfull');out.mkdir(exist_ok=True)
 corpus=(' '.join(f'w{i:04}' for i in range(1,1025))+' ')*512
 corpus=corpus.rstrip()+'\n';(out/'corpus.txt').write_text(corpus)
 s=(S/'word2vec.cu').read_text();s=replace(s,'int main(int argc, char **argv)', 'int source_main(int argc, char **argv)')
 s=replace(s,'  TrainModel();','  TrainModel();\n  for(int q=0;q<vocab_size;q++){free(vocab[q].word);free(vocab[q].code);free(vocab[q].point);}\n  free(syn0);')
 (P/'observed_host.inc').write_text(s)
 s=(S/'cbow.cu').read_text();s=replace(s,'unsigned int *__restrict__ d_random)','unsigned int *__restrict__ d_random, unsigned int* trace, int slot)')
 a='  if (sentence_position < MAX_SENTENCE_LENGTH) {'
 b='''  unsigned int* rec=trace+(size_t)slot*TRACE_WORDS;
  if(blockIdx.x==0){
    if(threadIdx.x==0){rec[0]=atomicAdd(trace_execution_counter,1u);rec[1]=sentence_num;}
    for(int q=threadIdx.x;q<SEN_WORDS;q+=blockDim.x)rec[2+q]=((unsigned int*)d_sen)[q];
  }
'''+a
 s=replace(s,a,b)
 a='    unsigned int next_random = d_random[sentence_position];'
 s=replace(s,a,a+'\n    if(idInWarp==0)rec[2+SEN_WORDS+sentence_position]=next_random;')
 a='    if (idInWarp == 0 ) d_random[sentence_position] = next_random;'
 s=replace(s,a,a+'\n    if(idInWarp==0)rec[2+SEN_WORDS+MAX_SENTENCE_LENGTH+sentence_position]=next_random;')
 s=replace(s,'void cleanUpGPU(){','void cleanUpGPU(){\n  observe_final();')
 a='  numBlock = MAX_SENTENCE_LENGTH / (BLOCK_SIZE/THREADS_PER_WORD) + 1;'
 s=replace(s,a,'  memset(sen,0,SEN_WORDS*4);\n  cudaCheck(cudaMemset(d_sen,0,SEN_WORDS*4));\n'+a)
 a='  shared_mem_usage = (BLOCK_SIZE + (BLOCK_SIZE/THREADS_PER_WORD) * layer1_size_aligned * 2) * sizeof(real);'
 s=replace(s,a,a+'\n  observe_initial();')
 a='void TrainGPU(int sentence_num) {\n  TransferDataToGPU();'
 s=replace(s,a,'void TrainGPU(int sentence_num) {\n  int slot=trace_submitted.fetch_add(1);\n  if(slot>=TRACE_CAP){fprintf(stderr,"TRACE_OVERFLOW\\n");abort();}\n  TransferDataToGPU();')
 a='negative, table_size, vocab_size, d_sen, d_table, d_syn0, d_syn1neg, d_random);'
 s=replace(s,a,'negative, table_size, vocab_size, d_sen, d_table, d_syn0, d_syn1neg, d_random, trace_data, slot);')
 (P/'observed_gpu.inc').write_text(s)
 declared=json.loads((P/'protocol.json').read_text())
 assert declared['corpus']['sha256']==hashlib.sha256(corpus.encode()).hexdigest()
 for path,sha in declared['source_hashes'].items():
  assert hashlib.sha256((R.parent/path).read_bytes()).hexdigest()==sha,path

if __name__=='__main__':main()

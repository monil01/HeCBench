#include <iris/iris.hpp>
#include <stdexcept>
#include <cstring>
#include "kernel.hpp"
#include "utils.hpp"
extern int gpu_device;

void kernel_driver_aa(std::string filename,
    std::vector<std::string> &reads, 
    std::vector<std::string> &contigs,
    short h_scoring_matrix[], 
    short openGap, 
    short extendGap)
{

  unsigned maxContigSize = getMaxLength(contigs);
  unsigned maxReadSize = getMaxLength(reads);
  unsigned totalAlignments = contigs.size(); 

  //std::cout <<"max contig:"<<maxContigSize<<std::endl;
  //std::cout <<"max read:"<<maxReadSize<<std::endl;
  //std::cout <<"total aligns:"<<totalAlignments<<std::endl;

  short* h_ref_begin    = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_ref_end      = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_query_begin  = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_query_end    = (short*) malloc (sizeof(short) * totalAlignments);
  short* h_top_scores   = (short*) malloc (sizeof(short) * totalAlignments);
  unsigned* h_offsetA = (unsigned*) malloc (sizeof(unsigned) * totalAlignments);
  unsigned* h_offsetB = (unsigned*) malloc (sizeof(unsigned) * totalAlignments);
  char* h_strA = (char*) malloc (maxContigSize * totalAlignments);
  char* h_strB = (char*) malloc (maxContigSize * totalAlignments);

  short h_encoding_matrix[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    23,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,20,4,3,6,
    13,7,8,9,0,11,10,12,2,0,14,5,
    1,15,16,0,19,17,22,18,21};

  // total number of iterations
  int its = (totalAlignments>20000)?(ceil((float)totalAlignments/20000)):1;
  unsigned NBLOCKS    = totalAlignments;
  unsigned leftOvers    = NBLOCKS % its;
  unsigned stringsPerIt = NBLOCKS / its;
  unsigned maxAlignments = stringsPerIt + leftOvers;

  // Flushes target batch staging, never already-completed artifact rows.
  std::vector<short> batch_ref_start(maxAlignments),batch_ref_end(maxAlignments),
    batch_query_start(maxAlignments),batch_query_end(maxAlignments),batch_scores(maxAlignments);
  iris::DMem d_ref_start(batch_ref_start.data(),maxAlignments * sizeof(short));

  iris::DMem d_ref_end(batch_ref_end.data(),maxAlignments * sizeof(short));

  iris::DMem d_query_start(batch_query_start.data(),maxAlignments * sizeof(short));

  iris::DMem d_query_end(batch_query_end.data(),maxAlignments * sizeof(short));

  iris::DMem d_scores(batch_scores.data(),maxAlignments * sizeof(short));

  iris::DMem d_offset_ref(h_offsetA,maxAlignments * sizeof(unsigned));

  iris::DMem d_offset_query(h_offsetB,maxAlignments * sizeof(unsigned));

  iris::DMem d_strA(h_strA,maxContigSize * maxAlignments * sizeof(char));

  iris::DMem d_strB(h_strB,maxReadSize * maxAlignments * sizeof(char));

  iris::DMem d_encoding_matrix(h_encoding_matrix,ENCOD_MAT_SIZE * sizeof(short));
  d_encoding_matrix.update(h_encoding_matrix);

  iris::DMem d_scoring_matrix(h_scoring_matrix,SCORE_MAT_SIZE * sizeof(short));
  d_scoring_matrix.update(h_scoring_matrix);

  short* ref_begin    = h_ref_begin;
  short* ref_end      = h_ref_end;  
  short* query_begin  = h_query_begin;
  short* query_end    = h_query_end;
  short* top_scores   = h_top_scores;   

  std::cout<<"Number of loop iterations: " << its << std::endl;

  auto fetch=[&](iris::DMem& mem,void* out,size_t bytes){
    iris::Task task;task.flush_out(mem);
    if(task.submit(gpu_device,nullptr,1)||mem.fetch(out,bytes))throw std::runtime_error("BSW flush failed");
  };
  auto launch=[&](bool inverse,size_t blocks,size_t threads){
    if(!threads||threads>1024)throw std::runtime_error("BSW block size outside original hardware bound");
    iris::Task task;
    void* p[]={&inverse,&d_strA,&d_strB,&d_offset_ref,&d_offset_query,&d_ref_start,&d_ref_end,
      &d_query_start,&d_query_end,&d_scores,&openGap,&extendGap,&d_scoring_matrix,&d_encoding_matrix};
    int info[]={sizeof(inverse),iris_r,iris_r,iris_r,iris_r,iris_rw,iris_rw,iris_rw,iris_rw,iris_rw,
      sizeof(openGap),sizeof(extendGap),iris_r,iris_r};
    size_t g[]={blocks*threads},l[]={threads};
    if(task.kernel("sequence_aa_kernel",1,nullptr,g,l,14,p,info)||task.submit(gpu_device,nullptr,1))throw std::runtime_error("BSW kernel failed");
  };
  auto start = NOW;

  for (int perGPUIts = 0; perGPUIts < its; perGPUIts++)
  {
    int  blocksLaunched;
    std::vector<std::string>::const_iterator beginAVec;
    std::vector<std::string>::const_iterator endAVec;
    std::vector<std::string>::const_iterator beginBVec;
    std::vector<std::string>::const_iterator endBVec;

    beginAVec = contigs.begin() + perGPUIts * stringsPerIt;
    beginBVec = reads.begin() + (perGPUIts * stringsPerIt);

    if(perGPUIts == its - 1)
    {
      endAVec = contigs.begin() + (perGPUIts + 1) * stringsPerIt + leftOvers;
      endBVec = reads.begin() + ((perGPUIts + 1) * stringsPerIt) + leftOvers;
      blocksLaunched = stringsPerIt + leftOvers;
    }
    else
    {
      endAVec = contigs.begin() + (perGPUIts + 1) * stringsPerIt;
      endBVec = reads.begin() +  (perGPUIts + 1) * stringsPerIt;
      blocksLaunched = stringsPerIt;
    }

    std::vector<std::string> sequencesA(beginAVec, endAVec);
    std::vector<std::string> sequencesB(beginBVec, endBVec);
    unsigned running_sum = 0;
    int sequences_per_stream = blocksLaunched;

    for (unsigned int i = 0; i < sequencesA.size(); i++)
    {
      running_sum += sequencesA[i].size();
      h_offsetA[i] = running_sum;//sequencesA[i].size();
      // std::cout << "offsetA_h: " << h_offsetA[i] << std::endl;
    }
    unsigned totalLengthA = h_offsetA[sequencesA.size() - 1];
    // std::cout << "totalLengthA: " << totalLengthA << std::endl;

    running_sum = 0;
    for (unsigned int i = 0; i < sequencesB.size(); i++)
    {
      running_sum += sequencesB[i].size();
      h_offsetB[i] = running_sum; //sequencesB[i].size();
      // std::cout << "offsetB_h: " << h_offsetB[i] << std::endl;
    }
    unsigned totalLengthB = h_offsetB[sequencesB.size() - 1];
    // std::cout << "totalLengthB: " << totalLengthB << std::endl;

    unsigned offsetSumA = 0;
    unsigned offsetSumB = 0;

    for (unsigned int i = 0; i < sequencesA.size(); i++)
    {
      char* seqptrA = h_strA + offsetSumA;
      memcpy(seqptrA, sequencesA[i].c_str(), sequencesA[i].size());
      char* seqptrB = h_strB + offsetSumB;
      memcpy(seqptrB, sequencesB[i].c_str(), sequencesB[i].size());
      offsetSumA += sequencesA[i].size();
      offsetSumB += sequencesB[i].size();
    }

    d_offset_ref.update(h_offsetA);

    d_offset_query.update(h_offsetB);

    d_strA.update(h_strA);

    d_strB.update(h_strB);

    unsigned minSize = (maxReadSize < maxContigSize) ? maxReadSize : maxContigSize;
    unsigned totShmem = 3 * (minSize + 1) * sizeof(short);
    unsigned alignmentPad = 4 + (4 - totShmem % 4);
    size_t   ShmemBytes = totShmem + alignmentPad;
    printf("Shared memory bytes: %lu\n", ShmemBytes);
    printf("sequences per stream (CUDA grid size): %d\n", sequences_per_stream);
    printf("minSize (CUDA block size): %d\n", minSize);


    launch(false,sequences_per_stream,minSize);

    // copyin back end index so that we can find new min
    fetch(d_ref_end,ref_end,sizeof(short) * sequences_per_stream);

    fetch(d_query_end,query_end,sizeof(short) * sequences_per_stream);


    // find the new largest of smaller lengths
    int newMin = get_new_min_length(ref_end, query_end, blocksLaunched);


    launch(true,sequences_per_stream,newMin);

    fetch(d_ref_start,ref_begin,sizeof(short) * sequences_per_stream);

    fetch(d_query_start,query_begin,sizeof(short) * sequences_per_stream);

    fetch(d_scores,top_scores,sizeof(short) * sequences_per_stream);

    ref_begin += stringsPerIt;
    query_begin += stringsPerIt;
    ref_end += stringsPerIt;
    query_end += stringsPerIt;
    top_scores += stringsPerIt;

  }  // iterations end here

  auto end  = NOW;


  std::chrono::duration<double> diff = end - start;
  std::cout << "Total Alignments:" << totalAlignments << "\n" 
            << "Max Reference Size:" << maxContigSize << "\n"
            << "Max Query Size:"<< maxReadSize << "\n" 
            << "Total loop iteration time (seconds):"<< diff.count() << "\n";

  std::ofstream results_file(filename);
  if (results_file.is_open()) {

    for(unsigned int k = 0; k < reads.size(); k++){
      results_file << h_top_scores[k] <<"\t"
                   << h_ref_begin[k] <<"\t"
                   << h_ref_end[k] - 1 <<"\t"
                   << h_query_begin[k] <<"\t"
                   << h_query_end[k] - 1
                   << std::endl;
    }
    results_file.flush();
    results_file.close();
  } else {
    std::cerr << "Error opening the result file "
              << filename << std::endl;
  }

  long long int total_cells = 0;
  for (unsigned int l = 0; l < reads.size(); l++) {
    total_cells += reads.at(l).size()*contigs.at(l).size();
  }

  std::cout << "Total Cells:"<<total_cells<<std::endl;

  free(h_top_scores);
  free(h_ref_begin);
  free(h_ref_end);
  free(h_query_begin);
  free(h_query_end);
  free(h_offsetA);
  free(h_offsetB);
  free(h_strA);
  free(h_strB);
}// end of amino acids kernel

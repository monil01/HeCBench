// Modified for IRIS DMem masking, 2026-10-13; AGPLv3-or-later notices retained.
/****
  DIAMOND protein aligner
  Copyright (C) 2013-2017 Benjamin Buchfink <buchfink@gmail.com>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ****/
#include "iris_support.h"
#include "../diamond-sycl/src/basic/masking.h"


#define SEQ_LEN 33

struct DiamondIrisSession {
  iris::Platform platform;
  int device;
  DiamondIrisSession() {
    iris_check(platform.init(nullptr,nullptr,true),"IRIS init");
    device=iris_hip_device();
  }
  ~DiamondIrisSession() { platform.finalize(); }
};
static DiamondIrisSession* diamond_iris_session = nullptr;
int diamond_application_main(int, const char**);
int main(int argc, const char** argv) {
  const int result = diamond_application_main(argc, argv);
  // Finalize while application globals and HIP library state are still alive.
  delete diamond_iris_session;
  diamond_iris_session = nullptr;
  return result;
}
static void dump_masking(const unsigned char* seq,size_t bytes) {
  if(!std::getenv("HECBENCH_DUMP")) return;
  static unsigned index=0;
  char filename[80];
  snprintf(filename,sizeof(filename),"diamond_masked_%03u.bin",index++);
  FILE* fp=fopen(filename,"wb");
  if(!fp || fwrite(seq,1,bytes,fp)!=bytes) throw std::runtime_error("Mask artifact write failed");
  fclose(fp);
}
auto_ptr<Masking> Masking::instance;
const uint8_t Masking::bit_mask = 128;

Masking::Masking(const Score_matrix &score_matrix)
{
  const double lambda = score_matrix.lambda(); // 0.324032
  for (unsigned i = 0; i < size; ++i) {
    mask_table_x_[i] = value_traits.mask_char;
    mask_table_bit_[i] = (uint8_t)i | bit_mask;
    for (unsigned j = 0; j < size; ++j)
      if (i < value_traits.alphabet_size && j < value_traits.alphabet_size)
        likelihoodRatioMatrix_[i][j] = exp(lambda * score_matrix(i, j));
  }
  std::copy(likelihoodRatioMatrix_, likelihoodRatioMatrix_ + size, probMatrixPointers_);
  int firstGapCost = score_matrix.gap_extend() + score_matrix.gap_open();
  firstGapProb_ = exp(-lambda * firstGapCost);
  otherGapProb_ = exp(-lambda * score_matrix.gap_extend());
  firstGapProb_ /= (1 - otherGapProb_);
}

void Masking::operator()(Letter *seq, size_t len) const
{

  tantan::maskSequences((tantan::uchar*)seq, (tantan::uchar*)(seq + len), 50,
      (tantan::const_double_ptr*)probMatrixPointers_,
      0.005, 0.05,
      0.9,
      0, 0,
      0.5, (const tantan::uchar*)mask_table_x_);
}

unsigned char* Masking::call_opt(Sequence_set &seqs) const
{
  const int n = seqs.get_length();
  int total = 0;
  for (int i=0; i < n; i++)
    total += seqs.length(i);

  printf("There are %d sequences and the total sequence length is %d\n", n, total);
  unsigned char *seqs_device = NULL;
  // IRIS pins host ranges: isolate staging buffers on complete host pages.
  posix_memalign((void**)&seqs_device, 4096, (static_cast<size_t>(total)+4095)&~size_t(4095));

  unsigned char *p = seqs_device;
  for (int i=0; i < n; i++) {
    memcpy(p, seqs.ptr(i), seqs.length(i));
    p += seqs.length(i);
  }

  double *probMat_device = NULL;
  posix_memalign((void**)&probMat_device, 4096, (size*size*sizeof(double)+4095)&~size_t(4095));
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      probMat_device[i*size+j] = probMatrixPointers_[i][j];

  unsigned char *mask_table_device = NULL;
  posix_memalign((void**)&mask_table_device, 4096, (size*sizeof(unsigned char)+4095)&~size_t(4095));
  for (int i = 0; i < size; i++)
    mask_table_device[i] = mask_table_x_[i];

  int len = 33;

  printf("Timing the mask sequences on device...\n");
  Timer t;
  t.start();

  const int size = len;
  const int maxRepeatOffset = 50;
  const double repeatProb = 0.005;
  const double repeatEndProb = 0.05;
  const double repeatOffsetProbDecay = 0.9;
  const double firstGapProb = 0;
  const double otherGapProb = 0;
  const double minMaskProb = 0.5;
  const int seqs_len = n;

  if (!diamond_iris_session) diamond_iris_session = new DiamondIrisSession;
  DiamondIrisSession& session = *diamond_iris_session;
  {
    iris::DMem d_seqs(seqs_device,total);
    iris::DMem d_maskTable(mask_table_device,size);
    iris::DMem d_probMat(probMat_device,sizeof(double)*size*size);
    size_t lws=128,gws=((seqs_len+128)/128)*lws;
    iris::Task task;
    void* args[]={&d_seqs,&d_probMat,&d_maskTable,(void*)&size,(void*)&maxRepeatOffset,
                  (void*)&repeatProb,(void*)&repeatEndProb,(void*)&repeatOffsetProbDecay,
                  (void*)&firstGapProb,(void*)&otherGapProb,(void*)&minMaskProb,(void*)&seqs_len};
    int info[]={iris_rw,iris_r,iris_r,sizeof(size),sizeof(maxRepeatOffset),
                sizeof(repeatProb),sizeof(repeatEndProb),sizeof(repeatOffsetProbDecay),
                sizeof(firstGapProb),sizeof(otherGapProb),sizeof(minMaskProb),sizeof(seqs_len)};
    iris_check(task.kernel("maskSequences",1,nullptr,&gws,&lws,12,args,info),"mask kernel");
    iris_check(task.flush_out(d_seqs),"mask flush");
    iris_check(task.submit(session.device,nullptr,true),"HIP masking task");
  }
  dump_masking(seqs_device,total);

  message_stream << "Total time (maskSequences) on the device = " <<
    t.getElapsedTimeInMicroSec() / 1e6 << " s" << std::endl;

  free(probMat_device);
  free(mask_table_device);
  return seqs_device;
}

void Masking::call_opt(Letter *seq, size_t len) const
{
  // CPU
  tantale::maskSequences((tantan::uchar*)seq, (tantan::uchar*)(seq + len), 50,
      (tantan::const_double_ptr*)probMatrixPointers_,
      0.005, 0.05,
      0.9,
      0, 0,
      0.5, (const tantan::uchar*)mask_table_x_);
}



void Masking::mask_bit(Letter *seq, size_t len) const
{

  tantan::maskSequences((tantan::uchar*)seq, (tantan::uchar*)(seq + len), 50,
      (tantan::const_double_ptr*)probMatrixPointers_,
      0.005, 0.05,
      0.9,
      0, 0,
      0.5,		(const tantan::uchar*)mask_table_bit_);
}

void Masking::bit_to_hard_mask(Letter *seq, size_t len, size_t &n) const
{
  for (size_t i = 0; i < len; ++i)
    if (seq[i] & bit_mask) {
      seq[i] = value_traits.mask_char;
      ++n;
    }
}

void Masking::remove_bit_mask(Letter *seq, size_t len) const
{
  for (size_t i = 0; i < len; ++i)
    if (seq[i] & bit_mask)
      seq[i] &= ~bit_mask;
}

void mask_worker(Atomic<size_t> *next, Sequence_set *seqs, const Masking *masking, bool hard_mask)
{
  size_t i;
  //int cnt = 0;

  while ((i = (*next)++) < seqs->get_length())
  {
    if (hard_mask)
      //masking->operator()(seqs->ptr(i), seqs->length(i));
      masking->call_opt(seqs->ptr(i), seqs->length(i));
    else
      masking->mask_bit(seqs->ptr(i), seqs->length(i));
    //cnt++;
    //if (cnt == 2) break;
  }
}

void mask_seqs(Sequence_set &seqs, const Masking &masking, bool hard_mask)
{

  assert(hard_mask==true);
  const int n = seqs.get_length();

  printf("Timing the mask sequences on CPU...\n");
  Timer total;
  total.start();

#if not defined(_OPENMP)
  Thread_pool threads;
  Atomic<size_t> next(0);
  for (size_t i = 0; i < config.threads_; ++i)
    threads.push_back(launch_thread(mask_worker, &next, &seqs, &masking, hard_mask));
  threads.join_all();

#else

#pragma omp parallel for num_threads(config.threads_)
  for (int i=0; i < n; i++){
    masking.call_opt(seqs.ptr(i), seqs.length(i));
  }

#endif

  message_stream << "Total time (maskSequences) on the CPU = " <<
    total.getElapsedTimeInMicroSec() / 1e6 << " s" << std::endl;

  // on the device
  unsigned char* seqs_device = masking.call_opt(seqs);

  printf("Verify the sequences...\n");
  unsigned char* p = seqs_device;
  int error = 0;
  for (int i = 0; i < n; i++) {
    if (0 != strncmp((const char*)p, seqs.ptr(i), seqs.length(i))) {
      printf("error at i=%d  length=%zu\n", i, seqs.length(i));
      printf("host=");
      char* s = seqs.ptr(i);
      for (int j = 0; j < seqs.length(i); j++) {
        printf("%02d", s[j]);
      }
      printf("\ndevice=");
      for (int j = 0; j < seqs.length(i); j++)
        printf("%02d", *(seqs_device+i*33+j));
      printf("\n");
      error++;
    }
    p += seqs.length(i);
  }
  if (error == 0) printf("Success\n");
}

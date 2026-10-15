/*
Copyright 1998–2018 Bernhard Esslinger and the CrypTool Team. Permission
is granted to copy, distribute and/or modify this document under the terms of the
GNU Free Documentation License, Version 1.3 or any later version published by the
Free Software Foundation (FSF). A copy of the license is included in the section
entitled "GNU Free Documentation License".
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include "iris_support.hpp"
#include <vector>

#include "config.hpp"
bool extractBigrams(float *scores, const char* filename) {
  FILE* bigramsFile = fopen(filename, "r");
  if (bigramsFile == NULL) {
    fprintf(stderr, "Failed to open file %s. Exit\n", filename);
    return true;
  }
  while(1){
    char tempBigram[3];
    float tempBigramScore = 0.0;
    if (fscanf(bigramsFile, "%2s %f", tempBigram, &tempBigramScore) < 2)
    { break; } 
    scores[(tempBigram[0]-'a')*ALPHABET + tempBigram[1]-'a'] = tempBigramScore; 
  }
  fclose(bigramsFile);
  return false;
}

bool verify(int* encrMap) {
  bool pass = true;
  const char *expect = DECRYPTED_T;
  for (int j=0; j<ENCRYPTEDLEN; ++j) {
    if (encrMap[j] + 'a' != expect[j]) {
       pass = false; break;
    }
  }
  return pass;
}

float candidateScore(int* decrMsg, float* scores) {
  float total = 0.0;
  for (int j=0; j<ENCRYPTEDLEN-1; ++j) 
    total += scores[ALPHABET*decrMsg[j] + decrMsg[j+1]];  
  return total;
}


int main(int argc, char* argv[]) {
  std::cout.setf(std::ios::unitbuf);
  if (argc != 2) {
    printf("Usage: %s <path to file>\n", argv[0]);
    return 1;
  }
  const char* filename = argv[1];

  int encryptedMap[ENCRYPTEDLEN];

  for (int j=0; j<ENCRYPTEDLEN; ++j)
    encryptedMap[j] = ENCRYPTED_T[j] - 'a';

  float scores[totalBigrams];  
  bool fail = extractBigrams(scores, filename);
  if (fail) return 1;

  iris::Platform platform;
  int device=init_iris(platform,argc,argv);
  std::vector<int> plaintext(ENCRYPTEDLEN*THREADS);
  int* decrypted=plaintext.data();
  std::vector<unsigned int> state(THREADS);
  iris::DMem d_scores(scores,sizeof(scores));
  iris::DMem d_encrypted(encryptedMap,sizeof(encryptedMap));
  iris::DMem d_decrypted(decrypted,sizeof(int)*ENCRYPTEDLEN*THREADS);
  iris::DMem devStates(state.data(),THREADS*sizeof(unsigned int));
  int climbings = getenv("COLUMNAR_VALIDATION_CLIMBINGS") ?
      atoi(getenv("COLUMNAR_VALIDATION_CLIMBINGS")) : CLIMBINGS;
  auto start = std::chrono::steady_clock::now();

  {
    void* p[]={ &devStates }; int info[]={ iris_w };
    launch_iris(device,"setupKernel",B*T,T,1,p,info);
  }
  {
    void* p[]={ &d_scores, &d_encrypted, &devStates, &d_decrypted, &climbings };
    int info[]={ iris_r,iris_r,iris_r,iris_w,sizeof(climbings) };
    launch_iris(device,"decode",B*T,T,5,p,info);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time %f (s)\n", time * 1e-9f);

  flush_iris(device,d_decrypted);

  int bestCandidate = 0;
  float bestScore = CAP;
  float* scoreHistory = new float[B*T];

  //  Calculating the best score ..
  for (int j=0; j<THREADS; ++j)  {
    float currentScore = candidateScore(&decrypted[ENCRYPTEDLEN*j], scores);
    scoreHistory[j] = currentScore;
    if (currentScore < bestScore) {
      bestScore = currentScore;
      bestCandidate = j;
    }    
  }  

  printf("Best candidate index: %d score: %.9g\n", bestCandidate, bestScore);
  bool pass = verify(&decrypted[ENCRYPTEDLEN*bestCandidate]);
  printf("%s\n", pass ? "PASS" : "FAIL");

  if(const char* path=getenv("COLUMNAR_VALIDATION_DUMP")) {
    FILE* fp=fopen(path,"wb");
    if(!fp) return 1;
    size_t count=fwrite(decrypted,sizeof(int),ENCRYPTEDLEN*THREADS,fp);
    fclose(fp);
    if(count!=ENCRYPTEDLEN*THREADS) return 1;
  }

  delete[] scoreHistory;
  int errors=platform.error_count();
  check_iris(platform.finalize());
  return (errors || !pass) ? 1 : 0;
}

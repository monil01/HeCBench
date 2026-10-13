#include <hip/hip_runtime.h>
#include <chrono>
#include <cstring>
#include "seed.cuh"

int
__device__ getAlignScore(SeedL const &myseed){
  return myseed.score;
}

int
__device__ __host__ getBeginPositionH(SeedL const &myseed){
  return myseed.beginPositionH;
}

int
__device__ __host__ getBeginPositionV(SeedL const &myseed){
  return myseed.beginPositionV;
}

int
__device__ __host__ getEndPositionH(SeedL const &myseed){
  return myseed.endPositionH;
}

int
__device__ __host__ getEndPositionV(SeedL const &myseed){
  return myseed.endPositionV;
}

int
__device__ getSeedLLength(SeedL const &myseed){
  return myseed.seedLength;
}

int
__device__ getLowerDiagonal(SeedL const &myseed){
  return myseed.lowerDiagonal;
}

int
__device__ getUpperDiagonal(SeedL const &myseed){
  return myseed.upperDiagonal;
}

int
__device__ getBeginDiagonal(SeedL const &myseed){
  return myseed.beginDiagonal;
}

int
__device__ getEndDiagonal(SeedL const &myseed){
  return myseed.endDiagonal;
}

void
__device__ setAlignScore(SeedL &myseed,int const value){
  myseed.score = value;
}

void
__device__ __host__ setBeginPositionH(SeedL &myseed,int const value){
  myseed.beginPositionH = value;
}

void
__device__ __host__ setBeginPositionV(SeedL &myseed,int const value){
  myseed.beginPositionV = value;
}

void
__device__ __host__ setEndPositionH(SeedL &myseed,int const value){
  myseed.endPositionH = value;
}

void
__device__ __host__ setEndPositionV(SeedL &myseed,int const value){
  myseed.endPositionV = value;
}

void
__device__ __host__ setSeedLLength(SeedL &myseed,int const value){
  myseed.seedLength = value;
}

void
__device__ __host__ setLowerDiagonal(SeedL &myseed,int const value){
  myseed.lowerDiagonal = value;
}

void
__device__ __host__ setUpperDiagonal(SeedL &myseed,int const value){
  myseed.upperDiagonal = value;
}

void
__device__ __host__ setBeginDiagonal(SeedL &myseed,int const value){
  myseed.beginDiagonal = value;
}

void
__device__ __host__ setEndDiagonal(SeedL &myseed,int const value){
  myseed.endDiagonal = value;
}
#include "score.cuh"

// return match score
int
__device__ __host__  scoreMatch(ScoringSchemeL const& me) {
  return me.match_score;
}

// individually set match score
void
setScoreMatch(ScoringSchemeL & me, int const& value) {
  me.match_score = value;
}

// return mismatch score
int
__device__ __host__ scoreMismatch(ScoringSchemeL const& me) {
  return me.mismatch_score;
}

// individually set mismatch score
void
setScoreMismatch(ScoringSchemeL & me, int const& value) {
  me.mismatch_score = value;
}

// return gap extension score
int
scoreGapExtend(ScoringSchemeL const& me) {
  return me.gap_extend_score;
}

// individually set gap extension score
void
setScoreGapExtend(ScoringSchemeL & me, int const& value) {
  me.gap_extend_score = value;
}

// return gap opening score
int
scoreGapOpen(ScoringSchemeL const& me) {
  return me.gap_open_score;
}

//returns the gap_open_score NB: valid only for linear gap
int 
__device__ __host__ scoreGap(ScoringSchemeL const & me){
  return me.gap_extend_score;
}

// individually set gap opening score
void
setScoreGapOpen(ScoringSchemeL & me, int const& value) {
  me.gap_open_score = value;
}

// set gap opening and gap extend scores
void
setScoreGap(ScoringSchemeL & me, int const& value) {
  me.gap_extend_score = value;
  me.gap_open_score = value;
}

int
__device__ __host__ score(ScoringSchemeL const & me, char valH, char valV) {
  if (valH == valV)
    return scoreMatch(me);
  else
    return scoreMismatch(me);
}
//==================================================================
// Title:  x-drop seed-and-extend alignment algorithm
// Author: A. Zeni, G. Guidi
//==================================================================
#include "logan_functions.cuh"
#include "seed.cuh"

#include <chrono>

using namespace std;
using namespace chrono;

__inline__ __device__ void warpReduce(volatile short *input, int myTId)
{
  input[myTId] = (input[myTId] > input[myTId + 32]) ? input[myTId] : input[myTId + 32]; 
  input[myTId] = (input[myTId] > input[myTId + 16]) ? input[myTId] : input[myTId + 16];
  input[myTId] = (input[myTId] > input[myTId + 8]) ? input[myTId] : input[myTId + 8]; 
  input[myTId] = (input[myTId] > input[myTId + 4]) ? input[myTId] : input[myTId + 4];
  input[myTId] = (input[myTId] > input[myTId + 2]) ? input[myTId] : input[myTId + 2];
  input[myTId] = (input[myTId] > input[myTId + 1]) ? input[myTId] : input[myTId + 1];
}

__inline__ __device__ short reduce_max(short *input, int dim, int n_threads)
{
  unsigned int myTId = threadIdx.x;   
  if(dim>32){
    for(int i = n_threads/2; i >32; i>>=1){
      if(myTId < i){
        input[myTId] = (input[myTId] > input[myTId + i]) ? input[myTId] : input[myTId + i];
      }
      __syncthreads();
    }
  }
  if(myTId<32)
    warpReduce(input, myTId);
  __syncthreads();
  return input[0];
}

__inline__ __device__ void updateExtendedSeedL(
    SeedL &seed,
    ExtensionDirectionL direction, //as there are only 4 directions we may consider even smaller data types
    int cols,
    int rows,
    int lowerDiag,
    int upperDiag)
{
  if (direction == EXTEND_LEFTL)
  {
    int beginDiag = seed.beginDiagonal;
    // Set lower and upper diagonals.

    if (getLowerDiagonal(seed) > beginDiag + lowerDiag)
      setLowerDiagonal(seed, beginDiag + lowerDiag);
    if (getUpperDiagonal(seed) < beginDiag + upperDiag)
      setUpperDiagonal(seed, beginDiag + upperDiag);

    // Set new start position of seed.
    seed.beginPositionH -= rows;
    seed.beginPositionV -= cols;
  } else {  // direction == EXTEND_RIGHTL
    // Set new lower and upper diagonals.
    int endDiag = seed.endDiagonal;
    if (getUpperDiagonal(seed) < endDiag - lowerDiag)
      setUpperDiagonal(seed, (endDiag - lowerDiag));
    if (getLowerDiagonal(seed) > (endDiag - upperDiag))
      setLowerDiagonal(seed, endDiag - upperDiag);

    // Set new end position of seed.
    seed.endPositionH += rows;
    seed.endPositionV += cols;

  }
}

__inline__ __device__ void computeAntidiag(
    const short *antiDiag1,
    const short *antiDiag2,
          short *antiDiag3,
    const char* querySeg,
    const char* databaseSeg,
    const int best,
    const int scoreDropOff,
    const int cols,
    const int rows,
    const int minCol,
    const int maxCol,
    const int antiDiagNo,
    const int offset1,
    const int offset2,
    const ExtensionDirectionL direction,
    int n_threads)
{
  int tid = threadIdx.x;

  for(int i = 0; i < maxCol; i+=n_threads){

    int col = tid + minCol + i;
    int queryPos, dbPos;

    queryPos = col - 1;
    dbPos = col + rows - antiDiagNo - 1;

    if(col < maxCol){

      int tmp = max_logan(antiDiag2[col-offset2],antiDiag2[col-offset2-1]) + GAP_EXT;

      int score = (querySeg[queryPos] == databaseSeg[dbPos]) ? MATCH : MISMATCH;

      tmp = max_logan(antiDiag1[col-offset1-1]+score,tmp);

      antiDiag3[tid+1+i] = (tmp < best - scoreDropOff) ? UNDEF : tmp;

    }
  }
}

__inline__ __device__ void calcExtendedLowerDiag(int &lowerDiag,
    int const &minCol,
    int const &antiDiagNo)
{
  int minRow = antiDiagNo - minCol;
  if (minCol - minRow < lowerDiag)
    lowerDiag = minCol - minRow;
}

__inline__ __device__ void calcExtendedUpperDiag(
    int &upperDiag,
    int const &maxCol,
    int const &antiDiagNo)
{
  int maxRow = antiDiagNo + 1 - maxCol;
  if (maxCol - 1 - maxRow > upperDiag)
    upperDiag = maxCol - 1 - maxRow;
}

__inline__ __device__ void initAntiDiag3(
    short *antiDiag3,
    int &a3size,
    int const &offset,
    int const &maxCol,
    int const &antiDiagNo,
    int const &minScore,
    int const &gapCost,
    int const &undefined)
{
  a3size = maxCol + 1 - offset;

  antiDiag3[0] = undefined;
  antiDiag3[maxCol - offset] = undefined;

  if (antiDiagNo * gapCost > minScore)
  {
    if (offset == 0) // init first column
      antiDiag3[0] = antiDiagNo * gapCost;
    if (antiDiagNo - maxCol == 0) // init first row
      antiDiag3[maxCol - offset] = antiDiagNo * gapCost;
  }
}

__inline__ __device__ void initAntiDiags(
    short *antiDiag1,
    short *antiDiag2,
    short *antiDiag3,
    int &a2size,
    int &a3size,
    int const &dropOff,
    int const &gapCost,
    int const &undefined)
{
  a2size = 1;

  antiDiag2[0] = 0;

  a3size = 2;

  antiDiag3[0] = gapCost;
  antiDiag3[1] = gapCost;
}

extern "C" __global__ void extendSeedLGappedXDropOneDirectionGlobal(
    SeedL *__restrict__ seed,
    const char *__restrict__ querySegArray,
    const char *__restrict__ databaseSegArray,
    const ExtensionDirectionL direction,
    const int scoreDropOff,
    int *__restrict__ res,
    const int *__restrict__ offsetQuery,
    const int *__restrict__ offsetTarget,
    const int offAntidiag,
    short *__restrict__ antidiag,
    const int n_threads)
{
  __shared__ short temp_alloc[1024];
  short *temp= &temp_alloc[0];

  int myId = blockIdx.x;
  int myTId = threadIdx.x;
  const char *querySeg;
  const char *databaseSeg;

  if(myId==0){
    querySeg = querySegArray;
    databaseSeg = databaseSegArray;
  }
  else{
    querySeg = querySegArray + offsetQuery[myId-1];
    databaseSeg = databaseSegArray + offsetTarget[myId-1];
  }

  short *antiDiag1 = &antidiag[myId*offAntidiag*3]; 
  short* antiDiag2 = &antiDiag1[offAntidiag];
  short* antiDiag3 = &antiDiag2[offAntidiag];

  SeedL mySeed(seed[myId]);  
  //dimension of the antidiagonals
  int a1size = 0, a2size = 0, a3size = 0;
  int cols, rows;

  if(myId == 0){
    cols = offsetQuery[myId]+1;
    rows = offsetTarget[myId]+1;
  }
  else{
    cols = offsetQuery[myId]-offsetQuery[myId-1]+1;
    rows = offsetTarget[myId]-offsetTarget[myId-1]+1;
  }

  if (rows == 1 || cols == 1) return;

  int minCol = 1;
  int maxCol = 2;

  int offset1 = 0; // number of leading columns that need not be calculated in antiDiag1
  int offset2 = 0; //                                                       in antiDiag2
  int offset3 = 0; //                                                       in antiDiag3

  initAntiDiags(antiDiag1,antiDiag2, antiDiag3, a2size, a3size, scoreDropOff, GAP_EXT, UNDEF);
  int antiDiagNo = 1; // the currently calculated anti-diagonal

  int best = 0; // maximal score value in the DP matrix (for drop-off calculation)

  int lowerDiag = 0;
  int upperDiag = 0;

  while (minCol < maxCol)
  {  
    ++antiDiagNo;

    //antidiagswap
    //antiDiag2 -> antiDiag1
    //antiDiag3 -> antiDiag2
    //antiDiag1 -> antiDiag3
    short *t = antiDiag1;
    antiDiag1 = antiDiag2;
    antiDiag2 = antiDiag3;
    antiDiag3 = t;
    int t_l = a1size;
    a1size = a2size;
    a2size = a3size;
    a3size = t_l;
    offset1 = offset2;
    offset2 = offset3;
    offset3 = minCol-1;

    initAntiDiag3(antiDiag3, a3size, offset3, maxCol, antiDiagNo, best - scoreDropOff, GAP_EXT, UNDEF);

    computeAntidiag(antiDiag1, antiDiag2, antiDiag3, querySeg, databaseSeg,
                    best, scoreDropOff, cols, rows, minCol, maxCol, antiDiagNo,
                    offset1, offset2, direction, n_threads);     
    //roofline analysis
    __syncthreads();  

    int tmp, antiDiagBest = UNDEF;  
    for(int i=0; i<a3size; i+=n_threads){
      int size = a3size-i;

      if(myTId<n_threads){
        temp[myTId] = (myTId<size) ? antiDiag3[myTId+i]:UNDEF;        
      }
      __syncthreads();

      tmp = reduce_max(temp,size, n_threads);
      antiDiagBest = (tmp>antiDiagBest) ? tmp:antiDiagBest;

    }
    best = (best > antiDiagBest) ? best : antiDiagBest;

    while (minCol - offset3 < a3size && antiDiag3[minCol - offset3] == UNDEF &&
        minCol - offset2 - 1 < a2size && antiDiag2[minCol - offset2 - 1] == UNDEF)
    {
      ++minCol;
    }

    // Calculate new maxCol
    while (maxCol - offset3 > 0 && (antiDiag3[maxCol - offset3 - 1] == UNDEF) &&
        (antiDiag2[maxCol - offset2 - 1] == UNDEF))
    {
      --maxCol;
    }
    ++maxCol;

    // Calculate new lowerDiag and upperDiag of extended seed
    calcExtendedLowerDiag(lowerDiag, minCol, antiDiagNo);
    calcExtendedUpperDiag(upperDiag, maxCol - 1, antiDiagNo);

    // end of databaseSeg reached?
    minCol = (minCol > (antiDiagNo + 2 - rows)) ? minCol : (antiDiagNo + 2 - rows);
    // end of querySeg reached?
    maxCol = (maxCol < cols) ? maxCol : cols;
  }

  int longestExtensionCol = a3size + offset3 - 2;
  int longestExtensionRow = antiDiagNo - longestExtensionCol;
  int longestExtensionScore = antiDiag3[longestExtensionCol - offset3];
  
  if (longestExtensionScore == UNDEF)
  {
    if (antiDiag2[a2size -2] != UNDEF)
    {
      // reached end of query segment
      longestExtensionCol = a2size + offset2 - 2;
      longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
      longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
    }
    else if (a2size > 2 && antiDiag2[a2size-3] != UNDEF)
    {
      // reached end of database segment
      longestExtensionCol = a2size + offset2 - 3;
      longestExtensionRow = antiDiagNo - 1 - longestExtensionCol;
      longestExtensionScore = antiDiag2[longestExtensionCol - offset2];
    }
  }
  
  if (longestExtensionScore == UNDEF){
  
    // general case
    for (int i = 0; i < a1size; ++i){
  
      if (antiDiag1[i] > longestExtensionScore){
  
        longestExtensionScore = antiDiag1[i];
        longestExtensionCol = i + offset1;
        longestExtensionRow = antiDiagNo - 2 - longestExtensionCol;
  
      }
    }
  }
  
  if (longestExtensionScore != UNDEF)
    updateExtendedSeedL(mySeed, direction, longestExtensionCol, longestExtensionRow, lowerDiag, upperDiag);

  seed[myId] = mySeed;
  res[myId] = longestExtensionScore;
}


#include <bits/stdc++.h>
using std::min;using std::max;
#define __global__
struct D{int x=0;};D blockIdx,threadIdx,blockDim{128};int observedScore=-1;
__global__ void kernel_createIndex4(
    const char *reads, 
    const int *lengths, 
    const long *offsets,
    unsigned short *indexs, 
    unsigned short *orders, 
    long *words, 
    int *magicBase,
    const int readsCount)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= readsCount) return;  // out of range
  int start = offsets[index];
  int end = start + lengths[index];
  int magic0=0, magic1=0, magic2=0, magic3=0;  // magic base
  char bases[4];
  for(int i=0; i<4; i++) {  // default is gap
    bases[i] = 4;
  }
  int wordCount = 0;
  for (int i=start; i<end; i++) {
    for(int j=0; j<3; j++) {  // copy base to array
      bases[j] = bases[j+1];
    }
    bases[3] = reads[i];
    switch (bases[3]) {  // update magic
      case 0:
        magic0++;
        break;
      case 1:
        magic1++;
        break;
      case 2:
        magic2++;
        break;
      case 3:
        magic3++;
        break;
    }
    unsigned short indexValue = 0;
    int flag = 0;  // if has gap
    for (int j=0; j<4; j++) {
      indexValue += (bases[j]&3)<<(3-j)*2;
      flag += max((int)(bases[j] - 3), 0);
    }
    indexs[i] = flag?65535:indexValue;  // gap value is 65535
    wordCount += flag?0:1;
  }
  words[index] = wordCount;  // index length
  magicBase[index*4+0] = magic0;  // update magicBase
  magicBase[index*4+1] = magic1;
  magicBase[index*4+2] = magic2;
  magicBase[index*4+3] = magic3;
}

__global__ void kernel_createIndex5(
    const char *reads, 
    const int *lengths, 
    const long *offsets,
    unsigned short *indexs, 
    unsigned short *orders, 
    long *words, 
    int *magicBase,
    const int readsCount)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= readsCount) return;
  int start = offsets[index];
  int end = start + lengths[index];
  int magic0=0, magic1=0, magic2=0, magic3=0;
  char bases[5];
  for(int i=0; i<5; i++) {
    bases[i] = 4;
  }
  int wordCount = 0;
  for (int i=start; i<end; i++) {
    for(int j=0; j<4; j++) {
      bases[j] = bases[j+1];
    }
    bases[4] = reads[i];
    switch (bases[4]) {
      case 0:
        magic0++;
        break;
      case 1:
        magic1++;
        break;
      case 2:
        magic2++;
        break;
      case 3:
        magic3++;
        break;
    }
    unsigned short indexValue = 0;
    int flag = 0;
    for (int j=0; j<5; j++) {
      indexValue += (bases[j]&3)<<(4-j)*2;
      flag += max((int)(bases[j] - 3), 0);
    }
    indexs[i] = flag?65535:indexValue;
    wordCount += flag?0:1;
  }
  words[index] = wordCount;
  magicBase[index*4+0] = magic0;
  magicBase[index*4+1] = magic1;
  magicBase[index*4+2] = magic2;
  magicBase[index*4+3] = magic3;
}

__global__ void kernel_createIndex6(
    const char *reads, 
    const int *lengths, 
    const long *offsets,
    unsigned short *indexs, 
    unsigned short *orders, 
    long *words, 
    int *magicBase,
    const int readsCount)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= readsCount) return;
  int start = offsets[index];
  int end = start + lengths[index];
  int magic0=0, magic1=0, magic2=0, magic3=0;
  char bases[6];
  for(int i=0; i<6; i++) {
    bases[i] = 4;
  }
  int wordCount = 0;
  for (int i=start; i<end; i++) {
    for(int j=0; j<5; j++) {
      bases[j] = bases[j+1];
    }
    bases[5] = reads[i];
    switch (bases[5]) {
      case 0:
        magic0++;
        break;
      case 1:
        magic1++;
        break;
      case 2:
        magic2++;
        break;
      case 3:
        magic3++;
        break;
    }
    unsigned short indexValue = 0;
    int flag = 0;
    for (int j=0; j<6; j++) {
      indexValue += (bases[j]&3)<<(5-j)*2;
      flag += max((int)(bases[j] - 3), 0);
    }
    indexs[i] = flag?65535:indexValue;
    wordCount += flag?0:1;
  }
  words[index] = wordCount;
  magicBase[index*4+0] = magic0;
  magicBase[index*4+1] = magic1;
  magicBase[index*4+2] = magic2;
  magicBase[index*4+3] = magic3;
}

__global__ void kernel_createIndex7(
    const char *reads, 
    const int *lengths, 
    const long *offsets,
    unsigned short *indexs, 
    unsigned short *orders, 
    long *words, 
    int *magicBase,
    const int readsCount)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= readsCount) return;
  int start = offsets[index];
  int end = start + lengths[index];
  int magic0=0, magic1=0, magic2=0, magic3=0;
  char bases[7];
  for(int i=0; i<7; i++) {
    bases[i] = 4;
  }
  int wordCount = 0;
  for (int i=start; i<end; i++) {
    for(int j=0; j<6; j++) {
      bases[j] = bases[j+1];
    }
    bases[6] = reads[i];
    switch (bases[6]) {
      case 0:
        magic0++;
        break;
      case 1:
        magic1++;
        break;
      case 2:
        magic2++;
        break;
      case 3:
        magic3++;
        break;
    }
    unsigned short indexValue = 0;
    int flag = 0;
    for (int j=0; j<7; j++) {
      indexValue += (bases[j]&3)<<(6-j)*2;
      flag += max((int)(bases[j] - 3), 0);
    }
    indexs[i] = flag?65535:indexValue;
    wordCount += flag?0:1;
  }
  words[index] = wordCount;
  magicBase[index*4+0] = magic0;
  magicBase[index*4+1] = magic1;
  magicBase[index*4+2] = magic2;
  magicBase[index*4+3] = magic3;
}

__global__ void kernel_align(
    const float threshold,
    const int *lengths,
    const long *offsets, 
    const unsigned int *compressed,
    const int *gaps,
    const int representative,
    int *cluster,
    const int readsCount)
{
  // variables
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= readsCount) return;  // out of range
  if (cluster[index] != -3) return;  // not pass filter
  int target = representative;  // representative read
  int query = index;  // query read
  int minLength = ceil((float)lengths[index] * threshold);
  int targetLength = lengths[target] - gaps[target];  // representative base count
  int queryLength = lengths[query] - gaps[query];  // query base count
  int target32Length = targetLength/16+1;  // compressed target length
  int query32Length  = queryLength/16+1;  // compressed query length
  int targetOffset = offsets[target]/16;  // representative offset
  int queryOffset = offsets[query]/16;  // query offset
  short rowNow[3000] = {0};  // dynamic matrix row
  short rowPrevious[3000] = {0};  // dynamic matrix row
  int columnPrevious[17] = {0};  // dynamic matrix column
  int columnNow[17] = {0};  // dynamic matrix column
  int shift = ceil((float)targetLength - (float)queryLength*threshold);
  shift = ceil((float)shift / 16.f); // shift form diagonal
  // compute
  for (int i = 0; i < query32Length; i++) {  // query is column
    // first big loop
    for (int j=0; j<17; j++) {
      columnPrevious[j] = 0;
      columnNow[j] = 0;
    }
    int targetIndex = 0;  // target position
    unsigned int queryPack = compressed[queryOffset+i];  // get 16 query bases
    int jstart = i-shift;
    jstart = max(jstart, 0);
    int jend = i+shift;
    jend = min(jend, target32Length);
    for (int j=0; j<target32Length; j++) {  // target is row
      columnPrevious[0] = rowPrevious[targetIndex];
      unsigned int targetPack = compressed[targetOffset+j];  // get 16 target bases
      //---16*16core----//
      for (int k=30; k>=0; k-=2) {  // 16 loops get target bases
        // first small loop
        int targetBase = (targetPack>>k)&3;  // get base from target
        int m=0;
        columnNow[m] = rowPrevious[targetIndex+1];
        for (int l=30; l>=0; l-=2) {  // 16 loops get query bases
          m++;
          int queryBase = (queryPack>>l)&3;  // get base from query
          int diffScore = queryBase == targetBase;
          columnNow[m] = columnPrevious[m-1] + diffScore;
          columnNow[m] = max(columnNow[m], columnNow[m - 1]);
          columnNow[m] = max(columnNow[m], columnPrevious[m]);
        }
        targetIndex++;
        rowNow[targetIndex] = columnNow[16];
        if (targetIndex == targetLength) {  // last column of dynamic matirx
          if(i == query32Length-1) {  // complete
            int score = columnNow[queryLength%16]; observedScore=score;
            if (score >= minLength) {
              cluster[index] = target;
            } else {
              cluster[index] = -1;
            }
            return;
          }
          break;
        }
        // secode small loop exchange columnPrevious and columnNow
        k-=2;
        targetBase = (targetPack>>k)&3;
        m=0;
        columnPrevious[m] = rowPrevious[targetIndex+1];
        for (int l=30; l>=0; l-=2) {
          m++;
          int queryBase = (queryPack>>l)&3;
          int diffScore = queryBase == targetBase;
          columnPrevious[m] = columnNow[m-1] + diffScore;
          columnPrevious[m] =
            max(columnPrevious[m], columnPrevious[m - 1]);
          columnPrevious[m] =
            max(columnPrevious[m], columnNow[m]);
        }
        targetIndex++;
        rowNow[targetIndex] = columnPrevious[16];
        if (targetIndex == targetLength) {
          if(i == query32Length-1) {
            int score = columnPrevious[queryLength%16]; observedScore=score;
            if (score >= minLength) {
              cluster[index] = target;
            } else {
              cluster[index] = -1;
            }
            return;
          }
          break;
        }
      }
    }
    // secode big loop exchage rowPrevious and rowNow
    i++;
    for (int j=0; j<17; j++) {
      columnPrevious[j] = 0;
      columnNow[j] = 0;
    }
    targetIndex = 0;
    queryPack = compressed[queryOffset+i];
    jstart = i-shift;
    jstart = max(jstart, 0);
    jend = i+shift;
    jend = min(jend, target32Length);
    for (int j=0; j<target32Length; j++) {
      unsigned int targetPack = compressed[targetOffset+j];
      //---16*16 core----//
      for (int k=30; k>=0; k-=2) {
        // first small loop
        int targetBase = (targetPack>>k)&3;
        int m=0;
        columnNow[m] = rowNow[targetIndex+1];
        for (int l=30; l>=0; l-=2) {
          m++;
          int queryBase = (queryPack>>l)&3;
          int diffScore = queryBase == targetBase;
          columnNow[m] = columnPrevious[m-1] + diffScore;
          columnNow[m] = max(columnNow[m], columnNow[m - 1]);
          columnNow[m] = max(columnNow[m], columnPrevious[m]);
        }
        targetIndex++;
        rowPrevious[targetIndex] = columnNow[16];
        if (targetIndex == targetLength) {
          if(i == query32Length-1) {
            int score = columnNow[queryLength%16]; observedScore=score;
            if (score >= minLength) {
              cluster[index] = target;
            } else {
              cluster[index] = -1;
            }
            return;
          }
          break;
        }
        // second small loop
        k-=2;
        targetBase = (targetPack>>k)&3;
        m=0;
        columnPrevious[m] = rowNow[targetIndex+1];
        for (int l=30; l>=0; l-=2) {
          m++;
          int queryBase = (queryPack>>l)&3;
          int diffScore = queryBase == targetBase;
          columnPrevious[m] = columnNow[m-1] + diffScore;
          columnPrevious[m] =
            max(columnPrevious[m], columnPrevious[m - 1]);
          columnPrevious[m] =
            max(columnPrevious[m], columnNow[m]);
        }
        targetIndex++;
        rowPrevious[targetIndex] = columnPrevious[16];
        if (targetIndex == targetLength) {
          if(i == query32Length-1) {
            int score = columnPrevious[queryLength%16]; observedScore=score;
            if (score >= minLength) {
              cluster[index] = target;
            } else {
              cluster[index] = -1;
            }
            return;
          }
          break;
        }
      }
    }
  }
}


int main(){std::string seq="ACGTACGTTGCAAGTCGATCGTAACGATTCGATGCA";std::vector<char>read(seq.size()+32,4);for(size_t i=0;i<seq.size();i++)read[i]=std::string("ACGT").find(seq[i]);int len=seq.size();long off[2]={0,long(read.size())};std::vector<unsigned short>indexs(read.size(),0),orders(read.size(),0);long words=0;int magic[4];
for(int k=4;k<=7;k++){if(k==4)kernel_createIndex4(read.data(),&len,off,indexs.data(),orders.data(),&words,magic,1);if(k==5)kernel_createIndex5(read.data(),&len,off,indexs.data(),orders.data(),&words,magic,1);if(k==6)kernel_createIndex6(read.data(),&len,off,indexs.data(),orders.data(),&words,magic,1);if(k==7)kernel_createIndex7(read.data(),&len,off,indexs.data(),orders.data(),&words,magic,1);std::cout<<"INDEX "<<k<<" "<<words;for(int i=0;i<len;i++)std::cout<<" "<<indexs[i];std::cout<<"\n";}
std::mt19937 gen(717);for(int trial=0;trial<32;trial++){int lengths[2]={32+(trial*7)%161,24+(trial*11)%129};std::vector<std::vector<int>>str(2);for(int q=0;q<2;q++){for(int i=0;i<lengths[q];i++)str[q].push_back(gen()%4);}if(trial%3==0){for(int j=0;j<std::min(lengths[0],lengths[1]);j++)str[1][j]=str[0][j];}int gaps[2]={0,0};if(trial%5==0&&str[1].size()>30){str[1].erase(str[1].begin()+5);gaps[1]=1;}
long offsets[2]={0,long((lengths[0]/16+1)*16)};std::vector<unsigned>packed((offsets[1]+(lengths[1]/16+1)*16)/16,0);for(int q=0;q<2;q++)for(size_t i=0;i<str[q].size();i++)packed[offsets[q]/16+i/16]|=unsigned(str[q][i])<<((15-i%16)*2);int cluster[2]={0,-3};threadIdx.x=1;observedScore=-1;float threshold=trial%2?.8f:.95f;kernel_align(threshold,lengths,offsets,packed.data(),gaps,0,cluster,2);std::vector<int>row(str[0].size()+1,0),next(row.size());for(int y:str[1]){next[0]=0;for(size_t j=1;j<row.size();j++)next[j]=std::max({row[j-1]+int(str[0][j-1]==y),next[j-1],row[j]});row.swap(next);}std::cout<<"LCS "<<trial<<" "<<lengths[0]<<" "<<lengths[1]<<" "<<gaps[1]<<" "<<observedScore<<" "<<row.back()<<" "<<cluster[1]<<" "<<(row.back()>=std::ceil(float(lengths[1])*threshold)?0:-1)<<"\n";} }

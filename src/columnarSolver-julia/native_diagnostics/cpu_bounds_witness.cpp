#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <omp.h>
#define __device__
#define __global__
#define __shared__
#define __restrict
#define __syncthreads() ((void)0)
struct dim {int x;}; thread_local dim blockIdx,blockDim,threadIdx;
thread_local int CURRENT_IDX=-1,CURRENT_CYCLE=-1;thread_local unsigned long long DRAWS=0;thread_local unsigned int LAST_SEED=0;
struct BoundFailure{int idx,cycle,left,right;unsigned long long draws;unsigned int state;};
#define B ((int)32)
#define T ((int)32)
#define THREADS ((int)B*T)
#define CLIMBINGS 150000
#define ALPHABET 26
#define totalBigrams ((int)ALPHABET*ALPHABET)
#define CAP ((float)999999.0)

#define ENCRYPTED_T "tteohtedanisroudesereguwocubsoitoabbofeiaiutsdheeisatsarsturesuaastniersrotnesctrctxdiwmhcusyenorndasmhaipnnptmaeecspegdeislwoheoiymreeotbsspiatoanihrelhwctftrhpuunhoianunreetrioettatlsnehtbaecpvgtltcirottonesnobeeeireaymrtohaawnwtesssvassirsrhabapnsynntitsittchitoosbtelmlaouitrehhwfeiaandeitciegfreoridhdcsheucrnoihdeoswobaceeaorgndlstigeearsotoetduedininttpedststntefoeaheoesuetvmmiorftuuhsurof"
#define ENCRYPTEDLEN ((int)sizeof(ENCRYPTED_T)-1)

#define DECRYPTED_T "thedistinctionbetweentherouteciphertranspositionandthesubstitutioncipherwherewholewordsaresubstitutedforlettersoftheoriginaltextmustbemadeonthebasisofthewordsactuallyuseditisbettertoconsidersuchamessageasaroutecipherwhenthewordsusedappeartohavesomeconsecutivemeaningbearingonthesituationathandasubstitutioncipherofthisvarietywouldonlybeusedfortransmissionofashortmessageofgreatimportanceandsecrecy"

#define KEY_LENGTH 30
#define SECTION_CONSTANT ENCRYPTEDLEN/KEY_LENGTH

#define HEUR_THRESHOLD_OP1 50
#define HEUR_THRESHOLD_OP2 70

#define OP1_HOP 4
#define OP2_HOP 2



__device__
float LCG_random_float(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m; DRAWS++;LAST_SEED=*seed;
  return (float) (*seed) / (float) m;
}

__device__
void LCG_random_init(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
}

__global__
void setupKernel(unsigned int* state) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  state[idx] = idx;
  for (int i = 0; i < idx; i++)
    LCG_random_init(&state[idx]);
}

__device__
void decrypt(const int* encrypted, const int* key, int* decrypted) {

  int columns[KEY_LENGTH][SECTION_CONSTANT+1];
  int offset = 0;
  int colLength[KEY_LENGTH];

  for (int j=0; j<KEY_LENGTH; ++j) {
    colLength[j] = ENCRYPTEDLEN / KEY_LENGTH;
    if (j < ENCRYPTEDLEN % KEY_LENGTH)
      colLength[j]++;
  }

  for (int keyPos=0; keyPos < KEY_LENGTH; ++keyPos) {
    offset = 0;
    for (int i=0; i<KEY_LENGTH; ++i)
      if (key[i] < key[keyPos])
        offset += colLength[i];

    for (int j=0; j<colLength[keyPos]; ++j)   
      columns[key[keyPos]][j] = encrypted[offset+j];          
  } 

  for (int j=0; j<ENCRYPTEDLEN; ++j) 
    decrypted[j] = columns[key[j % KEY_LENGTH]][j / KEY_LENGTH];  
} 

__device__
void swapElements(int *key, int posLeft, int posRight) {
  if(posLeft<0||posLeft>=KEY_LENGTH||posRight<0||posRight>=KEY_LENGTH)throw BoundFailure{CURRENT_IDX,CURRENT_CYCLE,posLeft,posRight,DRAWS,LAST_SEED};
  if (posLeft != posRight)
  {
    key[posLeft] -= key[posRight];
    key[posRight] += key[posLeft];
    key[posLeft] = key[posRight] - key[posLeft];
  }
}

__device__ 
void swapBlock(int *key, int posLeft, int posRight, int length) {  
  for (int i=0; i<length; i++) 
    swapElements(key, (posLeft+i)%KEY_LENGTH, (posRight+i)%KEY_LENGTH);
}

__global__ 
void decode(const float *__restrict d_scores, 
            const int *__restrict d_encrypted,
            const unsigned int*__restrict  globalState, 
            int *__restrict d_decrypted) {

  __shared__ float shared_scores[ALPHABET*ALPHABET];

  int key[KEY_LENGTH];
  int localDecrypted[ENCRYPTEDLEN];  
  int bestLocalDecrypted[ENCRYPTEDLEN];  
  int leftLetter = 0;
  int rightLetter = 0;
  int backupKey[KEY_LENGTH];
  int shiftHelper[KEY_LENGTH];
  int blockStart, blockEnd;
  int l,f,t,t0,n,ff,tt;
  float tempScore = 0.f;
  float bestScore = CAP;
  int j = 0, jj = 0;

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int localState = globalState[idx];

  if (true) {
    for (j=0; j<ALPHABET;++j)
      for (jj=0; jj<ALPHABET; ++jj)
        shared_scores[j*ALPHABET + jj] = d_scores[j*ALPHABET + jj];
  }

  __syncthreads();

  for (j=0; j<KEY_LENGTH; ++j) 
    key[j]=j;

  for (j=0; j<KEY_LENGTH; ++j) {
    swapElements(key, j, LCG_random_float(&localState)*KEY_LENGTH);
  }

  for (int cycles=0; cycles<CLIMBINGS; ++cycles) { CURRENT_CYCLE=cycles;  

    for (j=0; j<KEY_LENGTH;j++)
      backupKey[j] = key[j];

    tempScore = 0.f;

    int branch = LCG_random_float(&localState)*100; 

    if (branch < HEUR_THRESHOLD_OP1)
    {
      for (j=0; j<1+LCG_random_float(&localState)*OP1_HOP; j++) 
      {
        leftLetter = LCG_random_float(&localState)*KEY_LENGTH;   
        rightLetter = LCG_random_float(&localState)*KEY_LENGTH; 
        swapElements(key, leftLetter, rightLetter);
      }            
    }

    else if (branch < HEUR_THRESHOLD_OP2)
    {
      for (j=0; j< 1+LCG_random_float(&localState)*OP2_HOP;j++)
      {
        blockStart = LCG_random_float(&localState)*KEY_LENGTH;
        blockEnd = LCG_random_float(&localState)*KEY_LENGTH;
        swapBlock(key, blockStart, blockEnd, 1+LCG_random_float(&localState)*(abs((blockStart-blockEnd))-1));
      }
    }

    else 
    {
      l = 1 + LCG_random_float(&localState)*(KEY_LENGTH-2);
      f = LCG_random_float(&localState)*(KEY_LENGTH-1);
      t = (f+1+(LCG_random_float(&localState)*(KEY_LENGTH-2)));
      t = t % KEY_LENGTH;

      for (j=0; j< KEY_LENGTH;j++)
        shiftHelper[j] = key[j];

      t0 = (t-f+KEY_LENGTH) % KEY_LENGTH;
      n = (t0+l) % KEY_LENGTH;

      for (j=0; j<n;j++) 
      {
        ff = (f+j) % KEY_LENGTH;
        tt = (((t0+j)%n)+f)%KEY_LENGTH;
        key[tt] = shiftHelper[ff];
      }        
    }      

    decrypt(d_encrypted, key, localDecrypted);    

    for (j=0; j<ENCRYPTEDLEN-1; ++j) {
      tempScore += shared_scores[ALPHABET*localDecrypted[j] + localDecrypted[j+1]];
    }

    if (tempScore < bestScore) {
      bestScore = tempScore;
      for (j=0; j<ENCRYPTEDLEN; ++j) {
        bestLocalDecrypted[j] = localDecrypted[j];
      }
    }    

    else 
    {
      for (j=0; j<KEY_LENGTH;j++)
        key[j] = backupKey[j];      
    }
  }

  for (j=0; j<ENCRYPTEDLEN; ++j)
    d_decrypted[idx*ENCRYPTEDLEN+j] = bestLocalDecrypted[j];
}



int main(){float scores[676]={};FILE*f=fopen(DATA_PATH,"r");char word[3];float value;int rows=0;while(fscanf(f,"%2s %f",word,&value)==2){scores[(word[0]-'a')*26+word[1]-'a']=value;rows++;}fclose(f);if(rows!=676)return 3;int encrypted[ENCRYPTEDLEN];for(int j=0;j<ENCRYPTEDLEN;j++)encrypted[j]=ENCRYPTED_T[j]-'a';unsigned int states[THREADS];int*out=new int[THREADS*ENCRYPTEDLEN];int fails=0,passes=0;
#pragma omp parallel for num_threads(16) schedule(dynamic,1)
for(int idx=0;idx<THREADS;idx++){blockIdx.x=idx/32;blockDim.x=32;threadIdx.x=idx%32;CURRENT_IDX=idx;CURRENT_CYCLE=-1;DRAWS=0;setupKernel(states);try{decode(scores,encrypted,states,out);__atomic_add_fetch(&passes,1,__ATOMIC_RELAXED);}catch(BoundFailure e){__atomic_add_fetch(&fails,1,__ATOMIC_RELAXED);
#pragma omp critical
{printf("BOUNDS_WITNESS {\"thread\":%d,\"cycle\":%d,\"left\":%d,\"right\":%d,\"draws\":%llu,\"state\":%u}\n",e.idx,e.cycle,e.left,e.right,e.draws,e.state);fflush(stdout);}}
}
printf("CPU_FIXED_FULL_WORKLOAD {\"threads\":1024,\"climbings\":150000,\"completed_safe\":%d,\"range_rejections\":%d}\n",passes,fails);delete[]out;return 0;}

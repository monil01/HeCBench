#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif

#define SCORE(i, j) input_itemsets_l[j + i * (BLOCK_SIZE+1)]
#define REF(i, j) reference_l[j + i * BLOCK_SIZE]
__device__ __host__
int maximum( int a, int b, int c){

  int k;
  if( a <= b )
    k = b;
  else
    k = a;
  if( k <=c )
    return(c);
  else
    return(k);
}

extern "C" __global__ void
kernel1 (int*__restrict__ d_input_itemsets,
         const int*__restrict__ d_reference,
         const int offset_r,
         const int offset_c,
         const int max_cols,
         const int blk,
         const int penalty)
{
  __shared__ int input_itemsets_l [(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)];
  __shared__ int reference_l [BLOCK_SIZE*BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  // Base elements
  int base = offset_r * max_cols + offset_c;

  int b_index_x = bx;
  int b_index_y = blk - 1 - bx;

  int index   =   base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( max_cols + 1 );
  int index_n   = base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  int index_w   = base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( max_cols );
  int index_nw =  base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

  if (tx == 0) SCORE(tx, 0) = d_input_itemsets[index_nw + tx];

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)  {
    REF(ty, tx) =  d_reference[index + max_cols * ty];
  }

  SCORE((tx + 1), 0) = d_input_itemsets[index_w + max_cols * tx];

  SCORE(0, (tx + 1)) = d_input_itemsets[index_n];

  __syncthreads();

  for( int m = 0 ; m < BLOCK_SIZE ; m++){
     if ( tx <= m ){
        int t_index_x =  tx + 1;
        int t_index_y =  m - tx + 1;

        SCORE(t_index_y, t_index_x) = maximum( SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
              SCORE((t_index_y),   (t_index_x-1)) - (penalty),
              SCORE((t_index_y-1), (t_index_x))   - (penalty));
     }
     __syncthreads();
  }

  for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){

     if ( tx <= m){
        int t_index_x =  tx + BLOCK_SIZE - m ;
        int t_index_y =  BLOCK_SIZE - tx;

        SCORE(t_index_y, t_index_x) = maximum(  SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
              SCORE((t_index_y),   (t_index_x-1)) - (penalty),
              SCORE((t_index_y-1), (t_index_x))   - (penalty));
     }
     __syncthreads();
  }

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++) {
     d_input_itemsets[index + max_cols * ty] = SCORE((ty+1), (tx+1));
  }

}

extern "C" __global__ void
kernel2 (int*__restrict__ d_input_itemsets,
         const int*__restrict__ d_reference,
         const int block_width,
         const int offset_r,
         const int offset_c,
         const int max_cols,
         const int blk,
         const int penalty)
{
   __shared__ int input_itemsets_l [(BLOCK_SIZE + 1) *(BLOCK_SIZE+1)];
   __shared__ int reference_l [BLOCK_SIZE*BLOCK_SIZE];
   int bx = blockIdx.x;
   int tx = threadIdx.x;

   // Base elements
   int base = offset_r * max_cols + offset_c;
   int b_index_x = bx + block_width - blk  ;
   int b_index_y = block_width - bx -1;

   int index   =   base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( max_cols + 1 );
   int index_n   = base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
   int index_w   = base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( max_cols );
   int index_nw =  base + max_cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

   if (tx == 0)
      SCORE(tx, 0) = d_input_itemsets[index_nw];

   for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
      REF(ty, tx) =  d_reference[index + max_cols * ty];

   SCORE((tx + 1), 0) = d_input_itemsets[index_w + max_cols * tx];

   SCORE(0, (tx + 1)) = d_input_itemsets[index_n];

   __syncthreads();

   for( int m = 0 ; m < BLOCK_SIZE ; m++){
      if ( tx <= m ){

         int t_index_x =  tx + 1;
         int t_index_y =  m - tx + 1;

         SCORE(t_index_y, t_index_x) = maximum(  SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
               SCORE((t_index_y),   (t_index_x-1)) - (penalty),
               SCORE((t_index_y-1), (t_index_x))   - (penalty));
      }
      __syncthreads();
   }

   for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){

      if ( tx <= m){

         int t_index_x =  tx + BLOCK_SIZE - m ;
         int t_index_y =  BLOCK_SIZE - tx;

         SCORE(t_index_y, t_index_x) = maximum( SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
               SCORE((t_index_y),   (t_index_x-1)) - (penalty),
               SCORE((t_index_y-1), (t_index_x))   - (penalty));

      }
      __syncthreads();
   }

   for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
      d_input_itemsets[index + ty * max_cols] = SCORE((ty+1), (tx+1));
}

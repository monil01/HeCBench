#include <cstdint>
#include <cstdio>
int main(){uint8_t mac[16];for(int i=0;i<16;i++)mac[i]=i+1;uint32_t schedule12=0,schedule13=0,schedule14=0,schedule15=0,schedule16=0,schedule17=0,schedule18=0,schedule19=0;
  auto condition = (
      schedule12 == ( (uint32_t)
        (((uint32_t)(mac[3] ^ ((uint8_t) (schedule16 >> 24) ))) << 24) | 
        (((uint32_t)(mac[2] ^ ((uint8_t) (schedule16 >> 16) ))) << 16) | 
        (((uint32_t)(mac[1] ^ ((uint8_t) (schedule16 >>  8) ))) <<  8) | 
        (((uint32_t)(mac[0] ^ ((uint8_t) (schedule16)))) << 0) )
      )
    &&
    (
     schedule13 == ( (uint32_t)
       (((uint32_t)(mac[7] ^ ((uint8_t) (schedule17 >> 24) ))) << 24) | 
       (((uint32_t)(mac[6] ^ ((uint8_t) (schedule17 >> 16) ))) << 16) | 
       (((uint32_t)(mac[5] ^ ((uint8_t) (schedule17 >>  8) ))) <<  8) | 
       (((uint32_t)(mac[4] ^ ((uint8_t) (schedule17)))) << 0) )
    )
    &&
    (
     schedule14 == ( (uint32_t)
       (((uint32_t)(mac[11] ^ ((uint8_t) (schedule18 >> 24) ))) << 24) | 
       (((uint32_t)(mac[10] ^ ((uint8_t) (schedule18 >> 16) ))) << 16) | 
       (((uint32_t)(mac[9]  ^ ((uint8_t) (schedule18 >>  8) ))) <<  8) | 
       (((uint32_t)(mac[8]  ^ ((uint8_t) (schedule18)))) << 0) )
    )
    &&
    (
     schedule15 == ( (uint32_t)
       (((uint32_t)(mac[15] ^ ((uint8_t) (schedule19 >> 24) ))) << 24) | 
       (((uint32_t)(mac[14] ^ ((uint8_t) (schedule19 >> 16) ))) << 16) | 
       (((uint32_t)(mac[13] ^ ((uint8_t) (schedule19 >>  8) ))) <<  8) | 
       (((uint32_t)(mac[12] ^ ((uint8_t) (schedule19)))) << 0) )
    );

printf("{\"source_condition\":%s,\"intended_full_MAC_equality\":false,\"computed_MAC_words\":[0,0,0,0],\"encrypted_MAC_bytes\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],\"counter_AES_words\":[0,0,0,0]}\n", condition?"true":"false");}

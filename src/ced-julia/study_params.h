#pragma once
struct CaseSpec{int id,caseid,frame,rows,cols,n;const char* input;};
constexpr int CASES=12,PREFIX=401922,COUNT=803730,TRACE_COUNT=669680,TILE=16,SHMEM=18*18*4;
const CaseSpec SPECS[]={{0,0,0,34,50,1700,"case00_input_u8.bin"},{1,0,1,34,50,1700,"case01_input_u8.bin"},{2,0,2,34,50,1700,"case02_input_u8.bin"},{3,0,3,34,50,1700,"case03_input_u8.bin"},{4,1,0,66,98,6468,"case04_input_u8.bin"},{5,1,1,66,98,6468,"case05_input_u8.bin"},{6,1,2,66,98,6468,"case06_input_u8.bin"},{7,1,3,66,98,6468,"case07_input_u8.bin"},{8,2,0,130,194,25220,"case08_input_u8.bin"},{9,2,1,130,194,25220,"case09_input_u8.bin"},{10,2,2,130,194,25220,"case10_input_u8.bin"},{11,2,3,130,194,25220,"case11_input_u8.bin"}};

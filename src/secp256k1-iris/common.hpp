#pragma once
using uint=unsigned int;
using ulong=unsigned long;
static_assert(sizeof(uint)==4 && sizeof(ulong)==8);
typedef struct {
  uint n[10];
} secp256k1_fe;

typedef struct {
  uint n[8];
} secp256k1_fe_storage;

typedef struct {
  secp256k1_fe x;
  secp256k1_fe y;
} secp256k1_ge;

typedef struct {
  secp256k1_fe x;
  secp256k1_fe y;
  secp256k1_fe z;
} secp256k1_gej;

typedef struct {
  secp256k1_fe_storage x;
  secp256k1_fe_storage y;
} secp256k1_ge_storage;

#define SECP256K1_FE_STORAGE_CONST(d7, d6, d5, d4, d3, d2, d1, d0) {{ (d0), (d1), (d2), (d3), (d4), (d5), (d6), (d7) }}
#define SECP256K1_GE_STORAGE_CONST(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) {SECP256K1_FE_STORAGE_CONST((a),(b),(c),(d),(e),(f),(g),(h)), SECP256K1_FE_STORAGE_CONST((i),(j),(k),(l),(m),(n),(o),(p))}
#define SC SECP256K1_GE_STORAGE_CONST



static_assert(sizeof(secp256k1_ge_storage)==64);

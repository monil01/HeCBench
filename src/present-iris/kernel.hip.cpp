/*
 * Copyright (c) 2009, Shanghai Jiao Tong University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in
 *   the documentation and/or other materials provided with the
 *   distribution.
 * - Neither the name of the Shanghai Jiao Tong University nor the
 *   names of its contributors may be used to endorse or promote
 *   products derived from this software without specific prior
 *   written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <hip/hip_runtime.h>
extern "C" __global__ void present(
    const int num,
    const int rounds,
    const uint8_t *__restrict__ plains,
    const uint8_t *__restrict__ keys,
          uint8_t *__restrict__ ciphers,
    const uint8_t *__restrict__ sbox,
    const uint8_t *__restrict__ sbox_pmt_0,
    const uint8_t *__restrict__ sbox_pmt_1,
    const uint8_t *__restrict__ sbox_pmt_2,
    const uint8_t *__restrict__ sbox_pmt_3)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= num) return;
  const uint8_t *plain = plains + gid * 8;
  const uint8_t *key = keys + gid * 10;
  uint8_t *cipher = ciphers + gid * 8;
  uint8_t rounh_counter = 1;

  uint8_t state[8];
  uint8_t rounh_key[10];

  // add key
  state[0] = plain[0] ^ key[0];
  state[1] = plain[1] ^ key[1];
  state[2] = plain[2] ^ key[2];
  state[3] = plain[3] ^ key[3];
  state[4] = plain[4] ^ key[4];
  state[5] = plain[5] ^ key[5];
  state[6] = plain[6] ^ key[6];
  state[7] = plain[7] ^ key[7];

  // update key
  rounh_key[9] = key[6] << 5 | key[7] >> 3;
  rounh_key[8] = key[5] << 5 | key[6] >> 3;
  rounh_key[7] = key[4] << 5 | key[5] >> 3;
  rounh_key[6] = key[3] << 5 | key[4] >> 3;
  rounh_key[5] = key[2] << 5 | key[3] >> 3;
  rounh_key[4] = key[1] << 5 | key[2] >> 3;
  rounh_key[3] = key[0] << 5 | key[1] >> 3;
  rounh_key[2] = key[9] << 5 | key[0] >> 3;
  rounh_key[1] = key[8] << 5 | key[9] >> 3;
  rounh_key[0] = key[7] << 5 | key[8] >> 3;

  rounh_key[0] = (rounh_key[0] & 0x0F) | sbox[rounh_key[0] >> 4];

  rounh_key[7] ^= rounh_counter >> 1;
  rounh_key[8] ^= rounh_counter << 7;

  // substitution and permutation
  cipher[0] =
    (sbox_pmt_3[state[0]] & 0xC0) |
    (sbox_pmt_2[state[1]] & 0x30) |
    (sbox_pmt_1[state[2]] & 0x0C) |
    (sbox_pmt_0[state[3]] & 0x03);
  cipher[1] =
    (sbox_pmt_3[state[4]] & 0xC0) |
    (sbox_pmt_2[state[5]] & 0x30) |
    (sbox_pmt_1[state[6]] & 0x0C) |
    (sbox_pmt_0[state[7]] & 0x03);

  cipher[2] =
    (sbox_pmt_0[state[0]] & 0xC0) |
    (sbox_pmt_3[state[1]] & 0x30) |
    (sbox_pmt_2[state[2]] & 0x0C) |
    (sbox_pmt_1[state[3]] & 0x03);
  cipher[3] =
    (sbox_pmt_0[state[4]] & 0xC0) |
    (sbox_pmt_3[state[5]] & 0x30) |
    (sbox_pmt_2[state[6]] & 0x0C) |
    (sbox_pmt_1[state[7]] & 0x03);

  cipher[4] =
    (sbox_pmt_1[state[0]] & 0xC0) |
    (sbox_pmt_0[state[1]] & 0x30) |
    (sbox_pmt_3[state[2]] & 0x0C) |
    (sbox_pmt_2[state[3]] & 0x03);
  cipher[5] =
    (sbox_pmt_1[state[4]] & 0xC0) |
    (sbox_pmt_0[state[5]] & 0x30) |
    (sbox_pmt_3[state[6]] & 0x0C) |
    (sbox_pmt_2[state[7]] & 0x03);

  cipher[6] =
    (sbox_pmt_2[state[0]] & 0xC0) |
    (sbox_pmt_1[state[1]] & 0x30) |
    (sbox_pmt_0[state[2]] & 0x0C) |
    (sbox_pmt_3[state[3]] & 0x03);
  cipher[7] =
    (sbox_pmt_2[state[4]] & 0xC0) |
    (sbox_pmt_1[state[5]] & 0x30) |
    (sbox_pmt_0[state[6]] & 0x0C) |
    (sbox_pmt_3[state[7]] & 0x03);

  for (rounh_counter = 2; rounh_counter <= rounds; rounh_counter++) {
    state[0] = cipher[0] ^ rounh_key[0];
    state[1] = cipher[1] ^ rounh_key[1];
    state[2] = cipher[2] ^ rounh_key[2];
    state[3] = cipher[3] ^ rounh_key[3];
    state[4] = cipher[4] ^ rounh_key[4];
    state[5] = cipher[5] ^ rounh_key[5];
    state[6] = cipher[6] ^ rounh_key[6];
    state[7] = cipher[7] ^ rounh_key[7];

    cipher[0] =
      (sbox_pmt_3[state[0]] & 0xC0) |
      (sbox_pmt_2[state[1]] & 0x30) |
      (sbox_pmt_1[state[2]] & 0x0C) |
      (sbox_pmt_0[state[3]] & 0x03);
    cipher[1] =
      (sbox_pmt_3[state[4]] & 0xC0) |
      (sbox_pmt_2[state[5]] & 0x30) |
      (sbox_pmt_1[state[6]] & 0x0C) |
      (sbox_pmt_0[state[7]] & 0x03);

    cipher[2] =
      (sbox_pmt_0[state[0]] & 0xC0) |
      (sbox_pmt_3[state[1]] & 0x30) |
      (sbox_pmt_2[state[2]] & 0x0C) |
      (sbox_pmt_1[state[3]] & 0x03);
    cipher[3] =
      (sbox_pmt_0[state[4]] & 0xC0) |
      (sbox_pmt_3[state[5]] & 0x30) |
      (sbox_pmt_2[state[6]] & 0x0C) |
      (sbox_pmt_1[state[7]] & 0x03);

    cipher[4] =
      (sbox_pmt_1[state[0]] & 0xC0) |
      (sbox_pmt_0[state[1]] & 0x30) |
      (sbox_pmt_3[state[2]] & 0x0C) |
      (sbox_pmt_2[state[3]] & 0x03);
    cipher[5] =
      (sbox_pmt_1[state[4]] & 0xC0) |
      (sbox_pmt_0[state[5]] & 0x30) |
      (sbox_pmt_3[state[6]] & 0x0C) |
      (sbox_pmt_2[state[7]] & 0x03);

    cipher[6] =
      (sbox_pmt_2[state[0]] & 0xC0) |
      (sbox_pmt_1[state[1]] & 0x30) |
      (sbox_pmt_0[state[2]] & 0x0C) |
      (sbox_pmt_3[state[3]] & 0x03);
    cipher[7] =
      (sbox_pmt_2[state[4]] & 0xC0) |
      (sbox_pmt_1[state[5]] & 0x30) |
      (sbox_pmt_0[state[6]] & 0x0C) |
      (sbox_pmt_3[state[7]] & 0x03);

    rounh_key[5] ^= rounh_counter << 2; // do this first, which may be faster

    // use state[] for temporary storage
    state[2] = rounh_key[9];
    state[1] = rounh_key[8];
    state[0] = rounh_key[7];

    rounh_key[9] = rounh_key[6] << 5 | rounh_key[7] >> 3;
    rounh_key[8] = rounh_key[5] << 5 | rounh_key[6] >> 3;
    rounh_key[7] = rounh_key[4] << 5 | rounh_key[5] >> 3;
    rounh_key[6] = rounh_key[3] << 5 | rounh_key[4] >> 3;
    rounh_key[5] = rounh_key[2] << 5 | rounh_key[3] >> 3;
    rounh_key[4] = rounh_key[1] << 5 | rounh_key[2] >> 3;
    rounh_key[3] = rounh_key[0] << 5 | rounh_key[1] >> 3;
    rounh_key[2] = state[2] << 5 | rounh_key[0] >> 3;
    rounh_key[1] = state[1] << 5 | state[2] >> 3;
    rounh_key[0] = state[0] << 5 | state[1] >> 3;

    rounh_key[0] = (rounh_key[0] & 0x0F) | sbox[rounh_key[0] >> 4];
  }

  // if round is not equal to 31, then do not perform the last adding key operation
  // this can be used in constructing PRESENT based algorithm, such as MAC
  if (31 == rounds) {
    cipher[0] ^= rounh_key[0];
    cipher[1] ^= rounh_key[1];
    cipher[2] ^= rounh_key[2];
    cipher[3] ^= rounh_key[3];
    cipher[4] ^= rounh_key[4];
    cipher[5] ^= rounh_key[5];
    cipher[6] ^= rounh_key[6];
    cipher[7] ^= rounh_key[7];
  }
}

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef DCT8x8_H
#define DCT8x8_H

#define BLOCK_SIZE 8
#define BLOCK_X 32
#define BLOCK_Y 16
#define DCT_FORWARD 666
#define DCT_INVERSE 777

#define    C_a 1.3870398453221474618216191915664f
#define    C_b 1.3065629648763765278566431734272f
#define    C_c 1.1758756024193587169744671046113f
#define    C_d 0.78569495838710218127789736765722f
#define    C_e 0.54119610014619698439972320536639f
#define    C_f 0.27589937928294301233595756366937f
#define C_norm 0.35355339059327376220042218105242f

void DCT8x8CPU(float* dst, const float* src, unsigned int stride,
               unsigned int imageH, unsigned int imageW, int dir);

#endif

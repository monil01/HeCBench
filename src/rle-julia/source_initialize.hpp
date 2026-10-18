/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 * Test of DeviceRunLengthEncode
 ******************************************************************************/


template <typename T>
void Initialize(int entropy_reduction, T *h_in, int num_items, int max_segment)
{
  unsigned int max_int = (unsigned int)-1;

  int key = 0;
  int i   = 0;
  while (i < num_items)
  {
    // Select number of repeating occurrences for the current run
    int repeat;
    if (max_segment < 0)
    {
      repeat = num_items;
    }
    else if (max_segment < 2)
    {
      repeat = 1;
    }
    else
    {
      std::mt19937 mt(19937+i);
      std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
      repeat = dist(mt);
      for (int n = 1; n <= entropy_reduction; n++) {
        repeat &= dist(mt);
      }
 
      repeat = (int)((double(repeat) * double(max_segment)) / double(max_int));
      repeat = max(1, repeat);
    }

    int j = i;
    while (j < min(i + repeat, num_items))
    {
      h_in[j] = key;
      j++;
    }

    i = j;
    key++;
  }

  if (g_verbose)
  {
    printf("Input:\n");
    DisplayResults(h_in, num_items);
    printf("\n\n");
  }
}


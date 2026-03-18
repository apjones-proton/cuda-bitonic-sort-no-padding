/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ------------------------------------------------------------------------
 * Modifications:
 * Copyright (c) 2026, Andrew Jones. All rights reserved.
 *
 * This file contains modifications to NVIDIA's sortingNetworks sample.
 * The changes implement a padding-free bitonic sort for arbitrary
 * (including odd and non-power-of-2) input sizes by modifying only
 * the comparison direction logic using a small number of extra
 * arithmetic and bitwise operations.
 *
 * These modifications are released under the same BSD 3-clause license.
 *
 */

// Based on http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/bitonicen.htm

#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>

#include "sortingNetworks_common.cuh"
#include "sortingNetworks_common.h"

////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
bitonicSortSharedAnySize(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint arrayLength, uint dir)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Shared memory storage for one or more short vectors
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    // Offset to the beginning of subbatch and load data
	#define OFFSET (blockIdx.x * SHARED_SIZE_LIMIT)
	#define STRIDE (SHARED_SIZE_LIMIT / 2)
	d_SrcKey += OFFSET + threadIdx.x;
	d_SrcVal += OFFSET + threadIdx.x;
	d_DstKey += OFFSET + threadIdx.x;
	d_DstVal += OFFSET + threadIdx.x;
	if (OFFSET + threadIdx.x < arrayLength) {
		s_key[threadIdx.x + 0]      = d_SrcKey[0];
		s_val[threadIdx.x + 0]      = d_SrcVal[0];
	}
	if (OFFSET + threadIdx.x + STRIDE < arrayLength) {
		s_key[threadIdx.x + STRIDE] = d_SrcKey[STRIDE];
		s_val[threadIdx.x + STRIDE] = d_SrcVal[STRIDE];
	}
	uint thingy = UMAD(blockIdx.x, blockDim.x, threadIdx.x) ^ ((arrayLength - 1) / 2);

	if (OFFSET + SHARED_SIZE_LIMIT <= arrayLength) {
		for (uint size = 2; size <= SHARED_SIZE_LIMIT && size/2 < arrayLength; size <<= 1) {
			// Bitonic merge
			uint ddd = dir ^ ((thingy & (size / 2)) != 0);
			for (uint stride = size / 2; stride > 0; stride >>= 1) {
				cg::sync(cta);
				uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
				Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
			}
		}
	} else {
		for (uint size = 2; size <= SHARED_SIZE_LIMIT && size/2 < arrayLength; size <<= 1) {
			// Bitonic merge
			uint ddd = dir ^ ((thingy & (size / 2)) != 0);
			for (uint stride = size / 2; stride > 0; stride >>= 1) {
				cg::sync(cta);
				uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
				if (OFFSET + pos + stride < arrayLength)
					Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
			}
		}
	}

	cg::sync(cta);
	if (OFFSET + threadIdx.x < arrayLength) {
		d_DstKey[0]      = s_key[threadIdx.x + 0];
		d_DstVal[0]      = s_val[threadIdx.x + 0];
	}
	if (OFFSET + threadIdx.x + STRIDE < arrayLength) {
		d_DstKey[STRIDE] = s_key[threadIdx.x + STRIDE];
		d_DstVal[STRIDE] = s_val[threadIdx.x + STRIDE];
	}
	#undef STRIDE
	#undef OFFSET
}


// Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
__global__ void bitonicMergeGlobalAnySize(uint *d_DstKey,
										  uint *d_DstVal,
										  uint *d_SrcKey,
										  uint *d_SrcVal,
										  uint  arrayLength,
										  uint  size,
										  uint  stride,
										  uint  dir)
{
    uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
    uint thingy        = global_comparatorI ^ ((arrayLength - 1) / 2);

    // Bitonic merge
    uint ddd = dir ^ ((thingy & (size / 2)) != 0);
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));
	if (pos + stride >= arrayLength) {
		if (pos < arrayLength) {
			d_DstKey[pos] = d_SrcKey[pos];
			d_DstVal[pos] = d_SrcVal[pos];
		}
		return;
	}

    uint keyA = d_SrcKey[pos + 0];
    uint valA = d_SrcVal[pos + 0];
    uint keyB = d_SrcKey[pos + stride];
    uint valB = d_SrcVal[pos + stride];

    Comparator(keyA, valA, keyB, valB, ddd);

    d_DstKey[pos + 0]      = keyA;
    d_DstVal[pos + 0]      = valA;
    d_DstKey[pos + stride] = keyB;
    d_DstVal[pos + stride] = valB;
}

// Combined bitonic merge steps for
// size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__ void bitonicMergeSharedAnySize(uint *d_DstKey,
										  uint *d_DstVal,
										  uint *d_SrcKey,
										  uint *d_SrcVal,
										  uint  arrayLength,
										  uint  size,
										  uint  dir)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Shared memory storage for current subarray
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

	#define OFFSET (blockIdx.x * SHARED_SIZE_LIMIT)
	#define STRIDE (SHARED_SIZE_LIMIT / 2)
    d_SrcKey += OFFSET + threadIdx.x;
    d_SrcVal += OFFSET + threadIdx.x;
    d_DstKey += OFFSET + threadIdx.x;
    d_DstVal += OFFSET + threadIdx.x;
	if (OFFSET + threadIdx.x < arrayLength) {
		s_key[threadIdx.x + 0]      = d_SrcKey[0];
		s_val[threadIdx.x + 0]      = d_SrcVal[0];
	}
	if (OFFSET + threadIdx.x + STRIDE < arrayLength) {
		s_key[threadIdx.x + STRIDE] = d_SrcKey[STRIDE];
		s_val[threadIdx.x + STRIDE] = d_SrcVal[STRIDE];
	}
    // Bitonic merge
    uint thingy = UMAD(blockIdx.x, blockDim.x, threadIdx.x) ^ ((arrayLength - 1) / 2);
    uint ddd    = dir ^ ((thingy & (size / 2)) != 0);

	if (OFFSET + SHARED_SIZE_LIMIT <= arrayLength) {
		for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
			cg::sync(cta);
			uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
			Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
		}
    } else {
		for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
			cg::sync(cta);
			uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
			if (OFFSET + pos + stride < arrayLength)
				Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
		}
	}

			cg::sync(cta);
	if (OFFSET + threadIdx.x < arrayLength) {
		d_DstKey[0]      = s_key[threadIdx.x + 0];
		d_DstVal[0]      = s_val[threadIdx.x + 0];
	}
	if (OFFSET + threadIdx.x + STRIDE < arrayLength) {
		d_DstKey[STRIDE] = s_key[threadIdx.x + STRIDE];
		d_DstVal[STRIDE] = s_val[threadIdx.x + STRIDE];
	}
	#undef STRIDE
	#undef OFFSET
}

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////

extern "C" uint
bitonicSortAnySize(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint arrayLength, uint dir)
{
    // Nothing to sort
    if (arrayLength < 2)
        return 0;

    dir = (dir != 0);

    uint blockCount  = (arrayLength + SHARED_SIZE_LIMIT - 1) / SHARED_SIZE_LIMIT;
	//uint threadCount = std::min(SHARED_SIZE_LIMIT / 2, (arrayLength + 1) / 2);
	uint threadCount = SHARED_SIZE_LIMIT / 2;

	bitonicSortSharedAnySize<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);

    if (arrayLength > SHARED_SIZE_LIMIT) {
        for (uint size = 2 * SHARED_SIZE_LIMIT; size/2 < arrayLength; size <<= 1)
            for (unsigned stride = size / 2; stride > 0; stride >>= 1)
                if (stride >= SHARED_SIZE_LIMIT) {
                    bitonicMergeGlobalAnySize<<<blockCount, threadCount>>>(
							d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, stride, dir);
                }
                else {
					bitonicMergeSharedAnySize<<<blockCount, threadCount>>>(
							d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, dir);
                    break;
                }
    }

    return threadCount;
}

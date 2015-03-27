/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE

	__shared__ float temp[BLOCK_SIZE];
	
	int vec_pos = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = gridDim.x * BLOCK_SIZE;
	float x;
	float y;
	
	if (vec_pos < size) {
	
	if (vec_pos + offset < size) {
		x = in[vec_pos];
		y = in[vec_pos + offset];
		temp[threadIdx.x] = fma(1.0f, x, y);
	}
	else {
		temp[threadIdx.x] = in[vec_pos];
	}
	
	__syncthreads();
	
	for (int stride = 1; stride <= (BLOCK_SIZE / 2); stride *= 2) {
		if (threadIdx.x + stride < BLOCK_SIZE) {
			x = temp[threadIdx.x];
			y = temp[threadIdx.x + stride];
			temp[threadIdx.x] = fma(1.0f, x, y);
			__syncthreads();
		}
	}
	
	__syncthreads();
	
	out[blockIdx.x] = temp[0];
	}
}

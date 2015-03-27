/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void preScanKernel(float *out, float *in, unsigned size, float *sum)
{

	__shared__ float temp[BLOCK_SIZE*2];
	
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	
	int globalPos = bx * (BLOCK_SIZE*2) + tx;
	
	int pout = 0;
	int pin  = 1;
	float tempsum;
	float tempsum2;
	float x;
	float y;
	
	// populate temp array without shifting because we need the final value and then shift everyone
	if (globalPos < size) {
		temp[pout*BLOCK_SIZE + tx] = in[globalPos];
	}
	else {
		temp[pout*BLOCK_SIZE + tx] = 0; // pad with zeroes
	}
	
	__syncthreads();
	
	// do the scan on the first half
	int stride;
	for (stride = 1; stride < BLOCK_SIZE; stride *= 2) {
		pout = 1 - pout;
		pin  = 1 - pout;	// swap double buffer so that threads aren't accessing same data as what's being written (mutex sorta)
		
		if (tx >= stride) {
			x = temp[pin*BLOCK_SIZE + tx];
			y = temp[pin*BLOCK_SIZE + (tx - stride)];
			temp[pout*BLOCK_SIZE + tx] = __fadd_rd(x, y);
		}
		else {
			temp[pout*BLOCK_SIZE + tx] = temp[pin*BLOCK_SIZE + tx];
		}
		__syncthreads();
	}

	// copy out last element into sum and then copy everything over shifted one
	tempsum = temp[pout*BLOCK_SIZE + (BLOCK_SIZE - 1)];
	
	if (globalPos < size) {
		out[globalPos] = (tx > 0) ? temp[pout*BLOCK_SIZE + (tx - 1)] : 0;
	}
	__syncthreads();
	// adjust everything to do the second half
	globalPos = bx * (BLOCK_SIZE*2) + (tx + BLOCK_SIZE);
	pout = 0;
	pin  = 1;
	
	if (globalPos < size) {
		temp[pout*BLOCK_SIZE + tx] = in[globalPos];
	}
	else {
		temp[pout*BLOCK_SIZE + tx] = 0;
	}
	
	__syncthreads();
	// add sum to the first element
	if (tx == 0) {
		x = temp[tx];
		y = tempsum;
		temp[tx] = __fadd_rd(x, y);
	}
	
	// do the scan on the second half
	
	for (stride = 1; stride < BLOCK_SIZE; stride *= 2) {
		pout = 1 - pout;
		pin  = 1 - pout;	// swap double buffer so that threads aren't accessing same data as what's being written (mutex sorta)
		
		if (tx >= stride) {
			x = temp[pin*BLOCK_SIZE + tx];
			y = temp[pin*BLOCK_SIZE + (tx - stride)];
			temp[pout*BLOCK_SIZE + tx] = __fadd_rd(x, y);
		}
		else {
			temp[pout*BLOCK_SIZE + tx] = temp[pin*BLOCK_SIZE + tx];
		}
		__syncthreads();
	}

	__syncthreads();
	// copy out last element into sum and then copy everything over shifted one
	tempsum2 = temp[pout*BLOCK_SIZE + (BLOCK_SIZE - 1)];
	
	if (sum != NULL) {
	// sum is finally copied out to the sum array
		sum[blockIdx.x] = tempsum2;
	}
	
	if (globalPos < size) {
		if (tx == 0) {
			out[globalPos] = tempsum;
		}
		else {
			out[globalPos] = temp[pout*BLOCK_SIZE + (tx - 1)];
		}
	}

}


__global__ void addKernel(float *out, float *sum, unsigned size)
{
	int globalPos = blockIdx.x * (BLOCK_SIZE*2) + threadIdx.x;
	float x;
	float y;
	
	if (globalPos < size) {
		x = out[globalPos];
		y = sum[blockIdx.x];
		out[globalPos] = __fadd_rd(x, y);
	}
	
	globalPos = blockIdx.x * (BLOCK_SIZE*2) + (threadIdx.x + BLOCK_SIZE);
	
	if (globalPos < size) {
		x = out[globalPos];
		y = sum[blockIdx.x];
		out[globalPos] = __fadd_rd(x, y);
	}
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, unsigned in_size)
{
    // INSERT CODE HERE
	float *sum;
	unsigned num_blocks;
	cudaError_t cuda_ret;
	dim3 dim_grid, dim_block;

	num_blocks = in_size/(BLOCK_SIZE*2);
	if(in_size%(BLOCK_SIZE*2) !=0) num_blocks++;

	dim_block.x = BLOCK_SIZE; dim_block.y = 1; dim_block.z = 1;
	dim_grid.x = num_blocks; dim_grid.y = 1; dim_grid.z = 1;

	if(num_blocks > 1) {
		cuda_ret = cudaMalloc((void**)&sum, num_blocks*sizeof(float));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

		preScanKernel<<<dim_grid, dim_block>>>(out, in, in_size, sum);
		preScan(sum, sum, num_blocks);
		addKernel<<<dim_grid, dim_block>>>(out, sum, in_size);

		cudaFree(sum);
	}
	else
		preScanKernel<<<dim_grid, dim_block>>>(out, in, in_size, NULL);
}


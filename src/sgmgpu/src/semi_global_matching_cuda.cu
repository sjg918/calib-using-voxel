
//original paper:
//Embedded Real-time Stereo Estimation via Semi-Global Matching on the{GPU}

//original code:
//https://github.com/dhernandez0/sgm

#include "configuration.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


/*************************************
GPU Side defines (ASM instructions)
**************************************/

// output temporal carry in internal register
#define UADD__CARRY_OUT(c, a, b) \
  asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(c) : "r"(a) , "r"(b))

// add & output with temporal carry of internal register
#define UADD__IN_CARRY_OUT(c, a, b) \
  asm volatile("addc.cc.u32 %0, %1, %2;" : "=r"(c) : "r"(a) , "r"(b))

// add with temporal carry of internal register
#define UADD__IN_CARRY(c, a, b) \
  asm volatile("addc.u32 %0, %1, %2;" : "=r"(c) : "r"(a) , "r"(b))

// packing and unpacking: from uint64_t to uint2
#define V2S_B64(v,s) \
  asm("mov.b64 %0, {%1,%2};" : "=l"(s) : "r"(v.x), "r"(v.y))

// packing and unpacking: from uint2 to uint64_t
#define S2V_B64(s,v) \
  asm("mov.b64 {%0,%1}, %2;" : "=r"(v.x), "=r"(v.y) : "l"(s))


/*************************************
DEVICE side basic block primitives
**************************************/



__inline__ __device__ int shfl_32(int scalarValue, const int lane) {
	#if FERMI
		return __emulated_shfl(scalarValue, (uint32_t)lane);
	#else
		return __shfl_sync(0xffffffff, scalarValue, lane);
	#endif
}

__inline__ __device__ int shfl_up_32(int scalarValue, const int n) {
	#if FERMI
		int lane = threadIdx.x % WARP_SIZE;
		lane -= n;
		return shfl_32(scalarValue, lane);
	#else
		return __shfl_up_sync(0xffffffff, scalarValue, n);
	#endif
}

__inline__ __device__ int shfl_down_32(int scalarValue, const int n) {
	#if FERMI
		int lane = threadIdx.x % WARP_SIZE;
		lane += n;
		return shfl_32(scalarValue, lane);
	#else
		return __shfl_down_sync(0xffffffff, scalarValue, n);
	#endif
}

__inline__ __device__ int shfl_xor_32(int scalarValue, const int n) {
	#if FERMI
		int lane = threadIdx.x % WARP_SIZE;
		lane = lane ^ n;
		return shfl_32(scalarValue, lane);
	#else
		return __shfl_xor_sync(0xffffffff, scalarValue, n);
	#endif
}

__device__ __forceinline__ uint32_t ld_gbl_ca(const __restrict__ uint32_t *addr) {
	uint32_t return_value;
	asm("ld.global.ca.u32 %0, [%1];" : "=r"(return_value) : "l"(addr));
	return return_value;
}

__device__ __forceinline__ uint32_t ld_gbl_cs(const __restrict__ uint32_t *addr) {
	uint32_t return_value;
	asm("ld.global.cs.u32 %0, [%1];" : "=r"(return_value) : "l"(addr));
	return return_value;
}

__device__ __forceinline__ void st_gbl_wt(const __restrict__ uint32_t *addr, const uint32_t value) {
	asm("st.global.wt.u32 [%0], %1;" :: "l"(addr), "r"(value));
}

__device__ __forceinline__ void st_gbl_cs(const __restrict__ uint32_t *addr, const uint32_t value) {
	asm("st.global.cs.u32 [%0], %1;" :: "l"(addr), "r"(value));
}

__device__ __forceinline__ uint32_t gpu_get_sm_idx(){
	uint32_t smid;
	asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
	return(smid);
}

__device__ __forceinline__ void uint32_to_uchars(const uint32_t s, int *u1, int *u2, int *u3, int *u4) {
	//*u1 = s & 0xff;
	*u1 = __byte_perm(s, 0, 0x4440);
	//*u2 = (s>>8) & 0xff;
	*u2 = __byte_perm(s, 0, 0x4441);
	//*u3 = (s>>16) & 0xff;
	*u3 = __byte_perm(s, 0, 0x4442);
	//*u4 = s>>24;
	*u4 = __byte_perm(s, 0, 0x4443);
}

__device__ __forceinline__ uint32_t uchars_to_uint32(int u1, int u2, int u3, int u4) {
	//return u1 | (u2<<8) | (u3<<16) | (u4<<24);
	//return __byte_perm(u1, u2, 0x7740) + __byte_perm(u3, u4, 0x4077);
	return u1 | (u2<<8) | __byte_perm(u3, u4, 0x4077);
}

__device__ __forceinline__ uint32_t uchar_to_uint32(int u1) {
	return __byte_perm(u1, u1, 0x0);
}

__device__ __forceinline__ unsigned int vcmpgeu4(unsigned int a, unsigned int b) {
    unsigned int r, c;
    c = a-b;
    asm ("prmt.b32 %0,%1,0,0xba98;" : "=r"(r) : "r"(c));// build mask from msbs
    return r;           // byte-wise unsigned gt-eq comparison with mask result
}

__device__ __forceinline__ unsigned int vminu4(unsigned int a, unsigned int b) {
    unsigned int r, s;
    s = vcmpgeu4 (b, a);// mask = 0xff if a >= b
    r = a & s;          // select a when b >= a
    s = b & ~s;         // select b when b < a
    r = r | s;          // combine byte selections
    return r;
}

__device__ __forceinline__ void print_uchars(const char* str, const uint32_t s) {
	int u1, u2, u3, u4;
	uint32_to_uchars(s, &u1, &u2, &u3, &u4);
	printf("%s: %d %d %d %d\n", str, u1, u2, u3, u4);
}

template<class T>
__device__ __forceinline__ int popcount(T n) {
#if CSCT or CSCT_RECOMPUTE
	return __popc(n);
#else
	return __popcll(n);
#endif
}

__inline__ __device__ uint8_t minu8_index4(int *min_idx, const uint8_t val1, const int dis, const uint8_t val2, const int dis2, const uint8_t val3, const int dis3, const uint8_t val4, const int dis4) {
	int min_idx1 = dis;
	uint8_t min1 = val1;
	if(val1 > val2) {
		min1 = val2;
		min_idx1 = dis2;
	}

	int min_idx2 = dis3;
	uint8_t min2 = val3;
	if(val3 > val4) {
		min2 = val4;
		min_idx2 = dis4;
	}

	uint8_t minval = min1;
	*min_idx = min_idx1;
	if(min1 > min2) {
		minval = min2;
		*min_idx = min_idx2;
	}
	return minval;
}

__inline__ __device__ uint8_t minu8_index8(int *min_idx, const uint8_t val1, const int dis, const uint8_t val2, const int dis2, const uint8_t val3, const int dis3, const uint8_t val4, const int dis4, const uint8_t val5, const int dis5, const uint8_t val6, const int dis6, const uint8_t val7, const int dis7, const uint8_t val8, const int dis8) {
	int min_idx1, min_idx2;
	uint8_t minval1, minval2;

	minval1 = minu8_index4(&min_idx1, val1, dis, val2, dis2, val3, dis3, val4, dis4);
	minval2 = minu8_index4(&min_idx2, val5, dis5, val6, dis6, val7, dis7, val8, dis8);

	*min_idx = min_idx1;
	uint8_t minval = minval1;
	if(minval1 > minval2) {
		*min_idx = min_idx2;
		minval = minval2;
	}
	return minval;
}

__inline__ __device__ int warpReduceMinIndex2(int *val, int idx) {
	for(int d = 1; d < WARP_SIZE; d *= 2) {
		int tmp = shfl_xor_32(*val, d);
		int tmp_idx = shfl_xor_32(idx, d);
		if(*val > tmp) {
			*val = tmp;
			idx = tmp_idx;
		}
	}
	return idx;
}

__inline__ __device__ int warpReduceMinIndex(int val, int idx) {
	for(int d = 1; d < WARP_SIZE; d *= 2) {
		int tmp = shfl_xor_32(val, d);
		int tmp_idx = shfl_xor_32(idx, d);
		if(val > tmp) {
			val = tmp;
			idx = tmp_idx;
		}
	}
	return idx;
}

__inline__ __device__ int warpReduceMin(int val) {
	val = min(val, shfl_xor_32(val, 1));
	val = min(val, shfl_xor_32(val, 2));
	val = min(val, shfl_xor_32(val, 4));
	val = min(val, shfl_xor_32(val, 8));
	val = min(val, shfl_xor_32(val, 16));
	return val;
}

__inline__ __device__ int blockReduceMin(int val) {
	static __shared__ int shared[WARP_SIZE]; // Shared mem for WARP_SIZE partial sums
	const int lane = threadIdx.x % WARP_SIZE;
	const int wid = threadIdx.x / WARP_SIZE;

	val = warpReduceMin(val);     // Each warp performs partial reduction

	if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : INT_MAX;

	if (wid==0) val = warpReduceMin(val); //Final reduce within first warp

	return val;
}

__inline__ __device__ int blockReduceMinIndex(int val, int idx) {
	static __shared__ int shared_val[WARP_SIZE]; // Shared mem for WARP_SIZE partial mins
	static __shared__ int shared_idx[WARP_SIZE]; // Shared mem for WARP_SIZE indexes
	const int lane = threadIdx.x % WARP_SIZE;
	const int wid = threadIdx.x / WARP_SIZE;

	idx = warpReduceMinIndex2(&val, idx);     // Each warp performs partial reduction

	if (lane==0) {
		shared_val[wid]=val;
		shared_idx[wid]=idx;
	}

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared_val[lane] : INT_MAX;
	idx = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared_idx[lane] : INT_MAX;

	if (wid==0) {
		idx = warpReduceMinIndex2(&val, idx); //Final reduce within first warp
	}

	return idx;
}


__inline__ __device__ bool blockAny(bool local_condition) {
	__shared__ bool conditions[WARP_SIZE];
	const int lane = threadIdx.x % WARP_SIZE;
	const int wid = threadIdx.x / WARP_SIZE;

	local_condition = __any_sync(0xffffffff, local_condition);     // Each warp performs __any

	if (lane==0) {
		conditions[wid]=local_condition;
	}

	__syncthreads();              // Wait for all partial __any

	//read from shared memory only if that warp existed
	local_condition = (threadIdx.x < blockDim.x / WARP_SIZE) ? conditions[lane] : false;

	if (wid==0) {
		local_condition = __any_sync(0xffffffff, local_condition); //Final __any within first warp
	}

	return local_condition;
}


__global__ void 
__launch_bounds__(1024, 2)
CenterSymmetricCensusKernelSM2(const uint8_t *im, const uint8_t *im2, uint32_t *transform, uint32_t *transform2, const int rows, const int cols) {
	const int idx = blockIdx.x*blockDim.x+threadIdx.x;
	const int idy = blockIdx.y*blockDim.y+threadIdx.y;

	const int win_cols = (32+LEFT*2); // 32+4*2 = 40
	const int win_rows = (32+TOP*2); // 32+3*2 = 38

	__shared__ uint8_t window[win_cols*win_rows];
	__shared__ uint8_t window2[win_cols*win_rows];

	const int id = threadIdx.y*blockDim.x+threadIdx.x;
	const int sm_row = id / win_cols;
	const int sm_col = id % win_cols;

	const int im_row = blockIdx.y*blockDim.y+sm_row-TOP;
	const int im_col = blockIdx.x*blockDim.x+sm_col-LEFT;
	const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
	window[sm_row*win_cols+sm_col] = boundaries ? im[im_row*cols+im_col] : 0;
	window2[sm_row*win_cols+sm_col] = boundaries ? im2[im_row*cols+im_col] : 0;

	// Not enough threads to fill window and window2
	const int block_size = blockDim.x*blockDim.y;
	if(id < (win_cols*win_rows-block_size)) {
		const int id = threadIdx.y*blockDim.x+threadIdx.x+block_size;
		const int sm_row = id / win_cols;
		const int sm_col = id % win_cols;

		const int im_row = blockIdx.y*blockDim.y+sm_row-TOP;
		const int im_col = blockIdx.x*blockDim.x+sm_col-LEFT;
		const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
		window[sm_row*win_cols+sm_col] = boundaries ? im[im_row*cols+im_col] : 0;
		window2[sm_row*win_cols+sm_col] = boundaries ? im2[im_row*cols+im_col] : 0;
	}

	__syncthreads();
	uint32_t census = 0;
	uint32_t census2 = 0;
	if(idy < rows && idx < cols) {
			for(int k = 0; k < CENSUS_HEIGHT/2; k++) {
				for(int m = 0; m < CENSUS_WIDTH; m++) {
					const uint8_t e1 = window[(threadIdx.y+k)*win_cols+threadIdx.x+m];
					const uint8_t e2 = window[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];
					const uint8_t i1 = window2[(threadIdx.y+k)*win_cols+threadIdx.x+m];
					const uint8_t i2 = window2[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];

					const int shft = k*CENSUS_WIDTH+m;
					// Compare to the center
					uint32_t tmp = (e1 >= e2);
					// Shift to the desired position
					tmp <<= shft;
					// Add it to its place
					census |= tmp;
					// Compare to the center
					uint32_t tmp2 = (i1 >= i2);
					// Shift to the desired position
					tmp2 <<= shft;
					// Add it to its place
					census2 |= tmp2;
				}
			}
			if(CENSUS_HEIGHT % 2 != 0) {
				const int k = CENSUS_HEIGHT/2;
				for(int m = 0; m < CENSUS_WIDTH/2; m++) {
					const uint8_t e1 = window[(threadIdx.y+k)*win_cols+threadIdx.x+m];
					const uint8_t e2 = window[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];
					const uint8_t i1 = window2[(threadIdx.y+k)*win_cols+threadIdx.x+m];
					const uint8_t i2 = window2[(threadIdx.y+2*TOP-k)*win_cols+threadIdx.x+2*LEFT-m];

					const int shft = k*CENSUS_WIDTH+m;
					// Compare to the center
					uint32_t tmp = (e1 >= e2);
					// Shift to the desired position
					tmp <<= shft;
					// Add it to its place
					census |= tmp;
					// Compare to the center
					uint32_t tmp2 = (i1 >= i2);
					// Shift to the desired position
					tmp2 <<= shft;
					// Add it to its place
					census2 |= tmp2;
				}
			}

		transform[idy*cols+idx] = census;
		transform2[idy*cols+idx] = census2;
	}
}


__global__ void
HammingDistanceCostKernel (  const uint32_t *d_transform0, const uint32_t *d_transform1,
		uint8_t *d_cost, const int rows, const int cols ) {
	//const int Dmax=   blockDim.x;  // Dmax is CTA size
	const int y=      blockIdx.x;  // y is CTA Identifier
	const int THRid = threadIdx.x; // THRid is Thread Identifier

	__shared__ uint32_t SharedMatch[2*MAX_DISPARITY];
	__shared__ uint32_t SharedBase [MAX_DISPARITY];

	SharedMatch [MAX_DISPARITY+THRid] = d_transform1[y*cols+0];  // init position

	int n_iter = cols/MAX_DISPARITY;
	for (int ix=0; ix<n_iter; ix++) {
		const int x = ix*MAX_DISPARITY;
		SharedMatch [THRid]      = SharedMatch [THRid + MAX_DISPARITY];
		SharedMatch [THRid+MAX_DISPARITY] = d_transform1 [y*cols+x+THRid];
		SharedBase  [THRid]      = d_transform0 [y*cols+x+THRid];

		__syncthreads();
		for (int i=0; i<MAX_DISPARITY; i++) {
			const uint32_t base  = SharedBase [i];
			const uint32_t match = SharedMatch[(MAX_DISPARITY-1-THRid)+1+i];
			d_cost[(y*cols+x+i)*MAX_DISPARITY+THRid] = popcount( base ^ match );
		}
		__syncthreads();
	}
	// For images with cols not multiples of MAX_DISPARITY
	const int x = MAX_DISPARITY*(cols/MAX_DISPARITY);
	const int left = cols-x;
	if(left > 0) {
		SharedMatch [THRid]      = SharedMatch [THRid + MAX_DISPARITY];
		if(THRid < left) {
			SharedMatch [THRid+MAX_DISPARITY] = d_transform1 [y*cols+x+THRid];
			SharedBase  [THRid]      = d_transform0 [y*cols+x+THRid];
		}

		__syncthreads();
		for (int i=0; i<left; i++) {
			const uint32_t base  = SharedBase [i];
			const uint32_t match = SharedMatch[(MAX_DISPARITY-1-THRid)+1+i];
			d_cost[(y*cols+x+i)*MAX_DISPARITY+THRid] = popcount( base ^ match );
		}
		__syncthreads();
	}
}


template<int add_col, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationGenericIndexesIncrement(int *index, int *index_im, int *col, const int add_index, const int add_imindex) {
	*index += add_index;
	if(recompute || join_dispcomputation) {
		*index_im += add_imindex;
		if(recompute) {
			*col += add_col;
		}
	}
}

template<int add_index, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationDiagonalGenericIndexesIncrement(int *index, int *index_im, int *col, const int cols, const int initial_row, const int i, const int dis) {
	*col += add_index;
	if(add_index > 0 && *col >= cols) {
		*col = 0;
	} else if(*col < 0) {
		*col = cols-1;
	}
	*index = abs(initial_row-i)*cols*MAX_DISPARITY+*col*MAX_DISPARITY+dis;
	if(recompute || join_dispcomputation) {
		*index_im = abs(initial_row-i)*cols+*col;
	}
}

template<class T, int iter_type, int min_type, int dir_type, bool first_iteration, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationGenericIteration(int index, int index_im, int col, uint32_t *old_values, int *old_value1, int *old_value2, int *old_value3, int *old_value4, uint32_t *min_cost, uint32_t *min_cost_p2, uint8_t* d_cost, uint8_t *d_L, const int p1_vector, const int p2_vector, const T *_d_transform0, const T *_d_transform1, const int lane, const int MAX_PAD, const int dis, T *rp0, T *rp1, T *rp2, T *rp3, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const T __restrict__ *d_transform0 = _d_transform0;
	const T __restrict__ *d_transform1 = _d_transform1;
	uint32_t costs, next_dis, prev_dis;

	if(iter_type == ITER_NORMAL) {
		// First shuffle
		int prev_dis1 = shfl_up_32(*old_value4, 1);
		if(lane == 0) {
			prev_dis1 = MAX_PAD;
		}

		// Second shuffle
		int next_dis4 = shfl_down_32(*old_value1, 1);
		if(lane == 31) {
			next_dis4 = MAX_PAD;
		}

		// Shift + rotate
		//next_dis = __funnelshift_r(next_dis4, *old_values, 8);
		next_dis = __byte_perm(*old_values, next_dis4, 0x4321);
		prev_dis = __byte_perm(*old_values, prev_dis1, 0x2104);

		next_dis = next_dis + p1_vector;
		prev_dis = prev_dis + p1_vector;
	}
	if(recompute) {
		const int dif = col - dis;
		if(dir_type == DIR_LEFTRIGHT) {
			if(lane == 0) {
				// lane = 0 is dis = 0, no need to subtract dis
				*rp0 = d_transform1[index_im];
			}
		} else if(dir_type == DIR_RIGHTLEFT) {
			// First iteration, load D pixels
			if(first_iteration) {
				const uint4 right = reinterpret_cast<const uint4*>(&d_transform1[index_im-dis-3])[0];
				*rp3 = right.x;
				*rp2 = right.y;
				*rp1 = right.z;
				*rp0 = right.w;
			} else if(lane == 31 && dif >= 3) {
				*rp3 = d_transform1[index_im-dis-3];
			}
		} else {
	/*
			__shared__ T right_p[MAX_DISPARITY+32];
			const int warp_id = threadIdx.x / WARP_SIZE;
			if(warp_id < 5) {
				const int block_imindex = index_im - warp_id + 32;
				const int rp_index = warp_id*WARP_SIZE+lane;
				const int col_cpy = col-warp_id+32;
				right_p[rp_index] = ((col_cpy-(159-rp_index)) >= 0) ? ld_gbl_cs(&d_transform1[block_imindex-(159-rp_index)]) : 0;
			}*/

			__shared__ T right_p[128+32];
			const int warp_id = threadIdx.x / WARP_SIZE;
			const int block_imindex = index_im - warp_id + 2;
			const int rp_index = warp_id*WARP_SIZE+lane;
			const int col_cpy = col-warp_id+2;
			right_p[rp_index] = ((col_cpy-(129-rp_index)) >= 0) ? d_transform1[block_imindex-(129-rp_index)] : 0;
			right_p[rp_index+64] = ((col_cpy-(129-rp_index-64)) >= 0) ? d_transform1[block_imindex-(129-rp_index-64)] : 0;
			//right_p[rp_index+128] = ld_gbl_cs(&d_transform1[block_imindex-(129-rp_index-128)]);
			if(warp_id == 0) {
				right_p[128+lane] = ld_gbl_cs(&d_transform1[block_imindex-(129-lane)] );
			}
			__syncthreads();

			const int px = MAX_DISPARITY+warp_id-dis-1;
			*rp0 = right_p[px];
			*rp1 = right_p[px-1];
			*rp2 = right_p[px-2];
			*rp3 = right_p[px-3];
		}
		const T left_pixel = d_transform0[index_im];
		*old_value1 = popcount(left_pixel ^ *rp0);
		*old_value2 = popcount(left_pixel ^ *rp1);
		*old_value3 = popcount(left_pixel ^ *rp2);
		*old_value4 = popcount(left_pixel ^ *rp3);
		if(iter_type == ITER_COPY) {
			*old_values = uchars_to_uint32(*old_value1, *old_value2, *old_value3, *old_value4);
		} else {
			costs = uchars_to_uint32(*old_value1, *old_value2, *old_value3, *old_value4);
		}
		// Prepare for next iteration
		if(dir_type == DIR_LEFTRIGHT) {
			*rp3 = shfl_up_32(*rp3, 1);
		} else if(dir_type == DIR_RIGHTLEFT) {
			*rp0 = shfl_down_32(*rp0, 1);
		}
	} else {
		if(iter_type == ITER_COPY) {
			*old_values = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index]));
		} else {
			costs = ld_gbl_ca(reinterpret_cast<const uint32_t*>(&d_cost[index]));
		}
	}

	if(iter_type == ITER_NORMAL) {
		const uint32_t min1 = __vminu4(*old_values, prev_dis);
		const uint32_t min2 = __vminu4(next_dis, *min_cost_p2);
		const uint32_t min_prev = __vminu4(min1, min2);
		*old_values = costs + (min_prev - *min_cost);
	}
	if(iter_type == ITER_NORMAL || !recompute) {
		uint32_to_uchars(*old_values, old_value1, old_value2, old_value3, old_value4);
	}

	if(join_dispcomputation) {
		const uint32_t L0_costs = *((uint32_t*) (d_L0+index));
		const uint32_t L1_costs = *((uint32_t*) (d_L1+index));
		const uint32_t L2_costs = *((uint32_t*) (d_L2+index));
		#if pathA == 8
			const uint32_t L3_costs = *((uint32_t*) (d_L3+index));
			const uint32_t L4_costs = *((uint32_t*) (d_L4+index));
			const uint32_t L5_costs = *((uint32_t*) (d_L5+index));
			const uint32_t L6_costs = *((uint32_t*) (d_L6+index));
		#endif

		int l0_x, l0_y, l0_z, l0_w;
		int l1_x, l1_y, l1_z, l1_w;
		int l2_x, l2_y, l2_z, l2_w;
		#if pathA == 8
			int l3_x, l3_y, l3_z, l3_w;
			int l4_x, l4_y, l4_z, l4_w;
			int l5_x, l5_y, l5_z, l5_w;
			int l6_x, l6_y, l6_z, l6_w;
		#endif

		uint32_to_uchars(L0_costs, &l0_x, &l0_y, &l0_z, &l0_w);
		uint32_to_uchars(L1_costs, &l1_x, &l1_y, &l1_z, &l1_w);
		uint32_to_uchars(L2_costs, &l2_x, &l2_y, &l2_z, &l2_w);
		#if pathA == 8
			uint32_to_uchars(L3_costs, &l3_x, &l3_y, &l3_z, &l3_w);
			uint32_to_uchars(L4_costs, &l4_x, &l4_y, &l4_z, &l4_w);
			uint32_to_uchars(L5_costs, &l5_x, &l5_y, &l5_z, &l5_w);
			uint32_to_uchars(L6_costs, &l6_x, &l6_y, &l6_z, &l6_w);
		#endif

		#if pathA == 8
			const uint16_t val1 = l0_x + l1_x + l2_x + l3_x + l4_x + l5_x + l6_x + *old_value1;
			const uint16_t val2 = l0_y + l1_y + l2_y + l3_y + l4_y + l5_y + l6_y + *old_value2;
			const uint16_t val3 = l0_z + l1_z + l2_z + l3_z + l4_z + l5_z + l6_z + *old_value3;
			const uint16_t val4 = l0_w + l1_w + l2_w + l3_w + l4_w + l5_w + l6_w + *old_value4;
		#else
			const uint16_t val1 = l0_x + l1_x + l2_x + *old_value1;
			const uint16_t val2 = l0_y + l1_y + l2_y + *old_value2;
			const uint16_t val3 = l0_z + l1_z + l2_z + *old_value3;
			const uint16_t val4 = l0_w + l1_w + l2_w + *old_value4;
		#endif
		int min_idx1 = dis;
		uint16_t min1 = val1;
		if(val1 > val2) {
			min1 = val2;
			min_idx1 = dis+1;
		}

		int min_idx2 = dis+2;
		uint16_t min2 = val3;
		if(val3 > val4) {
			min2 = val4;
			min_idx2 = dis+3;
		}

		uint16_t minval = min1;
		int min_idx = min_idx1;
		if(min1 > min2) {
			minval = min2;
			min_idx = min_idx2;
		}

		const int min_warpindex = warpReduceMinIndex(minval, min_idx);
		if(lane == 0) {
			d_disparity[index_im] = min_warpindex;
		}
	} else {
		st_gbl_cs(reinterpret_cast<uint32_t*>(&d_L[index]), *old_values);
	}
	if(min_type == MIN_COMPUTE) {
		int min_cost_scalar = min(min(*old_value1, *old_value2), min(*old_value3, *old_value4));
		*min_cost = uchar_to_uint32(warpReduceMin(min_cost_scalar));
		*min_cost_p2 = *min_cost + p2_vector;
	}
}

template<class T, int add_col, int dir_type, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationGeneric(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int initial_row, const int initial_col, const int max_iter, const int cols, int add_index, const T *_d_transform0, const T *_d_transform1, const int add_imindex, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int lane = threadIdx.x % WARP_SIZE;
	const int dis = 4*lane;
	int index = initial_row*cols*MAX_DISPARITY+initial_col*MAX_DISPARITY+dis;
	int col, index_im;
	if(recompute || join_dispcomputation) {
		if(recompute) {
			col = initial_col;
		}
		index_im = initial_row*cols+initial_col;
	}

	const int MAX_PAD = UCHAR_MAX-P1;
	const uint32_t p1_vector = uchars_to_uint32(P1, P1, P1, P1);
	const uint32_t p2_vector = uchars_to_uint32(P2, P2, P2, P2);
	int old_value1;
	int old_value2;
	int old_value3;
	int old_value4;
	uint32_t min_cost, min_cost_p2, old_values;
	T rp0, rp1, rp2, rp3;

	if(recompute) {
		if(dir_type == DIR_LEFTRIGHT) {
			CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			for(int i = 4; i < max_iter-3; i+=4) {
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			}
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
		} else if(dir_type == DIR_RIGHTLEFT) {
			CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			for(int i = 4; i < max_iter-3; i+=4) {
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			}
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp1, &rp2, &rp3, &rp0, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp2, &rp3, &rp0, &rp1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp3, &rp0, &rp1, &rp2, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
		} else {
			CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			for(int i = 1; i < max_iter; i++) {
				CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
				CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
			}
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
		}
	} else {
		CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);

		for(int i = 1; i < max_iter; i++) {
			CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
		}
		CostAggregationGenericIndexesIncrement<add_col, recompute, join_dispcomputation>(&index, &index_im, &col, add_index, add_imindex);
		CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}

template<int add_index, class T, int dir_type, bool recompute, bool join_dispcomputation>
__device__ __forceinline__ void CostAggregationDiagonalGeneric(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int initial_row, const int initial_col, const int max_iter, const int col_nomin, const int col_copycost, const int cols, const T *_d_transform0, const T *_d_transform1, uint8_t* __restrict__ d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int lane = threadIdx.x % WARP_SIZE;
	const int dis = 4*lane;
	int col = initial_col;
	int index = initial_row*cols*MAX_DISPARITY+initial_col*MAX_DISPARITY+dis;
	int index_im;
	if(recompute || join_dispcomputation) {
		index_im = initial_row*cols+col;
	}
	const int MAX_PAD = UCHAR_MAX-P1;
	const uint32_t p1_vector = uchars_to_uint32(P1, P1, P1, P1);
	const uint32_t p2_vector = uchars_to_uint32(P2, P2, P2, P2);
	int old_value1;
	int old_value2;
	int old_value3;
	int old_value4;
	uint32_t min_cost, min_cost_p2, old_values;
	T rp0, rp1, rp2, rp3;

	CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, true, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	for(int i = 1; i < max_iter; i++) {
		CostAggregationDiagonalGenericIndexesIncrement<add_index, recompute, join_dispcomputation>(&index, &index_im, &col, cols, initial_row, i, dis);
		if(col == col_copycost) {
			CostAggregationGenericIteration<T, ITER_COPY, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
		} else {
			CostAggregationGenericIteration<T, ITER_NORMAL, MIN_COMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
		}
	}

	CostAggregationDiagonalGenericIndexesIncrement<add_index, recompute, join_dispcomputation>(&index, &index_im, &col, cols, max_iter, initial_row, dis);
	if(col == col_copycost) {
		CostAggregationGenericIteration<T, ITER_COPY, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	} else {
		CostAggregationGenericIteration<T, ITER_NORMAL, MIN_NOCOMPUTE, dir_type, false, recompute, join_dispcomputation>(index, index_im, col, &old_values, &old_value1, &old_value2, &old_value3, &old_value4, &min_cost, &min_cost_p2, d_cost, d_L, p1_vector, p2_vector, _d_transform0, _d_transform1, lane, MAX_PAD, dis, &rp0, &rp1, &rp2, &rp3, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}

template<class T>

__global__ void CostAggregationKernelDiagonalDownUpRightLeft(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int initial_col = cols - (blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE)) - 1;
	if(initial_col < cols) {
		const int initial_row = rows-1;
		const int add_index = -1;
		const int col_nomin = 0;
		const int col_copycost = cols-1;
		const int max_iter = rows-1;
		const bool recompute = false;
		const bool join_dispcomputation = false;

		CostAggregationDiagonalGeneric<add_index, T, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}

template<class T>
__global__ void CostAggregationKernelDiagonalDownUpLeftRight(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int initial_col = cols - (blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE)) - 1;
	if(initial_col >= 0) {
		const int initial_row = rows-1;
		const int add_index = 1;
		const int col_nomin = cols-1;
		const int col_copycost = 0;
		const int max_iter = rows-1;
		const bool recompute = false;
		const bool join_dispcomputation = false;

		CostAggregationDiagonalGeneric<add_index, T, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}

template<class T>

__global__ void CostAggregationKernelDiagonalUpDownRightLeft(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_col < cols) {
		const int initial_row = 0;
		const int add_index = -1;
		const int col_nomin = 0;
		const int col_copycost = cols-1;
		const int max_iter = rows-1;
		const bool recompute = false;
		const bool join_dispcomputation = true;

		CostAggregationDiagonalGeneric<add_index, T, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}

template<class T>

__global__ void CostAggregationKernelDiagonalUpDownLeftRight(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_col < cols) {
		const int initial_row = 0;
		const int add_index = 1;
		const int col_nomin = cols-1;
		const int col_copycost = 0;
		const int max_iter = rows-1;
		const bool recompute = false;
		const bool join_dispcomputation = false;

		CostAggregationDiagonalGeneric<add_index, T, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, col_nomin, col_copycost, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}

template<class T>
__global__ void CostAggregationKernelLeftToRight(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int initial_row = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_row < rows) {
		const int initial_col = 0;
		const int add_index = MAX_DISPARITY;
		const int add_imindex = 1;
		const int max_iter = cols-1;
		const int add_col = 1;
		const bool recompute = true;
		const bool join_dispcomputation = false;

		CostAggregationGeneric<T, add_col, DIR_LEFTRIGHT, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}

template<class T>
__global__ void CostAggregationKernelRightToLeft(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int initial_row = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_row < rows) {
		const int initial_col = cols-1;
		const int add_index = -MAX_DISPARITY;
		const int add_imindex = -1;
		const int max_iter = cols-1;
		const int add_col = -1;
		const bool recompute = true;
		const bool join_dispcomputation = false;

		CostAggregationGeneric<T, add_col, DIR_RIGHTLEFT, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}


template<class T>
__global__ void CostAggregationKernelRightToLeft_join(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int initial_row = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_row < rows) {
		const int initial_col = cols-1;
		const int add_index = -MAX_DISPARITY;
		const int add_imindex = -1;
		const int max_iter = cols-1;
		const int add_col = -1;
		const bool recompute = true;
		const bool join_dispcomputation = true;

		CostAggregationGeneric<T, add_col, DIR_RIGHTLEFT, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}


template<class T>
__global__ void CostAggregationKernelDownToUp_join(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);

	if(initial_col < cols) {
		const int initial_row = rows-1;
		const int add_index = -cols*MAX_DISPARITY;
		const int add_imindex = -cols;
		const int max_iter = rows-1;
		const int add_col = 0;
		const bool recompute = false;
		const bool join_dispcomputation = true;

		CostAggregationGeneric<T, add_col, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}


template<class T>
__global__ void CostAggregationKernelDownToUp(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);

	if(initial_col < cols) {
		const int initial_row = rows-1;
		const int add_index = -cols*MAX_DISPARITY;
		const int add_imindex = -cols;
		const int max_iter = rows-1;
		const int add_col = 0;
		const bool recompute = false;
		const bool join_dispcomputation = false;

		CostAggregationGeneric<T, add_col, DIR_DOWNUP, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}


template<class T>
//__launch_bounds__(64, 16)
__global__ void CostAggregationKernelUpToDown(uint8_t* d_cost, uint8_t *d_L, const int P1, const int P2, const int rows, const int cols, const T *d_transform0, const T *d_transform1, uint8_t* d_disparity, const uint8_t* d_L0, const uint8_t* d_L1, const uint8_t* d_L2, const uint8_t* d_L3, const uint8_t* d_L4, const uint8_t* d_L5, const uint8_t* d_L6, const int pathA) {
	const int initial_col = blockIdx.x*(blockDim.x/WARP_SIZE) + (threadIdx.x / WARP_SIZE);
	if(initial_col < cols) {
		const int initial_row = 0;
		const int add_index = cols*MAX_DISPARITY;
		const int add_imindex = cols;
		const int max_iter = rows-1;
		const int add_col = 0;
		const bool recompute = false;
		const bool join_dispcomputation = false;

		CostAggregationGeneric<T, add_col, DIR_UPDOWN, recompute, join_dispcomputation>(d_cost, d_L, P1, P2, initial_row, initial_col, max_iter, cols, add_index, d_transform0, d_transform1, add_imindex, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA);
	}
}

template<int n, typename T>
__inline__ __device__ void MedianFilter(const T* __restrict__ d_input, T* __restrict__ d_out, const uint32_t rows, const uint32_t cols) {
	const uint32_t idx = blockIdx.x*blockDim.x+threadIdx.x;
	const uint32_t row = idx / cols;
	const uint32_t col = idx % cols;
	T window[n*n];
	int half = n/2;

	if(row >= half && col >= half && row < rows-half && col < cols-half) {
		for(uint32_t i = 0; i < n; i++) {
			for(uint32_t j = 0; j < n; j++) {
				window[i*n+j] = d_input[(row-half+i)*cols+col-half+j];
			}
		}

		for(uint32_t i = 0; i < (n*n/2)+1; i++) {
			uint32_t min_idx = i;
			for(uint32_t j = i+1; j < n*n; j++) {
				if(window[j] < window[min_idx]) {
					min_idx = j;
				}
			}
			const T tmp = window[i];
			window[i] = window[min_idx];
			window[min_idx] = tmp;
		}
		d_out[idx] = window[n*n/2];
	} else if(row < rows && col < cols) {
		d_out[idx] = d_input[idx];
	}
}

__global__ void MedianFilter3x3(const uint8_t* __restrict__ d_input, uint8_t* __restrict__ d_out, const uint32_t rows, const uint32_t cols) {
	MedianFilter<3>(d_input, d_out, rows, cols);
}


void SemiGlobalMatchingCudaLauncher(
    const uint8_t * left, const uint8_t * right, uint32_t * d_transform0, uint32_t * d_transform1, uint8_t * d_cost,
    uint8_t * d_L0, uint8_t * d_L1, uint8_t * d_L2, uint8_t * d_L3, uint8_t * d_L4, uint8_t * d_L5, uint8_t * d_L6, uint8_t * d_L7,
    uint8_t * d_disparity, uint8_t * d_disparity_filtered_uchar,
    const int p1, const int p2, const int rows, const int cols, const int pathA
) {
    dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	dim3 grid_size;
	grid_size.x = (cols+block_size.x-1) / block_size.x;
	grid_size.y = (rows+block_size.y-1) / block_size.y;

    const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;
	const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;

    CenterSymmetricCensusKernelSM2<<<grid_size, block_size>>>(left, right, d_transform0, d_transform1, rows, cols);
    HammingDistanceCostKernel<<<rows, MAX_DISPARITY>>>(d_transform0, d_transform1, d_cost, rows, cols);
    
    if (pathA == 2) {
		CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ>>>(
        	d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
        );
		CostAggregationKernelRightToLeft_join<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ>>>(
        	d_cost, d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
        );
	}
	
	if (pathA == 4) {
		CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ>>>(
        	d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
        );
		CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ>>>(
        	d_cost, d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
        );
		CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE>>>(
        	d_cost, d_L2, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
        	);
		CostAggregationKernelDownToUp_join<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE>>>(
        	d_cost, d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
        	);
	}

	if (pathA == 8) {
		CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ>>>(
        	d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
        );
		CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ>>>(
        	d_cost, d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
        );
		CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE>>>(
        	d_cost, d_L2, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
        	);
		CostAggregationKernelDownToUp<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE>>>(
        	d_cost, d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
        	);
		CostAggregationKernelDiagonalDownUpLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE>>>(
			d_cost, d_L4, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
			);
		CostAggregationKernelDiagonalUpDownLeftRight<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE>>>(
			d_cost, d_L5, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
			);
		CostAggregationKernelDiagonalDownUpRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE>>>(
			d_cost, d_L6, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
			);
		CostAggregationKernelDiagonalUpDownRightLeft<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE>>>(
			d_cost, d_L7, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6, pathA
			);
	}

    MedianFilter3x3<<<(rows*cols+MAX_DISPARITY-1)/MAX_DISPARITY, MAX_DISPARITY>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);
}

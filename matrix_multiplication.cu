#include <cuda_runtime.h>
#include <stdio.h>

template <typename T> 
__global__ void matrix_multiplication_kernel(T const* A, T const* B, T* C, int M, int N, int K,
int A_col, int B_col, int C_col){
    size_t const col{threadIdx.x + blockDim.x * blockIdx.x};
    size_t const row{threadIdx.y + blockDim.y * blockIdx.y};

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int a = 0; a < N; a++) {
            sum += A[row * A_col + a] * B[a * B_col + col];
        }
        C[row * C_col + col] = sum;
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS,
size_t BLOCK_TILE_SKEW_SIZE_X=0U, size_t BLOCK_TILE_SKEW_SIZE_K=0U>
__device__ void load_data_to_shared_memory(T const* A, int A_col, T const* B, int B_col,
T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K],
T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X],
size_t thread_block_tile_idx, size_t thread_linear_id, size_t M, size_t N, size_t K){

    #pragma unroll
    for(size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U)/NUM_THREADS; load_idx++){
        size_t const A_thread_block_tile_row_idx{(thread_linear_id + load_idx * NUM_THREADS)/BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{(thread_linear_id + load_idx * NUM_THREADS)%BLOCK_TILE_SIZE_K};

        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y + A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + A_thread_block_tile_col_idx};

        T val{static_cast<T>(0)};
        if(A_row_idx < M && A_col_idx < N){
            val = A[A_row_idx * A_col + A_col_idx];
        }

        static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);

        A_thread_block_tile[A_thread_block_tile_row_idx][A_thread_block_tile_col_idx] = val;
    }

    #pragma unroll
    for(size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U)/NUM_THREADS; load_idx++){
        size_t const B_thread_block_tile_row_idx{(thread_linear_id + load_idx * NUM_THREADS)/BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{(thread_linear_id + load_idx * NUM_THREADS)%BLOCK_TILE_SIZE_X};

        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K + B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X + B_thread_block_tile_col_idx};

        T val{static_cast<T>(0)};
        if(B_row_idx < N && B_col_idx < K){
            val = B[B_row_idx * B_col + B_col_idx];
        }

        static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
        
        B_thread_block_tile[B_thread_block_tile_row_idx][B_thread_block_tile_col_idx] = val;
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K>
__global__ void matrix_mul(size_t M, size_t N, size_t K, T const* A, int A_col, T const* B, int B_col, T* C, int C_col){
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    size_t const thread_linear_id{threadIdx.y * blockDim.x + threadIdx.x};

    size_t const col{threadIdx.x + blockDim.x * blockIdx.x};
    size_t const row{threadIdx.y + blockDim.y * blockIdx.y};

    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(N + BLOCK_TILE_SIZE_K - 1)/BLOCK_TILE_SIZE_K};

    T sum{static_cast<T>(0)};
    for(size_t thread_block_tile{0U}; thread_block_tile<num_thread_block_tiles; thread_block_tile++){
        load_data_to_shared_memory<T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, A_col, B, B_col, A_thread_block_tile, B_thread_block_tile, thread_block_tile, thread_linear_id, M, N, K);
        __syncthreads();

        #pragma unroll
        for(size_t k_i{0U}; k_i<BLOCK_TILE_SIZE_K; k_i++){
            sum += A_thread_block_tile[threadIdx.y][k_i]*B_thread_block_tile[k_i][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < M && col < K){
        C[row*C_col+col]=sum;
    }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    printf("poop fart %d %d %d %d\n", BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS);

    dim3 threadsPerBlock(BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int A_col = N;
    int B_col = K;
    int C_col = K;

    //matrix_multiplication_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, A_col, B_col, C_col);
    matrix_mul<float, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
    <<<blocksPerGrid, threadsPerBlock>>>(M, N, K, A, A_col, B, B_col, C, C_col);
    cudaDeviceSynchronize();
}

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

template <typename T, size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y, size_t BLOCK_TILE_SIZE_K>
__global__ void matrix_mul(size_t M, size_t N, size_t K, T const* A, int A_col, T const* B, int B_col, T* C, int C_col){
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    size_t const thread_linear_id{threadIdx.y * blockDim.x + threadIdx.x};

    size_t const col{threadIdx.x + blockDim.x * blockIdx.x};
    size_t const row{threadIdx.y + blockDim.y * blockIdx.y};

    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_X][BLOCK_TILE_SIZE_Y];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];

    size_t const num_thread_block_tiles{(K + BLOCK_TILE_SIZE_K - 1)/BLOCK_TILE_SIZE_K};
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    printf("poop fart %d %d %d %d\n", BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K, NUM_THREADS);

    dim3 threadsPerBlock(BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int A_col = N;
    int B_col = K;
    int C_col = K;

    //matrix_multiplication_kernel<float><<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K, A_col, B_col, C_col);
    matrix_mul<float, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
    <<<blocksPerGrid, threadsPerBlock>>>(M, N, K, A, A_col, B, B_col, C, C_col);
    cudaDeviceSynchronize();
}


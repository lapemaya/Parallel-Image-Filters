#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;
#define MAX_KERNEL_SIZE 7 
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
             << " -> " << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)
__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// ============================================
// ============== CUDA KERNEL =================
// ============================================

template<int K>
__global__ void conv2d_kernel_optimized_K(
    const float* __restrict__ img,
    float* __restrict__ out,
    int H, int W,
    int tileSize)
{

    // ==== SHARED MEMORY TILE ====
    extern __shared__ float sharedMem[];
    float* s_tile = sharedMem;

    // ==== SETUP PARAMETERS ====
    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;

    constexpr int radius = K / 2;
    int sharedDim = tileSize + K - 1;
    int sharedPitch = sharedDim + 1; 

    int bx = (int)blockIdx.x * tileSize;
    int by = (int)blockIdx.y * tileSize;

    int gx = bx + tx - radius;
    int gy = by + ty - radius;

    // ==== LOAD IMAGE TILE TO SHARED MEMORY ====
    if (tx < sharedDim && ty < sharedDim) {
        if (gx >= 0 && gx < W && gy >= 0 && gy < H)
            s_tile[ty * sharedPitch + tx] = img[gy * W + gx];
        else
            s_tile[ty * sharedPitch + tx] = 0.0f;
    }
    __syncthreads();

    // ==== CONVOLUTION COMPUTE ====
    int ox = bx + tx;
    int oy = by + ty;
    if (tx < tileSize && ty < tileSize &&
        ox < W && oy < H) {
        float sum = 0.0f;
        #pragma unroll // Unroll for better performance
        for (int ky = 0; ky < K; ky++) {
            #pragma unroll // Unroll for better performance
            for (int kx = 0; kx < K; kx++) {
                float v = s_tile[(ty + ky) * sharedPitch + (tx + kx)];
                float w = d_kernel[ky * K + kx];
                sum += v * w;
            }
        }
    
    // ==== WRITE OUTPUT ON GLOBAL DEVICE ====
        out[oy * W + ox] = sum;
    }
}

// ==== IMAGE STRUCT AND LOADING FUNCTION ====
struct RGBImage {
    vector<vector<double>> r, g, b;
    int H, W;
};


RGBImage loadImageRGB(const string& path)
{
    Mat img = imread(path, IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Errore caricamento immagine\n";
        exit(EXIT_FAILURE);
    }

    RGBImage out;
    out.H = img.rows;
    out.W = img.cols;
    out.r.resize(img.rows, vector<double>(img.cols));
    out.g.resize(img.rows, vector<double>(img.cols));
    out.b.resize(img.rows, vector<double>(img.cols));

    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            Vec3b pixel = img.at<Vec3b>(r, c);
            out.b[r][c] = pixel[0];
            out.g[r][c] = pixel[1];
            out.r[r][c] = pixel[2];
        }
    }
    return out;
}


// ============================================
// ================ CONVOLUTION ===============
// ============================================
int convolveCUDA_RGB_Optimized(
    const vector<vector<double>>& img_r,
    const vector<vector<double>>& img_g,
    const vector<vector<double>>& img_b,
    vector<vector<float>>& out_r,
    vector<vector<float>>& out_g,
    vector<vector<float>>& out_b,
    const vector<float>& kernel,
    int K,
    int tileSize)

    {
        // ===== SETUP =====
        int H = (int)img_r.size();
        int W = (int)img_r[0].size();
        size_t imgBytes = (size_t)H * (size_t)W * sizeof(float);

        // ===== ALLOC PINNED HOST =====
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        float *h_img_pinned, *h_out_pinned;
        CUDA_CHECK(cudaMallocHost(&h_img_pinned, imgBytes));
        CUDA_CHECK(cudaMallocHost(&h_out_pinned, imgBytes));


        // ===== ALLOC CONSTANT & COPY KERNEL =====
        CUDA_CHECK(cudaMemcpyToSymbol(
            d_kernel, kernel.data(), (size_t)K * (size_t)K * sizeof(float)));
        
           
            
        // ===== ALLOC DEVICE GLOBAL =====
        float *d_img = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_img, imgBytes));
        CUDA_CHECK(cudaMalloc(&d_out, imgBytes));


        // ===== SHARED MEMORY CHECK =====
        int sharedDim = tileSize + K - 1;
        int sharedPitch = sharedDim + 1;
        size_t sharedMemSize = 4 * (size_t)sharedPitch * (size_t)sharedDim * sizeof(float);

        if (sharedMemSize > prop.sharedMemPerBlock) {
            cerr << "❌ Shared memory insufficiente\n";
            exit(EXIT_FAILURE);
        }

        // ===== GRID & BLOCK PARAMS =====
        dim3 block(sharedDim, sharedDim);
        dim3 grid(
            (W + tileSize - 1) / tileSize,
            (H + tileSize - 1) / tileSize
        );




        // ===== TIMER START =====
        auto t0 = chrono::high_resolution_clock::now();

        // ===== CONVOLVE CHANNEL R =====

        // ===== COPY FROM HOST TO PINNED =====
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                h_img_pinned[r * W + c] = (float)img_r[r][c];

        // ===== COPY FROM PINNED TO DEVICE GLOBAL =====
        CUDA_CHECK(cudaMemcpy(d_img, h_img_pinned, imgBytes, cudaMemcpyHostToDevice));


        // ===== KERNEL LAUNCH =====
        if (K == 5) {
            conv2d_kernel_optimized_K<5><<<grid, block, sharedMemSize>>>(
                d_img, d_out, H, W, tileSize);
        } else if (K == 7) {
            conv2d_kernel_optimized_K<7><<<grid, block, sharedMemSize>>>(
                d_img, d_out, H, W, tileSize);
        } else {
            conv2d_kernel_optimized_K<3><<<grid, block, sharedMemSize>>>(
                d_img, d_out, H, W, tileSize);
        }

        //===== ERROR CHECK & SYNC =====
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ===== COPY FROM DEVICE TO PINNED =====
        CUDA_CHECK(cudaMemcpy(h_out_pinned, d_out, imgBytes, cudaMemcpyDeviceToHost));


        // ===== COPY FROM PINNED TO HOST =====
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                out_r[r][c] = h_out_pinned[r * W + c];

        // ===== CONVOLVE CHANNEL G =====

        // ===== COPY FROM HOST TO PINNED =====
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                h_img_pinned[r * W + c] = (float)img_g[r][c];

        // ===== COPY FROM PINNED TO DEVICE GLOBAL =====
        CUDA_CHECK(cudaMemcpy(d_img, h_img_pinned, imgBytes, cudaMemcpyHostToDevice));

        // ===== KERNEL LAUNCH =====
        if (K == 5) {
            conv2d_kernel_optimized_K<5><<<grid, block, sharedMemSize>>>(
                d_img, d_out, H, W, tileSize);
        } else if (K == 7) {
            conv2d_kernel_optimized_K<7><<<grid, block, sharedMemSize>>>(
                d_img, d_out, H, W, tileSize);
        } else {
            conv2d_kernel_optimized_K<3><<<grid, block, sharedMemSize>>>(
                d_img, d_out, H, W, tileSize);
        }

        //===== ERROR CHECK & SYNC =====
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ===== COPY FROM DEVICE TO PINNED =====
        CUDA_CHECK(cudaMemcpy(h_out_pinned, d_out, imgBytes, cudaMemcpyDeviceToHost));

        // ===== COPY FROM PINNED TO HOST =====
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                out_g[r][c] = h_out_pinned[r * W + c];

        // ===== CONVOLVE CHANNEL B =====

        // ===== COPY FROM HOST TO PINNED =====
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                h_img_pinned[r * W + c] = (float)img_b[r][c];
        
        // ===== COPY FROM PINNED TO DEVICE GLOBAL =====
        CUDA_CHECK(cudaMemcpy(d_img, h_img_pinned, imgBytes, cudaMemcpyHostToDevice));

        // ===== KERNEL LAUNCH =====
        if (K == 5) {
            conv2d_kernel_optimized_K<5><<<grid, block, sharedMemSize>>>(
                d_img, d_out, H, W, tileSize);
        } else if (K == 7) {
            conv2d_kernel_optimized_K<7><<<grid, block, sharedMemSize>>>(
                d_img, d_out, H, W, tileSize);
        } else {
            conv2d_kernel_optimized_K<3><<<grid, block, sharedMemSize>>>(
                d_img, d_out, H, W, tileSize);
        }

        //===== ERROR CHECK & SYNC =====
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ===== COPY FROM DEVICE TO PINNED =====
        CUDA_CHECK(cudaMemcpy(h_out_pinned, d_out, imgBytes, cudaMemcpyDeviceToHost));

        // ===== COPY FROM PINNED TO HOST =====
        for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
                out_b[r][c] = h_out_pinned[r * W + c];

        // ===== TIMER STOP =====
        auto t1 = chrono::high_resolution_clock::now();
        const auto ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

        // ===== CLEANUP =====
        CUDA_CHECK(cudaFree(d_img));
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFreeHost(h_img_pinned));
        CUDA_CHECK(cudaFreeHost(h_out_pinned));
        return ms;
    }




// ============================================
// ================ MAIN =====================
// ============================================
int main(int argc, char** argv)
{
    string path = (argc > 1) ? argv[1] : "/home/lapemaya/CLionProjects/convCuda/place.png";

    int K = 7;
    if (argc > 2) {
        try {
            K = std::stoi(argv[2]);
        } catch (...) {
            cerr << "Invalid kernel size argument (expected 3, 5, or 7).\n";
            return 1;
        }
    }
    if (!(K == 3 || K == 5 || K == 7)) {
        cerr << "Invalid kernel size " << K << " (expected 3, 5, or 7).\n";
        return 1;
    }

    // Load image
    auto img = loadImageRGB(path);

    // Prepare output containers and input variables
    int H = img.H;
    int W = img.W;

    vector<vector<float>> output_r(H, vector<float>(W, 0.0f));
    vector<vector<float>> output_g(H, vector<float>(W, 0.0f));
    vector<vector<float>> output_b(H, vector<float>(W, 0.0f));

    vector<float> kernel;
    if (K == 3) {
        kernel = {
            1, 2, 1,
            2, 4, 2,
            1, 2, 1
        };
        for (auto& v : kernel) v /= 16.0f;
    } else if (K == 5) {
        kernel = {
            1,  4,  6,  4, 1,
            4, 16, 24, 16, 4,
            6, 24, 36, 24, 6,
            4, 16, 24, 16, 4,
            1,  4,  6,  4, 1
        };
        for (auto& v : kernel) v /= 256.0f;
    } else { 
        kernel = {
            1,   6,  15,  20,  15,   6,  1,
            6,  36,  90, 120,  90,  36,  6,
           15,  90, 225, 300, 225,  90, 15,
           20, 120, 300, 400, 300, 120, 20,
           15,  90, 225, 300, 225,  90, 15,
            6,  36,  90, 120,  90,  36,  6,
            1,   6,  15,  20,  15,   6,  1
        };
        for (auto& v : kernel) v /= 4096.0f;
    }
    
    int tileSize = 16;
    if (argc > 3) {
        try {
            tileSize = std::stoi(argv[3]);
        } catch (...) {
            cerr << "Invalid tile size argument (expected positive integer).\n";
            return 1;
        }
    }
    if (tileSize <= 0 || tileSize > 32) {
        cerr << "Invalid tile size " << tileSize << " (expected 1-32).\n";
        return 1;
    }


    // Execute CUDA optimized RGB convolution
    const auto ms = convolveCUDA_RGB_Optimized(
        img.r, img.g, img.b,
        output_r, output_g, output_b,
        kernel, K, tileSize);

    // Print results
    cout << "CUDA_CONVOLVE_RGB_OPT_MS=" << ms << "\n";
    cout << "✅ Tempo CUDA RGB ottimizzato: " << ms << " ms\n";

    return 0;
}

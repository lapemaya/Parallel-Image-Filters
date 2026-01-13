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

/*
  NOTE GENERALI
  - Questo file fa una convoluzione 2D su immagine RGB usando CUDA.
  - L'immagine viene separata in 3 canali (R,G,B) e la convoluzione viene
    eseguita 3 volte (una per canale) richiamando `convolveCUDA`.
  - Il kernel CUDA usa shared memory per caricare un tile dell'immagine con halo
    (padding) per ridurre accessi in global memory.
  - Il filtro di convoluzione (`kernel`) viene caricato in constant memory
    (`d_kernel`) per sfruttare la cache dedicata (ottimo per piccoli filtri).

  PARAMETRI IMPORTANTI
  - tileSize: dimensione (in pixel) della porzione di output calcolata da un blocco.
  - K: dimensione del filtro (KxK), deve essere <= MAX_KERNEL_SIZE.

  LIMITI / ASSUNZIONI
  - `MAX_KERNEL_SIZE` limita la dimensione massima gestibile in constant memory.
  - Il blocco lanciato è `(tileSize + K - 1) x (tileSize + K - 1)`: attenzione
    a non superare 1024 thread per blocco e i limiti di shared memory.
  - Padding usato: zero-padding fuori dai bordi.
*/

/* ============================================================
   CONFIGURAZIONE
   ============================================================ */

// MAX_KERNEL_SIZE serve per dimensionare la constant memory.
// Se aumenti questo valore devi anche assicurarti che:
//  - il filtro stia in constant memory
//  - eventuali unroll o specializzazioni template vengano aggiornati
//  - i limiti di shared memory per blocco non vengano superati
#define MAX_KERNEL_SIZE 7   // max 7x7


#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
             << " -> " << cudaGetErrorString(err) << endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/* ============================================================
   CONSTANT MEMORY
   ============================================================ */

// d_kernel contiene il filtro KxK in constant memory.
// Vantaggio: tutti i thread leggono gli stessi coefficienti => cache efficace.
__constant__ float d_kernel[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

/* ============================================================
   KERNEL CUDA – CONVOLUZIONE 2D OTTIMIZZATA (TEMPLATE K)
   ============================================================ */

/*
   conv2d_kernel_optimized_K<K>
   - Versione templated (K nota a compile-time).
   - Permette al compilatore di fare unrolling completo dei loop su K,
     evitando overhead del loop e migliorando performance.

   Parametri:
   - img: immagine di input (flattened, H*W)
   - out: output (flattened, H*W)
   - H,W: dimensioni immagine
   - tileSize: dimensione tile output per blocco

   Griglia/blocco:
   - blockDim = (sharedDim, sharedDim) dove sharedDim = tileSize + K - 1
     (quindi include halo).
   - ogni thread carica 1 elemento in shared (tile + halo), poi TANTI thread
     calcolano output ma SOLO quelli in [0..tileSize) scrivono.

   Shared memory:
   - dimensione: sharedDim * sharedDim float
*/

template<int K>
__global__ void conv2d_kernel_optimized_K(
    const float* __restrict__ img,
    float* __restrict__ out,
    int H, int W,
    int tileSize)
{
    // Shared memory dinamica dichiarata al launch: <<<grid, block, sharedBytes>>>
    extern __shared__ float sharedMem[];
    float* s_tile = sharedMem;

    // Coordinate thread nel block.
    // Nota: cast a int per evitare warning clang-tidy su conversione da unsigned.
    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;

    // Per K dispari: radius e' K/2. Es: K=5 => radius=2.
    constexpr int radius = K / 2;

    // sharedDim include halo: tileSize + (K-1)
    // Aggiungiamo +1 di padding per evitare bank conflicts
    int sharedDim = tileSize + K - 1;
    int sharedPitch = sharedDim + 1;  // pitch con padding

    // Coordinate (in pixel) dell'angolo alto-sinistra del tile di OUTPUT.
    int bx = (int)blockIdx.x * tileSize;
    int by = (int)blockIdx.y * tileSize;

    // Coordinate globali del pixel da caricare in shared (tile+halo).
    // -radius per includere il bordo necessario alla convoluzione.
    int gx = bx + tx - radius;
    int gy = by + ty - radius;

    // Caricamento cooperativo in shared memory con __ldg per read-only cache
    // Ogni thread legge 1 elemento se dentro i limiti; altrimenti zero padding.
    if (tx < sharedDim && ty < sharedDim) {
        if (gx >= 0 && gx < W && gy >= 0 && gy < H)
            s_tile[ty * sharedPitch + tx] = __ldg(&img[gy * W + gx]);
        else
            s_tile[ty * sharedPitch + tx] = 0.0f;
    }

    // Sincronizza: tutti i dati devono essere in shared prima di convolvere.
    __syncthreads();

    // Coordinate output calcolate dal thread.
    // Solo i thread nel quadrato tileSize x tileSize producono output.
    int ox = bx + tx;
    int oy = by + ty;

    if (tx < tileSize && ty < tileSize &&
        ox < W && oy < H) {

        float sum = 0.0f;

        // Unrolling completo: #pragma unroll senza numero, con K compile-time.
        #pragma unroll
        for (int ky = 0; ky < K; ky++) {
            #pragma unroll
            for (int kx = 0; kx < K; kx++) {
                // Indice in shared del vicino (pixel + offset filtro) con pitch
                float v = s_tile[(ty + ky) * sharedPitch + (tx + kx)];
                // Coefficiente filtro dalla constant memory
                float w = d_kernel[ky * K + kx];
                sum += v * w;
            }
        }

        out[oy * W + ox] = sum;
    }
}

/*
  conv2d_kernel_optimized (runtime)
  - Versione con K passata a runtime.
  - Utile se vuoi supportare filtri K variabili senza compilare molte specializzazioni.
  - Meno ottimizzata: i loop su K non possono essere completamente unrolled.
*/
__global__ void conv2d_kernel_optimized(
    const float* __restrict__ img,
    float* __restrict__ out,
    int H, int W,
    int K,
    int tileSize)
{
    extern __shared__ float sharedMem[];
    float* s_tile = sharedMem;

    int tx = (int)threadIdx.x;
    int ty = (int)threadIdx.y;

    int radius = K / 2;
    int sharedDim = tileSize + K - 1;
    int sharedPitch = sharedDim + 1;  // padding per bank conflicts

    int bx = (int)blockIdx.x * tileSize;
    int by = (int)blockIdx.y * tileSize;

    int gx = bx + tx - radius;
    int gy = by + ty - radius;

    // Caricamento cooperativo tile + halo con __ldg
    if (tx < sharedDim && ty < sharedDim) {
        if (gx >= 0 && gx < W && gy >= 0 && gy < H)
            s_tile[ty * sharedPitch + tx] = __ldg(&img[gy * W + gx]);
        else
            s_tile[ty * sharedPitch + tx] = 0.0f;
    }

    __syncthreads();

    int ox = bx + tx;
    int oy = by + ty;

    if (tx < tileSize && ty < tileSize &&
        ox < W && oy < H) {

        float sum = 0.0f;

        for (int ky = 0; ky < K; ky++) {
            for (int kx = 0; kx < K; kx++) {
                float v = s_tile[(ty + ky) * sharedPitch + (tx + kx)];
                float w = d_kernel[ky * K + kx];
                sum += v * w;
            }
        }

        out[oy * W + ox] = sum;
    }
}

/* ============================================================
   UTILS
   ============================================================ */

// Struttura dati semplice: 3 canali separati (planar) come vector<vector<double>>.
// H,W vengono riempiti in loadImageRGB.
struct RGBImage {
    vector<vector<double>> r, g, b;
    int H, W;
};

// Carica immagine con OpenCV e la separa in canali.
// Nota: OpenCV legge in formato BGR, quindi facciamo mapping:
//   pixel[0]=B, pixel[1]=G, pixel[2]=R.
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
            out.b[r][c] = pixel[0];  // OpenCV usa BGR
            out.g[r][c] = pixel[1];
            out.r[r][c] = pixel[2];
        }
    }

    return out;
}

// Utility per mostrare un'immagine partendo da 3 canali float.
// I valori vengono clampati in [0,255] e convertiti in uchar.
void showImage(const vector<vector<float>>& r,
               const vector<vector<float>>& g,
               const vector<vector<float>>& b,
               const string& name)
{
    int H = r.size();
    int W = r[0].size();

    Mat m(H, W, CV_8UC3);
    for (int row = 0; row < H; row++) {
        for (int col = 0; col < W; col++) {
            m.at<Vec3b>(row, col)[0] = (uchar)min(255.f, max(0.f, b[row][col]));
            m.at<Vec3b>(row, col)[1] = (uchar)min(255.f, max(0.f, g[row][col]));
            m.at<Vec3b>(row, col)[2] = (uchar)min(255.f, max(0.f, r[row][col]));
        }
    }

    imshow(name, m);
    waitKey(0);
}

/* ============================================================
   HOST – CONVOLUZIONE CUDA RGB OTTIMIZZATA
   ============================================================ */

/*
  convolveCUDA_RGB_Optimized
  - Convoluzione ottimizzata per 3 canali RGB contemporaneamente.

  OTTIMIZZAZIONI IMPLEMENTATE:
  1) Pinned memory per trasferimenti H2D/D2H più veloci (~2x)
  2) Riuso buffer GPU: alloca 1 sola volta per tutti e 3 i canali
  3) Upload kernel in constant memory 1 sola volta (non 3)
  4) Kernel templated K=5 con unrolling completo

  Guadagno stimato: 30-40% rispetto alla versione originale
*/
void convolveCUDA_RGB_Optimized(
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
    int H = (int)img_r.size();
    int W = (int)img_r[0].size();
    size_t imgBytes = (size_t)H * (size_t)W * sizeof(float);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // ===== OTTIMIZZAZIONE 1: Pinned memory per trasferimenti veloci =====
    float *h_img_pinned, *h_out_pinned;
    CUDA_CHECK(cudaMallocHost(&h_img_pinned, imgBytes));
    CUDA_CHECK(cudaMallocHost(&h_out_pinned, imgBytes));

    // ===== Upload kernel in constant memory UNA SOLA VOLTA =====
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_kernel, kernel.data(), (size_t)K * (size_t)K * sizeof(float)));

    // ===== OTTIMIZZAZIONE 2: Alloca buffer GPU UNA SOLA VOLTA =====
    float *d_img = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_out, imgBytes));

    // Configurazione griglia
    int sharedDim = tileSize + K - 1;
    int sharedPitch = sharedDim + 1;  // +1 per padding anti-bank-conflict
    size_t sharedMemSize = (size_t)sharedPitch * (size_t)sharedDim * sizeof(float);

    if (sharedMemSize > prop.sharedMemPerBlock) {
        cerr << "❌ Shared memory insufficiente\n";
        exit(EXIT_FAILURE);
    }

    dim3 block(sharedDim, sharedDim);
    dim3 grid(
        (W + tileSize - 1) / tileSize,
        (H + tileSize - 1) / tileSize
    );

    // ===== PROCESSA I 3 CANALI CON RIUSO BUFFER =====

    // CANALE R
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            h_img_pinned[r * W + c] = (float)img_r[r][c];

    CUDA_CHECK(cudaMemcpy(d_img, h_img_pinned, imgBytes, cudaMemcpyHostToDevice));

    if (K == 5) {
        conv2d_kernel_optimized_K<5><<<grid, block, sharedMemSize>>>(
            d_img, d_out, H, W, tileSize);
    } else if (K == 7) {
        conv2d_kernel_optimized_K<7><<<grid, block, sharedMemSize>>>(
            d_img, d_out, H, W, tileSize);
    } else {
        conv2d_kernel_optimized<<<grid, block, sharedMemSize>>>(
            d_img, d_out, H, W, K, tileSize);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out_pinned, d_out, imgBytes, cudaMemcpyDeviceToHost));

    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            out_r[r][c] = h_out_pinned[r * W + c];

    // CANALE G
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            h_img_pinned[r * W + c] = (float)img_g[r][c];

    CUDA_CHECK(cudaMemcpy(d_img, h_img_pinned, imgBytes, cudaMemcpyHostToDevice));

    if (K == 5) {
        conv2d_kernel_optimized_K<5><<<grid, block, sharedMemSize>>>(
            d_img, d_out, H, W, tileSize);
    } else if (K == 7) {
        conv2d_kernel_optimized_K<7><<<grid, block, sharedMemSize>>>(
            d_img, d_out, H, W, tileSize);
    } else {
        conv2d_kernel_optimized<<<grid, block, sharedMemSize>>>(
            d_img, d_out, H, W, K, tileSize);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out_pinned, d_out, imgBytes, cudaMemcpyDeviceToHost));

    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            out_g[r][c] = h_out_pinned[r * W + c];

    // CANALE B
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            h_img_pinned[r * W + c] = (float)img_b[r][c];

    CUDA_CHECK(cudaMemcpy(d_img, h_img_pinned, imgBytes, cudaMemcpyHostToDevice));

    if (K == 5) {
        conv2d_kernel_optimized_K<5><<<grid, block, sharedMemSize>>>(
            d_img, d_out, H, W, tileSize);
    } else if (K == 7) {
        conv2d_kernel_optimized_K<7><<<grid, block, sharedMemSize>>>(
            d_img, d_out, H, W, tileSize);
    } else {
        conv2d_kernel_optimized<<<grid, block, sharedMemSize>>>(
            d_img, d_out, H, W, K, tileSize);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out_pinned, d_out, imgBytes, cudaMemcpyDeviceToHost));

    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            out_b[r][c] = h_out_pinned[r * W + c];

    // ===== CLEANUP =====
    CUDA_CHECK(cudaFree(d_img));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_img_pinned));
    CUDA_CHECK(cudaFreeHost(h_out_pinned));
}

/* ============================================================
   MAIN
   ============================================================ */

int main(int argc, char** argv)
{
    // path: se passato da riga di comando usa quello, altrimenti default.
    string path = (argc > 1) ? argv[1] : "/home/lapemaya/CLionProjects/convCuda/place.png";

    // Kernel size: optional argv[2] in {3,5,7}. Default 7.
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

    // Carica immagine e prepara canali.
    auto img = loadImageRGB(path);

    int H = img.H;
    int W = img.W;

    // Output come float (uno per canale).
    vector<vector<float>> output_r(H, vector<float>(W, 0.0f));
    vector<vector<float>> output_g(H, vector<float>(W, 0.0f));
    vector<vector<float>> output_b(H, vector<float>(W, 0.0f));

    // Gaussian kernel (binomial) normalized.
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
    } else { // K == 7
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

    // tileSize influenza performance e shared memory.
    // Tipicamente 16 e' un buon compromesso.
    int tileSize = 16;

    auto t0 = chrono::high_resolution_clock::now();

    // Convoluzione RGB ottimizzata (pinned memory + riuso buffer GPU)
    convolveCUDA_RGB_Optimized(
        img.r, img.g, img.b,
        output_r, output_g, output_b,
        kernel, K, tileSize);

    auto t1 = chrono::high_resolution_clock::now();

    const auto ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

    // Stable parse-friendly timing line for benchmarking.
    // Measures ONLY the region around convolveCUDA_RGB_Optimized() call.
    cout << "CUDA_CONVOLVE_RGB_OPT_MS=" << ms << "\n";

    // Human-friendly line (keep it for manual runs)
    cout << "✅ Tempo CUDA RGB ottimizzato: " << ms << " ms\n";

    // Decommenta se vuoi vedere l'immagine risultante.
    //showImage(output_r, output_g, output_b, "Convoluzione RGB CUDA");

    return 0;
}

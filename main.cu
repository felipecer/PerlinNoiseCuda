#include <iostream>

#include "cuda_runtime.h"
#include <stdio.h>
#include "ppm.h"
#include <random>
#include <algorithm>
#include <numeric>
#include "Managed.cuh"
#include <chrono>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::milli;

class PerlinNoise : public Managed {
public:
    __host__ PerlinNoise();
    __host__ PerlinNoise(unsigned int seed);
    __device__ double noise(int* dev_p, double x, double y, double z);
    std::vector<int> p;
private:
    __device__ double fade(double t);
    __device__ double lerp(double t, double a, double b);
    __device__ double grad(int hash, double x, double y, double z);
};

PerlinNoise::PerlinNoise() {
    p = {
            151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
            8,99,37,240,21,10,23,190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
            35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,
            134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
            55,46,245,40,244,102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208, 89,
            18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,
            250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
            189,28,42,223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167,
            43,172,9,129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,
            97,228,251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,
            107,49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
            138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180 };
    p.insert(p.end(), p.begin(), p.end());
}

// Generate a new permutation vector based on the value of seed
PerlinNoise::PerlinNoise(unsigned int seed) {
    p.resize(256);
    std::iota(p.begin(), p.end(), 0);
    std::default_random_engine engine(seed);
    std::shuffle(p.begin(), p.end(), engine);
    p.insert(p.end(), p.begin(), p.end());
}

__device__ double PerlinNoise::noise(int* dp, double x, double y, double z) {
    int X = (int)floor(x) & 255;
    int Y = (int)floor(y) & 255;
    int Z = (int)floor(z) & 255;

    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    double u = fade(x);
    double v = fade(y);
    double w = fade(z);

    int A = dp[X] + Y;
    int AA = dp[A] + Z;
    int AB = dp[A + 1] + Z;
    int B = dp[X + 1] + Y;
    int BA = dp[B] + Z;
    int BB = dp[B + 1] + Z;

    double res = lerp(w, lerp(v, lerp(u, grad(dp[AA], x, y, z), grad(dp[BA], x - 1, y, z)), lerp(u, grad(dp[AB], x, y - 1, z), grad(dp[BB], x - 1, y - 1, z))), lerp(v, lerp(u, grad(dp[AA + 1], x, y, z - 1), grad(dp[BA + 1], x - 1, y, z - 1)), lerp(u, grad(dp[AB + 1], x, y - 1, z - 1), grad(dp[BB + 1], x - 1, y - 1, z - 1))));
    return (res + 1.0) / 2.0;
}

__device__ double PerlinNoise::fade(double t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ double PerlinNoise::lerp(double t, double a, double b) {
    return a + t * (b - a);
}

__device__ double PerlinNoise::grad(int hash, double x, double y, double z) {
    int h = hash & 15;
    double u = h < 8 ? x : y,
            v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

__global__ void PerlinResult(unsigned int height, unsigned int width, int* dev_p, PerlinNoise* pn, unsigned char* r, unsigned char* g, unsigned char* b)
{
    unsigned int j = blockIdx.x* blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < height && j < width)
    {
        int kk = (i * width) + j;
        double x = (double)j / ((double)width);
        double y = (double)i / ((double)height);
        double n = pn->noise(dev_p, 10 * x, 10 * y, 0.8);
        r[kk] = floor(255 * n);
        g[kk] = floor(255 * n);
        b[kk] = floor(255 * n);
    }
}
int main() {
    auto startTime = high_resolution_clock::now();
    unsigned long long width = 2000, height = 2000;
    unsigned long long pixelCount = width * height;
    unsigned char *r, *g, *b;
    cudaMalloc(&r, pixelCount * sizeof(unsigned char));
    cudaMalloc(&g, pixelCount * sizeof(unsigned char));
    cudaMalloc(&b, pixelCount * sizeof(unsigned char));

    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    unsigned int seed = 237;
    PerlinNoise* pn = new PerlinNoise(seed);
    size_t grad_size = pn->p.size();
    int* grad;
    cudaMalloc(&grad, grad_size * sizeof(int));
    cudaMemcpy(grad, pn->p.data(), grad_size*sizeof(int), cudaMemcpyHostToDevice);
    ppm image(width, height);

    PerlinResult <<< gridSize, blockSize >>> (height, width, grad, pn, r, g, b);
    cudaDeviceSynchronize();
    unsigned char* rr = (unsigned char*)malloc(pixelCount * sizeof(unsigned char));
    unsigned char* rg = (unsigned char*)malloc(pixelCount * sizeof(unsigned char));
    unsigned char* rb = (unsigned char*)malloc(pixelCount * sizeof(unsigned char));

    cudaMemcpy(rr, r, pixelCount*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(rg, g, pixelCount*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(rb, b, pixelCount*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    auto endTime = high_resolution_clock::now();
    for (size_t i = 0; i < pixelCount; i++)
    {
        image.r[i] = rr[i];
        image.g[i] = rg[i];
        image.b[i] = rb[i];
    }
    printf("%s: Time: %fms\n", "CUDA Perlin Noise", duration_cast<duration<double, milli>>(endTime - startTime).count());

    image.write("figure_cuda_3.ppm");

    return 0;
}
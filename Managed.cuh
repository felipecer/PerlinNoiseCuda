//
// Created by felipecerda on 8/4/22.
//
#include "cuda_runtime.h"
#ifndef PERLINNOISE_MANAGED_CUH
#define PERLINNOISE_MANAGED_CUH



class Managed {
public:
    void* operator new(size_t len) {
        void* ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    void operator delete(void* ptr) {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

#endif //PERLINNOISE_MANAGED_CUH
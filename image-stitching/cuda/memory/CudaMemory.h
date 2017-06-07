//
// Created by tjb3 on 12/2/15.
//

#ifndef HTGS_CUDAMEMORY_H
#define HTGS_CUDAMEMORY_H

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>

class CudaMemory: public htgs::IMemoryAllocator<cuda_t> {

 public:
  CudaMemory(size_t size) : IMemoryAllocator(size) { }

  ~CudaMemory() { }

  cuda_t *memAlloc(size_t size) {
    cuda_t *ptr;
    cudaMalloc((void **) &ptr, sizeof(cuda_t) * size);
    return ptr;
  }

  cuda_t *memAlloc() {
    cuda_t *ptr;
    cudaMalloc((void **) &ptr, sizeof(cuda_t) * size());
    return ptr;
  }

  void memFree(cuda_t *&memory) {
    cudaFree(memory);
  }
};

#endif //HTGS_CUDAMEMORY_H

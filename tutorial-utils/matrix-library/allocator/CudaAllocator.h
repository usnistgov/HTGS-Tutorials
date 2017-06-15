//
// Created by tjb3 on 6/13/17.
//

#ifndef HTGS_TUTORIALS_CUDAALLOCATOR_H
#define HTGS_TUTORIALS_CUDAALLOCATOR_H
#include <cuda_runtime.h>
#include <htgs/api/IMemoryAllocator.hpp>
class CudaAllocator : public htgs::IMemoryAllocator<double> {
 public:
  CudaAllocator(size_t width, size_t height) : htgs::IMemoryAllocator<double>(width * height) {}

  double *memAlloc(size_t size) {
    double *mem = nullptr;
    cudaMalloc((void **)&mem, sizeof(double)*size);
    return mem;
  }

  double *memAlloc() {
    double *mem = nullptr;
    cudaMalloc((void **)&mem, sizeof(double)*this->size());
    return mem;
  }

  void memFree(double *&memory) {
    cudaFree(memory);
  }

};
#endif //HTGS_TUTORIALS_CUDAALLOCATOR_H

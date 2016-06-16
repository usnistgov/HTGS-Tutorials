//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_CUDAMATRIXALLOCATOR_H
#define HTGS_CUDAMATRIXALLOCATOR_H
#include <htgs/api/IMemoryAllocator.hpp>

class CudaMatrixAllocator : public htgs::IMemoryAllocator<double *> {
 public:
  CudaMatrixAllocator(int width, int height) : IMemoryAllocator((size_t) width * height) { }

  double *memAlloc(size_t size) {
    double *mem;
    cudaMallocHost((void **)&mem, sizeof(double) * size);
    return mem;
  }

  double *memAlloc() {
    double *mem;
    cudaMallocHost((void **)&mem, sizeof(double) * this->size());
    return mem;
  }

  void memFree(double *&memory) {
    cudaFreeHost(memory);
  }

};
#endif //HTGS_CUDAMATRIXALLOCATOR_H

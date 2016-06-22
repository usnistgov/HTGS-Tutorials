//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXALLOCATOR_H
#define HTGS_MATRIXALLOCATOR_H
#include <htgs/api/IMemoryAllocator.hpp>

class MatrixAllocator : public htgs::IMemoryAllocator<double *> {
 public:
  MatrixAllocator(int width, int height) : IMemoryAllocator((size_t) width * height) {}

  double *memAlloc(size_t size) {
    double *mem = new double[size];
    return mem;
  }

  double *memAlloc() {
    double *mem = new double[this->size()];
    return mem;
  }

  void memFree(double *&memory) {
    delete[] memory;
  }

};
#endif //HTGS_MATRIXALLOCATOR_H

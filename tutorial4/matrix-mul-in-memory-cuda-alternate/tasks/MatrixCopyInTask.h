//
// Created by tjb3 on 6/15/16.
//

#ifndef HTGS_TUTORIALS_MATRIXCOPYINTASK_H
#define HTGS_TUTORIALS_MATRIXCOPYINTASK_H

#include <htgs/api/ICudaTask.hpp>
#include "../data/MatrixBlockData.h"
#include <cuda.h>

class MatrixCopyInTask : public htgs::ICudaTask<MatrixBlockData<double *>, MatrixBlockData<MatrixMemoryData_t>> {
 public:
  MatrixCopyInTask(std::string name, int blockSize, int releaseCount,
                   CUcontext *contexts, int *cudaIds, int numGpus, long leadingDimensionFullMatrix);

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockData<double *>> data, CUstream stream);

  virtual std::string getName() {
    return "CudaCopyInTask(" + name + ")";
  }

  virtual MatrixCopyInTask *copy() {
    return new MatrixCopyInTask(this->name,
                                this->blockSize,
                                this->releaseCount,
                                this->getContexts(),
                                this->getCudaIds(),
                                this->getNumGPUs(),
                                this->leadingDimensionFullMatrix);
  }

 private:
  std::string name;
  int releaseCount;
  double *gpuMemPinned;
  double *scratchSpace;
  int blockSize;
  long leadingDimensionFullMatrix;
};

#endif //HTGS_TUTORIALS_MATRIXCOPYINTASK_H

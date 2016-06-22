//
// Created by tjb3 on 6/15/16.
//

#ifndef HTGS_TUTORIALS_MATRIXCOPYOUTTASK_H
#define HTGS_TUTORIALS_MATRIXCOPYOUTTASK_H

#include <htgs/api/ICudaTask.hpp>
#include "../data/MatrixBlockData.h"
#include <cuda.h>

class MatrixCopyOutTask : public htgs::ICudaTask<MatrixBlockData<MatrixMemoryData_t>, MatrixBlockData<double *>> {
 public:
  MatrixCopyOutTask(std::string name, int blockSize, CUcontext *contexts, int *cudaIds, int numGpus);

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> data, CUstream stream);

  virtual void shutdownCuda() {
//    cudaFreeHost(cudaMemPinned);
  }

  virtual std::string getName() {
    return "CudaCopyOutTask(" + name + ")";
  }

  virtual htgs::ITask<MatrixBlockData<MatrixMemoryData_t>, MatrixBlockData<double *>> *copy() {
    return new MatrixCopyOutTask(this->name, blockSize, this->getContexts(), this->getCudaIds(), this->getNumGPUs());
  }

 private:
  std::string name;
  int blockSize;
  double *cudaMemPinned;
};

#endif //HTGS_TUTORIALS_MATRIXCOPYOUTTASK_H

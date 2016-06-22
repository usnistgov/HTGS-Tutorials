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
                   CUcontext *contexts, int *cudaIds, int numGpus, long leadingDimensionFullMatrix) :
      ICudaTask(contexts, cudaIds, numGpus),
      name(name),
      releaseCount(releaseCount),
      blockSize(blockSize),
      leadingDimensionFullMatrix(leadingDimensionFullMatrix) {}

  virtual void
  initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId, int numPipelines) {
//    cudaMallocHost((void **)&gpuMemPinned, sizeof(double)*blockSize*blockSize);
//    scratchSpace = new double[blockSize*blockSize];
  }

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockData<double *>> data, CUstream stream) {
    std::string matrixName = matrixTypeToString(data->getRequest()->getType());

    // CPU Memory
    double *memoryIn = data->getMatrixData();

    // Cuda Memory
    auto memoryOut = this->memGet<double *>(matrixName + "Copy", new MatrixMemoryRule(releaseCount));

    cublasSetMatrixAsync((int) data->getMatrixHeight(), (int) data->getMatrixWidth(), sizeof(double),
                         memoryIn, (int) leadingDimensionFullMatrix,
                         memoryOut->get(), (int) data->getMatrixHeight(), stream);

    this->syncStream();

    this->addResult(new MatrixBlockData<MatrixMemoryData_t>(data->getRequest(),
                                                            memoryOut,
                                                            data->getMatrixWidth(),
                                                            data->getMatrixHeight()));
  }

  virtual void shutdownCuda() {
//    cudaFreeHost(gpuMemPinned);
//    delete [] scratchSpace;
  }

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

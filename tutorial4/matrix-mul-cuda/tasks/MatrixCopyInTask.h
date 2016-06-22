//
// Created by tjb3 on 6/15/16.
//

#ifndef HTGS_TUTORIALS_MATRIXCOPYINTASK_H
#define HTGS_TUTORIALS_MATRIXCOPYINTASK_H

#include <htgs/api/ICudaTask.hpp>
#include "../data/MatrixBlockData.h"
#include <cuda.h>

class MatrixCopyInTask : public htgs::ICudaTask<MatrixBlockData<MatrixMemoryData_t>,
                                                MatrixBlockData<MatrixMemoryData_t>> {
 public:
  MatrixCopyInTask(std::string name, int blockSize, int releaseCount, CUcontext *contexts, int *cudaIds, int numGpus)
      : ICudaTask(contexts, cudaIds, numGpus),
        name(name), releaseCount(releaseCount), blockSize(blockSize) {}

  virtual void
  initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId, int numPipelines) {
    cudaMallocHost((void **) &gpuMemPinned, sizeof(double) * blockSize * blockSize);
  }

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> data, CUstream stream) {
    std::string matrixName = matrixTypeToString(data->getRequest()->getType());

    // CPU Memory
    auto memoryIn = data->getMatrixData();

    // Cuda Memory
    auto memoryOut = this->memGet<double *>(matrixName + "Copy", new MatrixMemoryRule(releaseCount));

    gpuErrorChk(cudaMemcpyAsync(gpuMemPinned,
                                memoryIn->get(),
                                sizeof(double) * data->getMatrixHeight() * data->getMatrixWidth(),
                                cudaMemcpyHostToDevice,
                                stream));
    gpuErrorChk(cudaMemcpyAsync(memoryOut->get(),
                                gpuMemPinned,
                                sizeof(double) * data->getMatrixHeight() * data->getMatrixWidth(),
                                cudaMemcpyDeviceToDevice,
                                stream));

    this->syncStream();

    this->memRelease(matrixName, memoryIn);

    this->addResult(new MatrixBlockData<MatrixMemoryData_t>(data->getRequest(),
                                                            memoryOut,
                                                            data->getMatrixWidth(),
                                                            data->getMatrixHeight()));
  }

  virtual void shutdownCuda() {
    cudaFreeHost(gpuMemPinned);
  }

  virtual std::string getName() {
    return "CudaCopyInTask(" + name + ")";
  }

  virtual htgs::ITask<MatrixBlockData<MatrixMemoryData_t>, MatrixBlockData<MatrixMemoryData_t>> *copy() {
    return new MatrixCopyInTask(this->name,
                                this->blockSize,
                                this->releaseCount,
                                this->getContexts(),
                                this->getCudaIds(),
                                this->getNumGPUs());
  }

 private:
  std::string name;
  int releaseCount;
  double *gpuMemPinned;
  int blockSize;
};

#endif //HTGS_TUTORIALS_MATRIXCOPYINTASK_H

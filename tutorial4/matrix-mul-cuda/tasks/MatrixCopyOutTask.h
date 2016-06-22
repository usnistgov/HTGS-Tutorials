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
  MatrixCopyOutTask(std::string name, int blockSize, CUcontext *contexts, int *cudaIds, int numGpus) : ICudaTask(
      contexts,
      cudaIds,
      numGpus),
                                                                                                       name(name),
                                                                                                       blockSize(
                                                                                                           blockSize) {}

  virtual void
  initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId, int numPipelines) {
    cudaMallocHost((void **) &cudaMemPinned, sizeof(double) * blockSize * blockSize);
  }

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> data, CUstream stream) {
    // Cuda Memory
    auto memoryIn = data->getMatrixData();

    // CPU Memory
    double *memoryOut = new double[data->getMatrixHeight() * data->getMatrixWidth()];

    gpuErrorChk(cudaMemcpyAsync(cudaMemPinned,
                                memoryIn->get(),
                                sizeof(double) * data->getMatrixHeight() * data->getMatrixWidth(),
                                cudaMemcpyDeviceToDevice,
                                stream));
    gpuErrorChk(cudaMemcpyAsync(memoryOut,
                                cudaMemPinned,
                                sizeof(double) * data->getMatrixHeight() * data->getMatrixWidth(),
                                cudaMemcpyDeviceToHost,
                                stream));

    this->syncStream();

    this->memRelease(name, memoryIn);

    this->addResult(new MatrixBlockData<double *>(data->getRequest(),
                                                  memoryOut,
                                                  data->getMatrixWidth(),
                                                  data->getMatrixHeight()));
  }

  virtual void shutdownCuda() {
    cudaFreeHost(cudaMemPinned);
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

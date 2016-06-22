//
// Created by tjb3 on 6/22/16.
//

#include <cublas_v2.h>
#include "MatrixCopyOutTask.h"

MatrixCopyOutTask::MatrixCopyOutTask(std::string name, int blockSize, CUcontext *contexts, int *cudaIds, int numGpus)
  : ICudaTask(contexts, cudaIds, numGpus), name(name), blockSize(blockSize)
{}

void MatrixCopyOutTask::executeGPUTask(std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> data, CUstream stream) {
  // Cuda Memory
  auto memoryIn = data->getMatrixData();

  // CPU Memory
  double *memoryOut = new double[data->getMatrixHeight()*data->getMatrixWidth()];

  cublasGetMatrixAsync((int)data->getMatrixHeight(), (int)data->getMatrixWidth(), sizeof(double),
                       memoryIn->get(), (int)data->getMatrixHeight(),
                       memoryOut, (int)data->getMatrixHeight(), stream);

  this->syncStream();

  this->memRelease(name, memoryIn);

  this->addResult(new MatrixBlockData<double *>(data->getRequest(), memoryOut, data->getMatrixWidth(), data->getMatrixHeight()));
}
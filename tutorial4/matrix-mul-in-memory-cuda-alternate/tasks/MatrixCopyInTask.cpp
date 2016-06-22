//
// Created by tjb3 on 6/22/16.
//

#include <cublas_v2.h>
#include "MatrixCopyInTask.h"
#include "../memory/MatrixMemoryRule.h"

MatrixCopyInTask::MatrixCopyInTask(std::string name,
                                   int blockSize,
                                   int releaseCount,
                                   CUcontext *contexts,
                                   int *cudaIds,
                                   int numGpus,
                                   long leadingDimensionFullMatrix) :
    ICudaTask(contexts, cudaIds, numGpus),
    name(name),
    releaseCount(releaseCount),
    blockSize(blockSize),
    leadingDimensionFullMatrix(leadingDimensionFullMatrix) {}

void MatrixCopyInTask::executeGPUTask(std::shared_ptr<MatrixBlockData<double *>> data, CUstream stream) {
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
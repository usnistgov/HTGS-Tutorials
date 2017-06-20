//
// Created by tjb3 on 6/13/17.
//

#ifndef HTGS_TUTORIALS_CUDACOPYINTASK_H
#define HTGS_TUTORIALS_CUDACOPYINTASK_H
#include <htgs/api/ICudaTask.hpp>
#include <htgs/types/Types.hpp>
#include <cublas_v2.h>

#include "../../../tutorial-utils/matrix-library/data/MatrixBlockData.h"
#include "../../../tutorial-utils/matrix-library/rules/MatrixMemoryRule.h"


class CudaCopyInTask : public htgs::ICudaTask<MatrixBlockData<double *>, MatrixBlockData<htgs::m_data_t<double>>>
{
 public:
  CudaCopyInTask(CUcontext *contexts,
                 int *cudaIds,
                 size_t numGpus,
                 MatrixType matrixType,
                 size_t releaseCount) : ICudaTask(contexts, cudaIds, numGpus),
                                        matrixType(matrixType),
                                        releaseCount(releaseCount) {
    matrixName = matrixTypeToString(matrixType);
    numBytes = 0;
  }

  void initializeCudaGPU() override {
  }

  void shutdownCuda() override {
  }

  std::string getName() override {
    return "CudaCopyInTask(" + matrixTypeToString(matrixType) + ")";
  }

  void executeTask(std::shared_ptr<MatrixBlockData<double *>> data) override {
    // CPU Memory
    double *memoryIn = data->getMatrixData();

    // CUDA Memory
    htgs::m_data_t<double> memoryOut= this->getMemory<double>(matrixName + "Copy", new MatrixMemoryRule(releaseCount));

    // Copy from CPU to GPU
    cublasSetMatrixAsync((int) data->getMatrixHeight(), (int) data->getMatrixWidth(), sizeof(double),
                          memoryIn, (int) data->getLeadingDimension(),
                          memoryOut->get(), (int)data->getMatrixHeight(), this->getStream());

    numBytes += data->getMatrixHeight() * data->getMatrixWidth() * sizeof(double);
    this->syncStream();

    this->addResult(new MatrixBlockData<htgs::m_data_t<double>>(data->getRequest(),
                                                                memoryOut,
                                                                data->getMatrixWidth(),
                                                                data->getMatrixHeight(),
                                                                data->getMatrixHeight()));
  }
  htgs::ITask<MatrixBlockData<double *>, MatrixBlockData<htgs::m_data_t<double>>> *copy() override {
    return new CudaCopyInTask(this->getContexts(), this->getCudaIds(), this->getNumGPUs(), this->matrixType, this->releaseCount);
  }

  std::string getDotCustomProfile() override {
    auto microTime = this->getTaskComputeTime();
    double numGFlop = ((double)numBytes) * 1.0e-9d;

    double timeSec = (double)microTime / 1000000.0;

    return "Performance: " + std::to_string(numGFlop / timeSec) + " GB/s";
  }

 private:
  MatrixType matrixType;
  size_t releaseCount;
  std::string matrixName;
  size_t numBytes;

};
#endif //HTGS_TUTORIALS_CUDACOPYINTASK_H

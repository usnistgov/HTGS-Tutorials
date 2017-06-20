//
// Created by tjb3 on 6/13/17.
//

#ifndef HTGS_TUTORIALS_CUDACOPYOUTTASK_H
#define HTGS_TUTORIALS_CUDACOPYOUTTASK_H
#include <htgs/api/ICudaTask.hpp>
#include <htgs/types/Types.hpp>
#include <cublas_v2.h>

#include "../../../tutorial-utils/matrix-library/data/MatrixBlockData.h"
#include "../../../tutorial-utils/matrix-library/rules/MatrixMemoryRule.h"


class CudaCopyOutTask : public htgs::ICudaTask<MatrixBlockData<htgs::m_data_t<double>>, MatrixBlockData<double *>>
{
 public:
  CudaCopyOutTask(CUcontext *contexts,
                 int *cudaIds,
                 size_t numGpus,
                 MatrixType matrixType) : ICudaTask(contexts, cudaIds, numGpus),
                                        matrixType(matrixType) {
    matrixName = matrixTypeToString(matrixType);
    numBytes = 0;
  }

  void initializeCudaGPU() override {
  }

  void shutdownCuda() override {
  }

  std::string getName() override {
    return "CudaCopyOutTask(" + matrixTypeToString(matrixType) + ")";
  }

  void executeTask(std::shared_ptr<MatrixBlockData<htgs::m_data_t<double>>> data) override {
    // CUDA Memory
    htgs::m_data_t<double> memoryIn = data->getMatrixData();

    // CPU Memory
    double *memoryOut = new double[data->getMatrixHeight() * data->getMatrixWidth()];

    // Copy from CPU to GPU
    cublasGetMatrixAsync((int) data->getMatrixHeight(), (int) data->getMatrixWidth(), sizeof(double),
                          memoryIn->get(), (int) data->getLeadingDimension(),
                          memoryOut, (int)data->getLeadingDimension(), this->getStream());
    numBytes += data->getMatrixHeight() * data->getMatrixWidth() * sizeof(double);

    this->syncStream();

    this->releaseMemory(memoryIn);

    this->addResult(new MatrixBlockData<double *>(data->getRequest(),
                                                                memoryOut,
                                                                data->getMatrixWidth(),
                                                                data->getMatrixHeight(),
                                                                data->getLeadingDimension()));
  }
  htgs::ITask<MatrixBlockData<htgs::m_data_t<double>>, MatrixBlockData<double *>> *copy() override {
    return new CudaCopyOutTask(this->getContexts(), this->getCudaIds(), this->getNumGPUs(), this->matrixType);
  }

  std::string getDotCustomProfile() override {
    auto microTime = this->getTaskComputeTime();
    double numGFlop = ((double)numBytes) * 1.0e-9d;

    double timeSec = (double)microTime / 1000000.0;

    return "Performance: " + std::to_string(numGFlop / timeSec) + " GB/s";
  }


 private:
  MatrixType matrixType;
  std::string matrixName;
  size_t numBytes;

};
#endif //HTGS_TUTORIALS_CUDACOPYOUTTASK_H

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

 private:
  MatrixType matrixType;
  std::string matrixName;
};
#endif //HTGS_TUTORIALS_CUDACOPYOUTTASK_H

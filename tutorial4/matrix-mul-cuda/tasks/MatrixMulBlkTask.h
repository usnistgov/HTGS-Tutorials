//
// Created by tjb3 on 2/23/16.
//


#ifndef HTGS_MATRIXMULBLKTASK_H
#define HTGS_MATRIXMULBLKTASK_H

#include "../data/MatrixBlockMulData.h"
#include "../data/MatrixBlockData.h"
#include <htgs/api/ICudaTask.hpp>

class MatrixMulBlkTask : public htgs::ICudaTask<MatrixBlockMulData<MatrixMemoryData_t>,
                                                MatrixBlockData<MatrixMemoryData_t>> {

 public:
  MatrixMulBlkTask(CUcontext *contexts, int *cudaIds, int numGpus) : ICudaTask(contexts, cudaIds, numGpus) {}

  virtual ~MatrixMulBlkTask() {

  }

  virtual void
  initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId, int numPipelines) {
    cublasCreate_v2(&handle);
    cublasSetStream_v2(handle, stream);
    alpha = new double[1];
    alpha[0] = 1.0;
    beta = new double[1];
    beta[0] = 0.0;
  }

  virtual void shutdownCuda() {
    cublasDestroy_v2(handle);
    delete[] alpha;
    delete[] beta;
  }

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockMulData<MatrixMemoryData_t>> data, CUstream stream) {
    auto matAData = data->getMatrixA();
    auto matBData = data->getMatrixB();

    MatrixMemoryData_t matrixA = matAData->getMatrixData();
    MatrixMemoryData_t matrixB = matBData->getMatrixData();

    int width = matBData->getMatrixWidth();
    int height = matAData->getMatrixHeight();

    auto result = this->memGet<double *>("MatrixC", new MatrixMemoryRule(1));

    cublasDgemm_v2(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   height,
                   width,
                   matAData->getMatrixWidth(),
                   alpha,
                   matrixA->get(),
                   matAData->getMatrixHeight(),
                   matrixB->get(),
                   matBData->getMatrixHeight(),
                   beta,
                   result->get(),
                   height);

    this->syncStream();

    std::shared_ptr<MatrixRequestData> matReq(new MatrixRequestData(matAData->getRequest()->getRow(),
                                                                    matBData->getRequest()->getCol(),
                                                                    MatrixType::MatrixC));

    addResult(new MatrixBlockData<MatrixMemoryData_t>(matReq, result, width, height));

    std::string matrixNameA = matrixTypeToString(matAData->getRequest()->getType());
    std::string matrixNameB = matrixTypeToString(matBData->getRequest()->getType());

    this->memRelease(matrixNameA + "Copy", matrixA);
    this->memRelease(matrixNameB + "Copy", matrixB);
  }

  virtual std::string getName() {
    return "MatrixMulBlkTask";
  }
  virtual MatrixMulBlkTask *copy() {
    return new MatrixMulBlkTask(this->getContexts(), this->getCudaIds(), this->getNumGPUs());
  }
  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
    return inputConnector->isInputTerminated();
  }

 private:
  cublasHandle_t handle;
  double *alpha;
  double *beta;
};

#endif //HTGS_MATRIXMULBLKTASK_H

//
// Created by tjb3 on 2/23/16.
//


#ifndef HTGS_MATRIXMULBLKTASK_H
#define HTGS_MATRIXMULBLKTASK_H
#include <cuda.h>

#include "../data/MatrixBlockMulData.h"
#include "../data/MatrixBlockData.h"

#include <htgs/api/ITask.hpp>
#include <htgs/api/ICudaTask.hpp>
#include <cublas_v2.h>

class MatrixMulBlkTask : public htgs::ICudaTask<MatrixBlockMulData<MatrixMemoryData_t>, MatrixBlockData<MatrixMemoryData_t>>
{

 public:
  MatrixMulBlkTask(CUcontext *contexts, int *cudaIds, int numGpus, long fullMatrixWidthA, long fullMatrixHeightA, long fullMatrixWidthB, long fullMatrixHeightB, int blockSize) :
      ICudaTask(contexts, cudaIds, numGpus),
      fullMatrixWidthA(fullMatrixWidthA), fullMatrixHeightA(fullMatrixHeightA),
      fullMatrixWidthB(fullMatrixWidthB), fullMatrixHeightB(fullMatrixHeightB), blockSize(blockSize)
  {}

  virtual ~MatrixMulBlkTask() {

  }
  virtual void initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId, int numPipelines) {
    cublasCreate_v2(&handle);
    cublasSetStream_v2(handle, stream);
//    alpha = new double[1];
//    alpha[0] = 1.0;
//    beta = new double[1];
//    beta[0] = 0.0;
  }
  virtual void shutdownCuda() {
    cublasDestroy_v2(handle);
//    delete [] alpha;
//    delete [] beta;
  }

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockMulData<MatrixMemoryData_t>> data, CUstream stream) {

    auto matAData = data->getMatrixA();
    auto matBData = data->getMatrixB();

    auto matrixA = matAData->getMatrixData();
    auto matrixB = matBData->getMatrixData();

    long width = matBData->getMatrixWidth();
    long height = matAData->getMatrixHeight();

    int blkRowA = data->getMatrixA()->getRequest()->getRow();
    int blkColA = data->getMatrixA()->getRequest()->getCol();
    int blkRowB = data->getMatrixB()->getRequest()->getRow();
    int blkColB = data->getMatrixB()->getRequest()->getCol();

    auto result = this->memGet<double *>("MatrixC", new MatrixMemoryRule(1));

    double *alpha = new double[1] { 1.0};
    double *beta = new double[1] {0.0};

    cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)height, (int)width, (int)matAData->getMatrixWidth(), alpha, matrixA->get(), (int)matAData->getMatrixHeight(),
                   matrixB->get(), (int)matBData->getMatrixHeight(), beta, result->get(), (int)height);
//    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, height, width, matAData->getMatrixWidth(), 1.0, matrixA, fullMatrixWidthA,
//                matrixB, fullMatrixWidthB, 0.0, result, width);

    this->syncStream();


    std::shared_ptr<MatrixRequestData> matReq(new MatrixRequestData(matAData->getRequest()->getRow(), matBData->getRequest()->getCol(), MatrixType::MatrixC));

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
    return new MatrixMulBlkTask(this->getContexts(), this->getCudaIds(), this->getNumGPUs(), fullMatrixWidthA, fullMatrixHeightA, fullMatrixWidthB, fullMatrixHeightB, blockSize);
  }
  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
    return inputConnector->isInputTerminated();
  }

 private:
  long fullMatrixWidthA;
  long fullMatrixHeightA;
  long fullMatrixWidthB;
  long fullMatrixHeightB;
  int blockSize;

  cublasHandle_t handle;
//  double *alpha;
//  double *beta;
};

#endif //HTGS_MATRIXMULBLKTASK_H

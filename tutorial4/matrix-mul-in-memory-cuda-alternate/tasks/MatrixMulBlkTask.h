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
  MatrixMulBlkTask(CUcontext *contexts, int *cudaIds, int numGpus, long fullMatrixWidthA, long fullMatrixHeightA,
                   long fullMatrixWidthB, long fullMatrixHeightB, int blockSize);


  virtual void initializeCudaGPU(CUcontext context, CUstream stream, int cudaId, int numGPUs, int pipelineId, int numPipelines);
  virtual void shutdownCuda();

  virtual void executeGPUTask(std::shared_ptr<MatrixBlockMulData<MatrixMemoryData_t>> data, CUstream stream);

  virtual std::string getName() {
    return "MatrixMulBlkTask";
  }
  virtual MatrixMulBlkTask *copy() {
    return new MatrixMulBlkTask(this->getContexts(), this->getCudaIds(), this->getNumGPUs(), fullMatrixWidthA, fullMatrixHeightA, fullMatrixWidthB, fullMatrixHeightB, blockSize);
  }

 private:
  long fullMatrixWidthA;
  long fullMatrixHeightA;
  long fullMatrixWidthB;
  long fullMatrixHeightB;
  int blockSize;

  cublasHandle_t handle;
  double *alpha;
  double *beta;
};

#endif //HTGS_MATRIXMULBLKTASK_H

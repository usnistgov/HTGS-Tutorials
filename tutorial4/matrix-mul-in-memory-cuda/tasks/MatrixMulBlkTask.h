
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

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

class MatrixMulBlkTask : public htgs::ICudaTask<MatrixBlockMulData<MatrixMemoryData_t>,
                                                MatrixBlockData<MatrixMemoryData_t>> {

 public:
  MatrixMulBlkTask(CUcontext *contexts,
                   int *cudaIds,
                   int numGpus,
                   long fullMatrixWidthA,
                   long fullMatrixHeightA,
                   long fullMatrixWidthB,
                   long fullMatrixHeightB,
                   int blockSize) :
      ICudaTask(contexts, cudaIds, numGpus),
      fullMatrixWidthA(fullMatrixWidthA), fullMatrixHeightA(fullMatrixHeightA),
      fullMatrixWidthB(fullMatrixWidthB), fullMatrixHeightB(fullMatrixHeightB), blockSize(blockSize) {}

  virtual ~MatrixMulBlkTask() {

  }
  virtual void initializeCudaGPU(CUcontext context,
                                 CUstream stream,
                                 int cudaId,
                                 int numGPUs,
                                 int pipelineId,
                                 int numPipelines) {
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

    auto result = this->memGet<double *>("MatrixC", new MatrixMemoryRule(1));

    double *alpha = new double[1]{1.0};
    double *beta = new double[1]{0.0};

    cublasDgemm_v2(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   (int) height,
                   (int) width,
                   (int) matAData->getMatrixWidth(),
                   alpha,
                   matrixA->get(),
                   (int) matAData->getMatrixHeight(),
                   matrixB->get(),
                   (int) matBData->getMatrixHeight(),
                   beta,
                   result->get(),
                   (int) height);
//    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, height, width, matAData->getMatrixWidth(), 1.0, matrixA, fullMatrixWidthA,
//                matrixB, fullMatrixWidthB, 0.0, result, width);

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
    return new MatrixMulBlkTask(this->getContexts(),
                                this->getCudaIds(),
                                this->getNumGPUs(),
                                fullMatrixWidthA,
                                fullMatrixHeightA,
                                fullMatrixWidthB,
                                fullMatrixHeightB,
                                blockSize);
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


// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//


#ifndef HTGS_MATRIXMULBLKCUDATASK_H
#define HTGS_MATRIXMULBLKCUDATASK_H
#include <cuda.h>
#include <cublas_v2.h>

#include <htgs/api/ICudaTask.hpp>
#include "../../tutorial-utils/matrix-library/data/MatrixBlockData.h"
#include "../../tutorial-utils/matrix-library/data/MatrixBlockMulData.h"
#include "../../tutorial-utils/matrix-library/rules/MatrixMemoryRule.h"

class MatrixMulBlkCudaTask : public htgs::ICudaTask<MatrixBlockMulData<htgs::m_data_t<double>>,
                                                MatrixBlockData<htgs::m_data_t<double>>> {

 public:
  MatrixMulBlkCudaTask(CUcontext *contexts,
                   int *cudaIds,
                   size_t numGpus) :
      ICudaTask(contexts, cudaIds, numGpus) {
    alpha[0] = 1.0;
    beta[0] = 0.0;
  }

  virtual ~MatrixMulBlkCudaTask() {

  }
  void initializeCudaGPU() override {

    cublasCreate_v2(&handle);
    cublasSetStream_v2(handle, this->getStream());
  }
  void shutdownCuda() override {
    cublasDestroy_v2(handle);
  }

  void executeTask(std::shared_ptr<MatrixBlockMulData<htgs::m_data_t<double>>> data) override {

    auto matAData = data->getMatrixA();
    auto matBData = data->getMatrixB();

    auto matrixA = matAData->getMatrixData();
    auto matrixB = matBData->getMatrixData();

    size_t width = matBData->getMatrixWidth();
    size_t height = matAData->getMatrixHeight();

    auto result = this->getMemory<double>(matrixTypeToString(MatrixType::MatrixC), new MatrixMemoryRule(1));

    cublasDgemm_v2(handle,
                   CUBLAS_OP_N,
                   CUBLAS_OP_N,
                   (int) height,
                   (int) width,
                   (int) matAData->getMatrixWidth(),
                   alpha,
                   matrixA->get(),
                   (int) matAData->getLeadingDimension(),
                   matrixB->get(),
                   (int) matBData->getLeadingDimension(),
                   beta,
                   result->get(),
                   (int) height);

    this->syncStream();

    std::shared_ptr<MatrixRequestData> matReq(new MatrixRequestData(matAData->getRequest()->getRow(),
                                                                    matBData->getRequest()->getCol(),
                                                                    MatrixType::MatrixC));

    addResult(new MatrixBlockData<htgs::m_data_t<double>>(matReq, result, width, height, height));

    this->releaseMemory(matrixA);
    this->releaseMemory(matrixB);

  }
  std::string getName() override {
    return "MatrixMulBlkCudaTask";
  }

  MatrixMulBlkCudaTask *copy() override{
    return new MatrixMulBlkCudaTask(this->getContexts(),
                                this->getCudaIds(),
                                this->getNumGPUs());
  }

 private:

  double alpha[1];
  double beta[1];
  cublasHandle_t handle;
};

#endif //HTGS_MATRIXMULBLKCUDATASK_H

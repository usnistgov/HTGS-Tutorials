
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 6/15/16.
//

#ifndef HTGS_TUTORIALS_MATRIXCOPYINGEMMTASK_H
#define HTGS_TUTORIALS_MATRIXCOPYINGEMMTASK_H

#include <htgs/api/ICudaTask.hpp>
#include "../../common/data/MatrixBlockData.h"
#include <cuda.h>

class MatrixCopyInGemmTask : public htgs::ICudaTask<MatrixBlockMulDataPreCopy, MatrixBlockMulData<MatrixMemoryData_t>> {
 public:
  MatrixCopyInGemmTask(int blockSize,
                   CUcontext *contexts, int *cudaIds, int numGpus, long leadingDimensionFullMatrix) :
      ICudaTask(contexts, cudaIds, numGpus),
      blockSize(blockSize),
      leadingDimensionFullMatrix(leadingDimensionFullMatrix) {}

  virtual void executeTask(std::shared_ptr<MatrixBlockMulDataPreCopy> data) {

    // CPU Memory
    auto upperMatrix = data->getUpperMatrix();

    double *upperMem = upperMatrix->getMatrixData();

    // Cuda Memory
//    std::cout << "Getting memory upper" << std::endl;
    auto memoryOutUpper = this->getMemory<double>("FactorUpperMem", new MatrixMemoryRule(1));
//    std::cout << "Done getting memory upper" << "row: " << upperMatrix->getRequest()->getRow() << ", " << upperMatrix->getRequest()->getCol() << std::endl;

    cublasSetMatrixAsync((int) upperMatrix->getMatrixHeight(), (int) upperMatrix->getMatrixWidth(), sizeof(double),
                         upperMem, (int) leadingDimensionFullMatrix,
                         memoryOutUpper->get(), (int) upperMatrix->getMatrixHeight(), this->getStream());


    auto resultMatrix = data->getResultMatrix();
    double *resultMem = resultMatrix->getMatrixData();

//    std::cout << "Getting memory result" << std::endl;
    auto memoryOutResult = this->getMemory<double>("ResultMatrixMem", new MatrixMemoryRule(1));
//    std::cout << "Done getting memory result" << "row: " << resultMatrix->getRequest()->getRow() << ", " << resultMatrix->getRequest()->getCol() << std::endl;

    cublasSetMatrixAsync((int) resultMatrix->getMatrixHeight(), (int) resultMatrix->getMatrixWidth(), sizeof(double),
                         resultMem, (int) leadingDimensionFullMatrix,
                         memoryOutResult->get(), (int) resultMatrix->getMatrixHeight(), this->getStream());


    std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> upperMatrixOut(
        new MatrixBlockData<MatrixMemoryData_t>(upperMatrix->getRequest(),
                                                memoryOutUpper,
                                                upperMatrix->getMatrixWidth(),
                                                upperMatrix->getMatrixHeight()));

    std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> resultMatrixOut(
        new MatrixBlockData<MatrixMemoryData_t>(resultMatrix->getRequest(),
                                                memoryOutResult,
                                                resultMatrix->getMatrixWidth(),
                                                resultMatrix->getMatrixHeight()));

    this->syncStream();

    this->addResult(new MatrixBlockMulData<MatrixMemoryData_t>(data->getLowerMatrix(),
                                                               upperMatrixOut, resultMatrixOut));
  }

  virtual void shutdownCuda() {
  }

  virtual std::string getName() {
    return "CudaCopyInGEMMTask";
  }

  virtual MatrixCopyInGemmTask *copy() {
    return new MatrixCopyInGemmTask(
                                this->blockSize,
                                this->getContexts(),
                                this->getCudaIds(),
                                this->getNumGPUs(),
                                this->leadingDimensionFullMatrix);
  }

 private:
  int blockSize;
  long leadingDimensionFullMatrix;
};

#endif //HTGS_TUTORIALS_MATRIXCOPYINGEMMTASK_H

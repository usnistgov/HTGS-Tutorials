
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//


#ifndef HTGS_MATMULBLKTASKWITHMEMRELEASE_H
#define HTGS_MATMULBLKTASKWITHMEMRELEASE_H

#include "../../../tutorial-utils/matrix-library/operations/matmul.h"
#include <htgs/api/ITask.hpp>

class MatMulBlkTaskWithMemRelease : public htgs::ITask<MatrixBlockMulData<htgs::m_data_t<double>>, MatrixBlockData<double *>> {

 public:
  MatMulBlkTaskWithMemRelease(size_t numThreads, bool colMajor) :
      ITask(numThreads), colMajor(colMajor) {}


  virtual void executeTask(std::shared_ptr<MatrixBlockMulData<htgs::m_data_t<double>>> data) {

    auto matAData = data->getMatrixA();
    auto matBData = data->getMatrixB();

    htgs::m_data_t<double> matrixA = matAData->getMatrixData();
    htgs::m_data_t<double> matrixB = matBData->getMatrixData();

    size_t width = matBData->getMatrixWidth();
    size_t height = matAData->getMatrixHeight();

    size_t lda = matAData->getLeadingDimension();
    size_t ldb = matBData->getLeadingDimension();

    size_t ldc;

    if (colMajor)
      ldc = height;
    else
      ldc = width;

    double *result = new double[width * height];

    computeMatMul(height, width, matAData->getMatrixWidth(), 1.0, matrixA->get(), lda,
                  matrixB->get(), ldb, 0.0, result, ldc, colMajor);

    std::shared_ptr<MatrixRequestData> matReq(new MatrixRequestData(matAData->getRequest()->getRow(),
                                                                    matBData->getRequest()->getCol(),
                                                                    MatrixType::MatrixC));

//    std::cout << "Computing A(" << matAData->getRequest()->getRow() << ", " << matAData->getRequest()->getCol() <<
//              ") x B(" << matBData->getRequest()->getRow() << ", " << matBData->getRequest()->getCol() <<
//              ") = C(" << matReq->getRow() << ", " << matReq->getCol() << ")" << std::endl;

    this->addResult(new MatrixBlockData<double *>(matReq, result, width, height, ldc));

    this->releaseMemory(matrixA);
    this->releaseMemory(matrixB);
  }
  virtual std::string getName() {
    return "MatMulBlkTaskWithMemRelease";
  }

  virtual MatMulBlkTaskWithMemRelease *copy() {
    return new MatMulBlkTaskWithMemRelease(this->getNumThreads(), colMajor);
  }


 private:
  bool colMajor;

};

#endif //HTGS_MATMULBLKTASKWITHMEMRELEASE_H

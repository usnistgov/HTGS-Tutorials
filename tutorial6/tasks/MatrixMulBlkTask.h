
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//


#ifndef HTGS_MATRIXMULBLKTASK_H
#define HTGS_MATRIXMULBLKTASK_H

#include "../data/MatrixBlockMulData.h"
#include "../data/MatrixBlockData.h"
#include <htgs/api/ITask.hpp>

class MatrixMulBlkTask : public htgs::ITask<MatrixBlockMulData<double *>, MatrixBlockData<double *>> {

 public:
  MatrixMulBlkTask(int numThreads,
                   int fullMatrixWidthA,
                   int fullMatrixHeightA,
                   int fullMatrixWidthB,
                   int fullMatrixHeightB,
                   int blockSize) :
      ITask(numThreads), fullMatrixWidthA(fullMatrixWidthA), fullMatrixHeightA(fullMatrixHeightA),
      fullMatrixWidthB(fullMatrixWidthB), fullMatrixHeightB(fullMatrixHeightB), blockSize(blockSize) {}


  virtual void executeTask(std::shared_ptr<MatrixBlockMulData<double *>> data) {

    auto matAData = data->getMatrixA();
    auto matBData = data->getMatrixB();
    auto matCData = data->getMatrixC();

    double *matrixA = matAData->getMatrixData();
    double *matrixB = matBData->getMatrixData();
    double *result = matCData->getMatrixData();

    long width = matBData->getMatrixWidth();
    long height = matAData->getMatrixHeight();


    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                (int)height,
                (int)width,
                (int)matAData->getMatrixWidth(),
                -1.0,
                matrixA,
                fullMatrixHeightA,
                matrixB,
                fullMatrixHeightB,
                1.0,
                result,
                fullMatrixHeightA);

    addResult(matCData);

  }
  virtual std::string getName() {
    return "MatrixMulBlkTask";
  }
  virtual MatrixMulBlkTask *copy() {
    return new MatrixMulBlkTask(this->getNumThreads(),
                                fullMatrixWidthA,
                                fullMatrixHeightA,
                                fullMatrixWidthB,
                                fullMatrixHeightB,
                                blockSize);
  }


 private:
  int fullMatrixWidthA;
  int fullMatrixHeightA;
  int fullMatrixWidthB;
  int fullMatrixHeightB;
  int blockSize;
};

#endif //HTGS_MATRIXMULBLKTASK_H

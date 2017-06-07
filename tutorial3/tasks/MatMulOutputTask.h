
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 3/8/16.
//

#ifndef HTGS_MATMULOUTPUTTASKNOWRITE_H
#define HTGS_MATMULOUTPUTTASKNOWRITE_H

#include <fstream>
#include <htgs/api/ITask.hpp>
#include "../../tutorial-utils/util-filesystem.h"

#include "../../tutorial-utils/matrix-library/data/MatrixBlockData.h"
#include "../../tutorial-utils/util-matrix.h"

class MatMulOutputTask : public htgs::ITask<MatrixBlockData<double *>, MatrixRequestData> {
 public:

  MatMulOutputTask(double *matrix, size_t leadingDim, size_t blockSize, bool colMajor) :
      matrix(matrix), leadingDim(leadingDim), blockSize(blockSize), colMajor(colMajor) { }

  virtual void executeTask(std::shared_ptr<MatrixBlockData<double *>> data) {
    size_t col = data->getRequest()->getCol();
    size_t row = data->getRequest()->getRow();

    double *startLocation;

    if (colMajor)
      startLocation = &this->matrix[IDX2C(blockSize*row, blockSize*col, leadingDim)];
    else
      startLocation = &this->matrix[blockSize * col + blockSize * row * leadingDim];

    size_t dataWidth = data->getMatrixWidth();
    size_t dataHeight = data->getMatrixHeight();
    double *matrixData = data->getMatrixData();
    if (colMajor)
        for (size_t c = 0; c < dataWidth; c++) {
          for (size_t r = 0; r < dataHeight; r++) {
            startLocation[IDX2C(r, c, leadingDim)] = matrixData[IDX2C(r, c, data->getLeadingDimension())];
        }
      }
    else
      for (size_t r = 0; r < dataHeight; r++) {
        for (size_t c = 0; c < dataWidth; c++) {
            startLocation[r * leadingDim + c] = matrixData[r * data->getLeadingDimension() + c];
        }
      }

    delete[] matrixData;
    matrixData = nullptr;

    addResult(data->getRequest());
  }
  virtual std::string getName() {
    return "MatMulOutputTask";
  }
  virtual MatMulOutputTask *copy() {
    return new MatMulOutputTask(matrix, leadingDim, blockSize, colMajor);
  }

 private:
  double *matrix;
  size_t leadingDim;
  size_t blockSize;
  bool colMajor;

};
#endif //HTGS_MATMULOUTPUTTASKNOWRITE_H

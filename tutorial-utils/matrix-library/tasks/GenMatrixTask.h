
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//

#include <htgs/api/ITask.hpp>
#include <cmath>
#include "../data/MatrixRequestData.h"
#include "../data/MatrixBlockData.h"
#ifndef HTGS_GENMATRIXTASK_H
#define HTGS_GENMATRIXTASK_H

class GenMatrixTask : public htgs::ITask<MatrixRequestData, MatrixBlockData<double *>> {

 public:

  GenMatrixTask(size_t numThreads, size_t blockSize, size_t fullMatrixWidth, size_t fullMatrixHeight) :
      ITask(numThreads), blockSize(blockSize), fullMatrixHeight(fullMatrixHeight), fullMatrixWidth(fullMatrixWidth) {
    numBlocksRows = (size_t) ceil((double) fullMatrixHeight / (double) blockSize);
    numBlocksCols = (size_t) ceil((double) fullMatrixWidth / (double) blockSize);
  }

  virtual ~GenMatrixTask() {

  }

  virtual void executeTask(std::shared_ptr<MatrixRequestData> data) {
    size_t row = data->getRow();
    size_t col = data->getCol();

    size_t matrixWidth;
    size_t matrixHeight;

    if (col == numBlocksCols - 1 && fullMatrixWidth % blockSize != 0)
      matrixWidth = fullMatrixWidth % blockSize;
    else
      matrixWidth = blockSize;

    if (row == numBlocksRows - 1 && fullMatrixHeight % blockSize != 0)
      matrixHeight = fullMatrixHeight % blockSize;
    else
      matrixHeight = blockSize;

    // Allocate matrix Memory
    double *matrixData = new double[matrixHeight * matrixWidth];

    // Initialize with a simple value
    for (size_t i = 0; i < matrixWidth * matrixHeight; i++)
      matrixData[i] = 2.0;

    addResult(new MatrixBlockData<double *>(data, matrixData, matrixWidth, matrixHeight));

  }
  virtual std::string getName() {
    return "GenMatrixTask";
  }
  virtual htgs::ITask<MatrixRequestData, MatrixBlockData<double *>> *copy() {
    return new GenMatrixTask(this->getNumThreads(), blockSize, fullMatrixWidth, fullMatrixHeight);
  }

  size_t getNumBlocksRows() const {
    return numBlocksRows;
  }
  size_t getNumBlocksCols() const {
    return numBlocksCols;
  }
 private:
  size_t blockSize;
  size_t fullMatrixWidth;
  size_t fullMatrixHeight;
  size_t numBlocksRows;
  size_t numBlocksCols;

};

#endif //HTGS_GENMATRIXTASK_H

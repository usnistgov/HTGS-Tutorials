
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//
#ifndef HTGS_READMATRIXTASKMMAP_H
#define HTGS_READMATRIXTASKMMAP_H

#include <htgs/api/ITask.hpp>
#include <cmath>
#include <sys/mman.h>
#include <fcntl.h>
#include <err.h>

#include "../../../tutorial-utils/matrix-library/data/MatrixRequestData.h"
#include "../../../tutorial-utils/matrix-library/data/MatrixBlockData.h"

class ReadMatrixTaskMMap : public htgs::ITask<MatrixRequestData, MatrixBlockData<double *>> {

 public:

  ReadMatrixTaskMMap(size_t numThreads,
                 MatrixType type,
                 size_t blockSize,
                 size_t fullMatrixWidth,
                 size_t fullMatrixHeight,
                 std::string directory,
                 bool colMajor) :
      ITask(numThreads),
      blockSize(blockSize),
      fullMatrixHeight(fullMatrixHeight),
      fullMatrixWidth(fullMatrixWidth),
      directory(directory),
      colMajor(colMajor)
      {
    this->type = type;
    numBlocksRows = (size_t) ceil((double) fullMatrixHeight / (double) blockSize);
    numBlocksCols = (size_t) ceil((double) fullMatrixWidth / (double) blockSize);
        matrixName = matrixTypeToString(type);
  }

  virtual ~ReadMatrixTaskMMap() {
    munmap(this->mmapMatrix, sizeof(double) * fullMatrixHeight * fullMatrixWidth);
  }
  virtual void initialize() {
    std::string matrixName = matrixTypeToString(type);

    std::string fileName(directory + "/" + matrixName);
    int fd = -1;
    if ((fd = open(fileName.c_str(), O_RDONLY)) == -1) {
      std::cerr << "Failed to open file: " << fileName << std::endl;
      err(1, "open failed");
    }

    this->mmapMatrix =
        (double *) mmap(NULL, fullMatrixWidth * fullMatrixHeight * sizeof(double), PROT_READ, MAP_SHARED, fd, 0);

    if (this->mmapMatrix == MAP_FAILED) {
      close(fd);
      err(2, "Error mmapping file");
    }
  }

  virtual void executeTask(std::shared_ptr<MatrixRequestData> data) {
    std::string matrixName;

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

    // compute starting location of pointer
    double *memPtr;

    if (colMajor) {
      memPtr = mmapMatrix + (blockSize * col * fullMatrixHeight + blockSize * row);
      addResult(new MatrixBlockData<double *>(data, memPtr, matrixWidth, matrixHeight, fullMatrixHeight));
    } else{
      memPtr = mmapMatrix + (blockSize * col + blockSize * row * fullMatrixWidth);
      addResult(new MatrixBlockData<double *>(data, memPtr, matrixWidth, matrixHeight, fullMatrixWidth));
    }



  }
  virtual std::string getName() {
    return "ReadMatrixTaskMMap(" + matrixName + ")";
  }
  virtual ReadMatrixTaskMMap *copy() {
    return new ReadMatrixTaskMMap(this->getNumThreads(),
                              this->type,
                              blockSize,
                              fullMatrixWidth,
                              fullMatrixHeight,
                              directory,
                              colMajor);
  }

  size_t getNumBlocksRows() const {
    return numBlocksRows;
  }
  size_t getNumBlocksCols() const {
    return numBlocksCols;
  }

 private:
  MatrixType type;
  double *mmapMatrix;
  size_t blockSize;
  size_t fullMatrixWidth;
  size_t fullMatrixHeight;
  size_t numBlocksRows;
  size_t numBlocksCols;
  std::string directory;
  std::string matrixName;
  bool colMajor;

};

#endif //HTGS_READMATRIXTASKMMAP_H

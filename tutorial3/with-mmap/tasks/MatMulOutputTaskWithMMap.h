
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 3/8/16.
//

#ifndef HTGS_MATMULOUTPUTTASKWITHMMAP_H
#define HTGS_MATMULOUTPUTTASKWITHMMAP_H

#include <fstream>
#include <htgs/api/ITask.hpp>
#include <sys/mman.h>
#include <zconf.h>
#include <err.h>
#include <fcntl.h>
#include "../../../tutorial-utils/util-filesystem.h"
#include "../../../tutorial-utils/matrix-library/data/MatrixBlockData.h"
#include "../../../tutorial-utils/util-matrix.h"

class MatMulOutputTaskWithMMap : public htgs::ITask<MatrixBlockData<double *>, MatrixRequestData> {
 public:

  MatMulOutputTaskWithMMap(std::string directory, size_t matrixWidth, size_t matrixHeight, size_t blockSize, bool colMajor) :
      matrixWidth(matrixWidth), matrixHeight(matrixHeight), blockSize(blockSize), colMajor(colMajor) {
    this->directory = directory;
    create_dir(this->directory);
  }

  ~MatMulOutputTaskWithMMap() {
    munmap(this->mmapMatrix, sizeof(double) * matrixHeight * matrixWidth);
  }

  virtual void initialize() {
    std::string fileName(directory + "/matrixC_HTGS");
    int fd = -1;
    if ((fd = open(fileName.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0600)) == -1) {
      err(1, "write open failed");
    }

    // stretch the file to the size of the mmap
    if (lseek(fd, matrixHeight * matrixWidth * sizeof(double) - 1, SEEK_SET) == -1) {
      close(fd);
      err(2, "Error using lseek to stretch the file");
    }

    // Write at end to ensure file has the correct size
    if (write(fd, "", 1) == -1) {
      close(fd);
      err(3, "Error writing to complete stretching the file");
    }

    this->mmapMatrix =
        (double *) mmap(NULL, matrixHeight * matrixWidth * sizeof(double), PROT_WRITE, MAP_SHARED, fd, 0);

    if (this->mmapMatrix == MAP_FAILED) {
      close(fd);
      err(3, "Error mmaping write file");
    }

    close(fd);
  }

  virtual void shutdown() {
    if (msync(mmapMatrix, matrixHeight * matrixWidth * sizeof(double), MS_SYNC) == -1) {
      err(5, "Could not sync the file to disk");
    }
  }

  virtual void executeTask(std::shared_ptr<MatrixBlockData<double *>> data) {
    size_t col = data->getRequest()->getCol();
    size_t row = data->getRequest()->getRow();


    double *startLocation;
    if (colMajor)
      startLocation = &this->mmapMatrix[IDX2C(blockSize * row, blockSize * col, matrixHeight)];
    else
      startLocation = &this->mmapMatrix[blockSize * col + blockSize * row * matrixWidth];

    size_t dataWidth = data->getMatrixWidth();
    size_t dataHeight = data->getMatrixHeight();

    double *matrixData = data->getMatrixData();

    if (colMajor)
    {
      for (size_t c = 0; c < dataWidth; c++) {
        for (size_t r = 0; r < dataHeight; r++) {
          startLocation[IDX2C(r, c, matrixHeight)] = matrixData[IDX2C(r, c, data->getLeadingDimension())];
        }
      }
    }
    else
    {
      for (size_t r = 0; r < dataHeight; r++) {
        for (size_t c = 0; c < dataWidth; c++) {
          startLocation[r * matrixWidth + c] = matrixData[r * data->getLeadingDimension() + c];
        }
      }
    }

    if (msync(mmapMatrix, matrixWidth * matrixHeight* sizeof(double), MS_SYNC) == -1) {
      err(5, "Could not sync the file to disk");
    }
    delete[] matrixData;
    matrixData = nullptr;

    addResult(data->getRequest());
  }
  virtual std::string getName() {
    return "MatMulOutputTaskWithDisk";
  }
  virtual MatMulOutputTaskWithMMap *copy() {
    return new MatMulOutputTaskWithMMap(directory, matrixWidth, matrixHeight, blockSize, colMajor);
  }

 private:
  double *mmapMatrix;
  std::string directory;
  size_t matrixWidth;
  size_t matrixHeight;
  size_t blockSize;
  bool colMajor;

};
#endif //HTGS_MATMULOUTPUTTASKWITHMMAP_H

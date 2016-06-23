
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//
#ifndef HTGS_READMATRIXTASK_H
#define HTGS_READMATRIXTASK_H

#include <htgs/api/ITask.hpp>
#include <cmath>
#include "../memory/MatrixMemoryRule.h"

class ReadMatrixTask : public htgs::ITask<MatrixRequestData, MatrixBlockData<MatrixMemoryData_t>> {

 public:

  ReadMatrixTask(int numThreads,
                 int blockSize,
                 int fullMatrixWidth,
                 int fullMatrixHeight,
                 std::string directory,
                 std::string matrixName) :
      ITask(numThreads),
      blockSize(blockSize),
      fullMatrixHeight(fullMatrixHeight),
      fullMatrixWidth(fullMatrixWidth),
      directory(directory),
      matrixName(matrixName) {
    numBlocksRows = (int) ceil((double) fullMatrixHeight / (double) blockSize);
    numBlocksCols = (int) ceil((double) fullMatrixWidth / (double) blockSize);
  }

  virtual ~ReadMatrixTask() {}
  virtual void initialize(int pipelineId,
                          int numPipeline) {

  }
  virtual void shutdown() {
  }

  virtual void executeTask(std::shared_ptr<MatrixRequestData> data) {
    std::string matrixName;

    int numBlocksC = 0;
    matrixName = matrixTypeToString(data->getType());

    switch (data->getType()) {
      case MatrixType::MatrixA:
        numBlocksC = numBlocksCols;
        break;
      case MatrixType::MatrixB:
        numBlocksC = numBlocksRows;
        break;
      case MatrixType::MatrixC: return;
    }

    MatrixMemoryData_t matrixData = this->memGet<double *>(matrixName, new MatrixMemoryRule(numBlocksC));

    int row = data->getRow();
    int col = data->getCol();

    int matrixWidth;
    int matrixHeight;

    if (col == numBlocksCols - 1 && fullMatrixWidth % blockSize != 0)
      matrixWidth = fullMatrixWidth % blockSize;
    else
      matrixWidth = blockSize;

    if (row == numBlocksRows - 1 && fullMatrixHeight % blockSize != 0)
      matrixHeight = fullMatrixHeight % blockSize;
    else
      matrixHeight = blockSize;

    std::string fileName(directory + "/" + matrixName + "/" + std::to_string(row) + "_" + std::to_string(col));

//    std::cout << "Reading file: " << fileName << std::endl;

    // Read data
    std::ifstream file(fileName, std::ios::binary);

    file.read((char *) matrixData->get(), sizeof(double) * matrixWidth * matrixHeight);

    addResult(new MatrixBlockData<MatrixMemoryData_t>(data, matrixData, matrixWidth, matrixHeight));

  }
  virtual std::string getName() {
    return "ReadMatrixTask(" + matrixName + ")";
  }
  virtual htgs::ITask<MatrixRequestData, MatrixBlockData<MatrixMemoryData_t>> *copy() {
    return new ReadMatrixTask(this->getNumThreads(),
                              blockSize,
                              fullMatrixWidth,
                              fullMatrixHeight,
                              directory,
                              matrixName);
  }
  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
    return inputConnector->isInputTerminated();
  }

  int getNumBlocksRows() const {
    return numBlocksRows;
  }
  int getNumBlocksCols() const {
    return numBlocksCols;
  }
 private:
  int blockSize;
  int fullMatrixWidth;
  int fullMatrixHeight;
  int numBlocksRows;
  int numBlocksCols;
  std::string directory;
  std::string matrixName;

};

#endif //HTGS_GENERATEMATRIXTASK_H

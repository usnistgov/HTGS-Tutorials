
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 3/8/16.
//

#ifndef HTGS_OUTPUTTASK_H
#define HTGS_OUTPUTTASK_H

class OutputTask : public htgs::ITask<MatrixBlockData<double *>, MatrixBlockData<double *>> {
 public:

  OutputTask(double *matrix, int fullMatrixWidth, int fullMatrixHeight, int blockSize) :
      matrix(matrix), fullMatrixWidth(fullMatrixWidth), fullMatrixHeight(fullMatrixHeight), blockSize(blockSize) {
    numBlocksRows = (int) ceil((double) fullMatrixHeight / (double) blockSize);
    numBlocksCols = (int) ceil((double) fullMatrixWidth / (double) blockSize);
  }
  virtual ~OutputTask() {
  }

  virtual void initialize(int pipelineId, int numPipeline) {
  }

  virtual void shutdown() {
  }
  virtual void executeTask(std::shared_ptr<MatrixBlockData<double *>> data) {
    int col = data->getRequest()->getCol();
    int row = data->getRequest()->getRow();

    double *startLocation = &this->matrix[blockSize * col + blockSize * row * fullMatrixWidth];

    long dataWidth = data->getMatrixWidth();
    long dataHeight = data->getMatrixHeight();
    double *matrixData = data->getMatrixData();
    for (long r = 0; r < dataHeight; r++) {
      for (long c = 0; c < dataWidth; c++) {
        startLocation[r * fullMatrixWidth + c] = matrixData[r * dataWidth + c];
      }
    }

    delete[] matrixData;
    matrixData = nullptr;

    addResult(data);
  }
  virtual std::string getName() {
    return "OutputTask";
  }
  virtual OutputTask *copy() {
    return new OutputTask(matrix, this->fullMatrixWidth, this->fullMatrixHeight, this->blockSize);
  }
  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
    return inputConnector->isInputTerminated();
  }

 private:

  double *matrix;
  int fullMatrixWidth;
  int fullMatrixHeight;
  int blockSize;
  int numBlocksRows;
  int numBlocksCols;

};
#endif //HTGS_OUTPUTTASK_H

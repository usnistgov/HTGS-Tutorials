//
// Created by tjb3 on 3/8/16.
//

#ifndef HTGS_OUTPUTTASK_H
#define HTGS_OUTPUTTASK_H

class OutputTask : public htgs::ITask<MatrixBlockData<double *>, MatrixRequestData> {
 public:

  OutputTask(double *matrix, int fullMatrixWidth, int fullMatrixHeight, int blockSize) :
      matrix(matrix), fullMatrixWidth(fullMatrixWidth), fullMatrixHeight(fullMatrixHeight), blockSize(blockSize) {
    numBlocksRows = (int)ceil((double)fullMatrixHeight / (double)blockSize);
    numBlocksCols = (int)ceil((double)fullMatrixWidth / (double)blockSize);
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

    double *startLocation = &this->matrix[blockSize*col+blockSize*row*fullMatrixWidth];

    long dataWidth = data->getMatrixWidth();
    long dataHeight = data->getMatrixHeight();
    double *matrixData = data->getMatrixData();
    for (long r = 0; r < dataHeight; r++)
    {
      for (long c = 0; c < dataWidth; c++)
      {
        startLocation[r *fullMatrixWidth + c] = matrixData[r*dataWidth+c];
      }
    }

    delete [] matrixData;
    matrixData = nullptr;

    addResult(data->getRequest());
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

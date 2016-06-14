//
// Created by tjb3 on 3/8/16.
//

#ifndef HTGS_OUTPUTTASK_H
#define HTGS_OUTPUTTASK_H

class OutputTask : public htgs::ITask<MatrixBlockData<double *>, MatrixRequestData> {
 public:

  OutputTask(std::string directory, int fullMatrixWidth, int fullMatrixHeight, int blockSize) :
      directory(directory), fullMatrixWidth(fullMatrixWidth), fullMatrixHeight(fullMatrixHeight), blockSize(blockSize) {
    numBlocksRows = (int)ceil((double)fullMatrixHeight / (double)blockSize);
    numBlocksCols = (int)ceil((double)fullMatrixWidth / (double)blockSize);
  }
  virtual ~OutputTask() {
    munmap(this->mmapMatrix, sizeof(double)*fullMatrixHeight*fullMatrixWidth);

  }
  virtual void initialize(int pipelineId, int numPipeline) {
    std::string fileName(directory + "/matrixC_HTGS");
    int fd = -1;
    if ((fd = open(fileName.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600)) == -1) {
      std::cerr << "failed to open file for writing: " << fileName << std::endl;
      err(1, "write open failed");
    }

    // stretch the file to the size of the mmap
    if (lseek(fd, fullMatrixHeight*fullMatrixWidth*sizeof(double)-1, SEEK_SET) == -1)
    {
      close(fd);
      err(2, "Error using lseek to stretch the file");
    }

    // Write at end to ensure file has the correct size
    if (write(fd, "", 1) == -1)
    {
      close(fd);
      err(3, "Error writing to complete stretching the file");
    }

    this->mmapMatrix = (double *)mmap(NULL, fullMatrixWidth*fullMatrixHeight*sizeof(double), PROT_WRITE, MAP_SHARED, fd, 0);

    if (this->mmapMatrix == MAP_FAILED)
    {
      close(fd);
      err(3, "Error mmaping write file");
    }
  }
  virtual void shutdown() {
    if (msync(mmapMatrix, fullMatrixHeight*fullMatrixWidth*sizeof(double), MS_SYNC) == -1)
    {
      err(5, "Could not sync the file to disk");
    }
  }
  virtual void executeTask(std::shared_ptr<MatrixBlockData<double *>> data) {
    int col = data->getRequest()->getCol();
    int row = data->getRequest()->getRow();

    double *startLocation = this->mmapMatrix+(blockSize*col+blockSize*row*fullMatrixWidth);

    int dataWidth = data->getMatrixWidth();
    int dataHeight = data->getMatrixHeight();

    for (int r = 0; r < dataHeight; r++)
    {
      for (int c = 0; c < dataWidth; c++)
      {
        startLocation[r *fullMatrixWidth + c] = data->getMatrixData()[r*dataWidth+c];
      }
    }

    addResult(data->getRequest());
  }
  virtual std::string getName() {
    return "OutputTask";
  }
  virtual htgs::ITask<MatrixBlockData<double *>, MatrixRequestData> *copy() {
    return new OutputTask(directory, this->fullMatrixWidth, this->fullMatrixHeight, this->blockSize);
  }
  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
    return inputConnector->isInputTerminated();
  }


 private:

  double *mmapMatrix;
  std::string directory;
  int fullMatrixWidth;
  int fullMatrixHeight;
  int blockSize;
  int numBlocksRows;
  int numBlocksCols;

};
#endif //HTGS_OUTPUTTASK_H

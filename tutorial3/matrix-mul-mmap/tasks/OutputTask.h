
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 3/8/16.
//

#ifndef HTGS_OUTPUTTASK_H
#define HTGS_OUTPUTTASK_H

class OutputTask : public htgs::ITask<MatrixBlockData<double *>, MatrixRequestData> {
 public:

  OutputTask(std::string directory, int fullMatrixWidth, int fullMatrixHeight, int blockSize) :
      directory(directory), fullMatrixWidth(fullMatrixWidth), fullMatrixHeight(fullMatrixHeight), blockSize(blockSize) {
    numBlocksRows = (int) ceil((double) fullMatrixHeight / (double) blockSize);
    numBlocksCols = (int) ceil((double) fullMatrixWidth / (double) blockSize);
  }
  virtual ~OutputTask() {
    munmap(this->mmapMatrix, sizeof(double) * fullMatrixHeight * fullMatrixWidth);

  }
  virtual void initialize(int pipelineId, int numPipeline) {
    std::string fileName(directory + "/matrixC_HTGS");
    int fd = -1;
    if ((fd = open(fileName.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t) 0600)) == -1) {
      std::cerr << "failed to open file for writing: " << fileName << std::endl;
      err(1, "write open failed");
    }

    // stretch the file to the size of the mmap
    if (lseek(fd, fullMatrixHeight * fullMatrixWidth * sizeof(double) - 1, SEEK_SET) == -1) {
      close(fd);
      err(2, "Error using lseek to stretch the file");
    }

    // Write at end to ensure file has the correct size
    if (write(fd, "", 1) == -1) {
      close(fd);
      err(3, "Error writing to complete stretching the file");
    }

    this->mmapMatrix =
        (double *) mmap(NULL, fullMatrixWidth * fullMatrixHeight * sizeof(double), PROT_WRITE, MAP_SHARED, fd, 0);

    if (this->mmapMatrix == MAP_FAILED) {
      close(fd);
      err(3, "Error mmaping write file");
    }
  }
  virtual void shutdown() {
    if (msync(mmapMatrix, fullMatrixHeight * fullMatrixWidth * sizeof(double), MS_SYNC) == -1) {
      err(5, "Could not sync the file to disk");
    }
  }
  virtual void executeTask(std::shared_ptr<MatrixBlockData<double *>> data) {
    int col = data->getRequest()->getCol();
    int row = data->getRequest()->getRow();

    double *startLocation = this->mmapMatrix + (blockSize * col + blockSize * row * fullMatrixWidth);

    int dataWidth = data->getMatrixWidth();
    int dataHeight = data->getMatrixHeight();

    for (int r = 0; r < dataHeight; r++) {
      for (int c = 0; c < dataWidth; c++) {
        startLocation[r * fullMatrixWidth + c] = data->getMatrixData()[r * dataWidth + c];
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

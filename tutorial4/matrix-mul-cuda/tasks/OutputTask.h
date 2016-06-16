//
// Created by tjb3 on 3/8/16.
//

#ifndef HTGS_OUTPUTTASK_H
#define HTGS_OUTPUTTASK_H

#include "../../../tutorial-utils/util-filesystem.h"
class OutputTask : public htgs::ITask<MatrixBlockData<double *>, MatrixRequestData> {
 public:

  OutputTask(std::string directory) {
    this->directory = directory + "/matrixC_HTGS";
    create_dir(this->directory);
  }
  virtual ~OutputTask() {

  }
  virtual void initialize(int pipelineId, int numPipeline) {
  }
  virtual void shutdown() {
  }
  virtual void executeTask(std::shared_ptr<MatrixBlockData<double *>> data) {
    std::string fileName(directory + "/" + std::to_string(data->getRequest()->getRow()) + "_" + std::to_string(data->getRequest()->getCol()));

    std::ofstream out(fileName, std::ios::binary);
    out.write((char *)data->getMatrixData(), sizeof(double) * data->getMatrixWidth() * data->getMatrixHeight());

    delete [] data->getMatrixData();

    addResult(data->getRequest());
  }
  virtual std::string getName() {
    return "OutputTask";
  }
  virtual htgs::ITask<MatrixBlockData<double *>, MatrixRequestData> *copy() {
    return new OutputTask(directory);
  }
  virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
    return inputConnector->isInputTerminated();
  }


 private:

  std::string directory;

};
#endif //HTGS_OUTPUTTASK_H

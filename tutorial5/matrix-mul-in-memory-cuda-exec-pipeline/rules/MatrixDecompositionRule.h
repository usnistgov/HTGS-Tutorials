//
// Created by tjb3 on 6/21/16.
//

#ifndef HTGS_TUTORIALS_MATRIXDECOMPOSITIONRULE_H
#define HTGS_TUTORIALS_MATRIXDECOMPOSITIONRULE_H

class MatrixDecompositionRule : public htgs::IRule<MatrixRequestData, MatrixRequestData> {
 public:
  MatrixDecompositionRule(int numGpus) : numGpus(numGpus) {}

  virtual void applyRule(std::shared_ptr<MatrixRequestData> data, int pipelineId) {

    MatrixType type = data->getType();
    int row = data->getRow();
    int col = data->getCol();
    int gpuId;
    switch (type) {
      case MatrixType::MatrixA:
        // One column per GPU
        gpuId = col % numGpus;

        if (pipelineId == gpuId) {
          addResult(data);
//          std::cout << "sending matrixA: " << row << ", " << col << " to gpu: " << pipelineId << std::endl;
        }

        break;
      case MatrixType::MatrixB:
        // One row per GPU
        gpuId = row % numGpus;

        if (pipelineId == gpuId) {
          addResult(data);
//          std::cout << "sending matrixB: " << row << ", " << col << " to gpu: " << pipelineId << std::endl;
        }
        break;

      case MatrixType::MatrixC:break;
    }

  }
  virtual std::string getName() {
    return "MatrixDecompositionRule";
  }

 private:
  int numGpus;
};

#endif //HTGS_TUTORIALS_MATRIXDECOMPOSITIONRULE_H

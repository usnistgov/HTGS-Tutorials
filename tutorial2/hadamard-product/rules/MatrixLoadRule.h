//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXLOADRULE_H
#define HTGS_MATRIXLOADRULE_H
#include <htgs/api/IRule.hpp>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixBlockMulData.h"

class MatrixLoadRule : public htgs::IRule<MatrixBlockData<MatrixMemoryData_t>, MatrixBlockMulData> {

 public:
  MatrixLoadRule(int blockWidth, int blockHeight) {
    this->blockWidth = blockWidth;
    this->blockHeight = blockHeight;

    this->matrixAState = this->allocStateContainer(blockHeight, blockWidth);
    this->matrixBState = this->allocStateContainer(blockHeight, blockWidth);
  }

  ~MatrixLoadRule() {
    delete matrixAState;
    delete matrixBState;
  }

  bool isRuleTerminated(int pipelineId) {
    return false;
  }

  void shutdownRule(int pipelineId) {}

  void applyRule(std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> data, int pipelineId) {
    std::shared_ptr<MatrixRequestData> request = data->getRequest();

    switch (request->getType()) {
      case MatrixType::MatrixA:
        this->matrixAState->set(request->getRow(), request->getCol(), data);

        if (this->matrixBState->has(request->getRow(), request->getCol())) {
          addResult(new MatrixBlockMulData(data, this->matrixBState->get(request->getRow(), request->getCol())));
        }
        break;
      case MatrixType::MatrixB:
        this->matrixBState->set(request->getRow(), request->getCol(), data);

        if (this->matrixAState->has(request->getRow(), request->getCol())) {
          addResult(new MatrixBlockMulData(this->matrixAState->get(request->getRow(), request->getCol()), data));
        }
        break;
      case MatrixType::MatrixC:
        break;
    }
  }

  std::string getName() {
    return "MatrixLoadRule";
  }

 private:
  int blockWidth;
  int blockHeight;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>>> *matrixAState;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>>> *matrixBState;
};
#endif //HTGS_MATRIXLOADRULE_H

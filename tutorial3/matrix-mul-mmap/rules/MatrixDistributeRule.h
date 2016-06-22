//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXDISTRIBUTERULE_H
#define HTGS_MATRIXDISTRIBUTERULE_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixRequestData.h"
#include "../../../tutorial-utils/enums/MatrixType.h"

class MatrixDistributeRule : public htgs::IRule<MatrixRequestData, MatrixRequestData> {

 public:
  MatrixDistributeRule(MatrixType type) {
    this->type = type;
  }

  ~MatrixDistributeRule() {
  }

  bool isRuleTerminated(int pipelineId) {
    return false;
  }

  void shutdownRule(int pipelineId) {
  }

  void applyRule(std::shared_ptr<MatrixRequestData> data, int pipelineId) {
    if (data->getType() == this->type) {
      addResult(data);
    }
  }

  std::string getName() {
    return "MatrixDistributeRule";
  }

 private:
  MatrixType type;
};

#endif //HTGS_MATRIXDISTRIBUTERULE_H

//
// Created by tjb3 on 7/29/16.
//

#ifndef HTGS_TUTORIALS_UPDATERULE_H
#define HTGS_TUTORIALS_UPDATERULE_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixBlockData.h"
class UpdateRule : public htgs::IRule<MatrixBlockData<double *>, MatrixBlockData<double *>>
{
 public:
  UpdateRule(int totalBlocks) : totalBlocks(totalBlocks) {}

  virtual bool isRuleTerminated(int pipelineId) {
    return this->totalBlocks == 0;
  }

  virtual void applyRule(std::shared_ptr<MatrixBlockData<double *>> data, int pipelineId) {
//    int row = data->getRequest()->getRow();
//    int col = data->getRequest()->getCol();

//    std::cout << "Received " << row << ", " << col << " totalBlocks = " << totalBlocks << std::endl;
    // Notify a block has been updated
      totalBlocks--;
      addResult(data);

  }

  std::string getName() {
    return "UpdateRule";
  }

 private:
  int totalBlocks;
};


#endif //HTGS_TUTORIALS_UPDATERULE_H

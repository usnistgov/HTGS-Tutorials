//
// Created by tjb3 on 7/29/16.
//

#ifndef HTGS_TUTORIALS_UPDATERULEMATMUL_H
#define HTGS_TUTORIALS_UPDATERULEMATMUL_H

#include <htgs/api/IRule.hpp>
#include "../../common/data/MatrixBlockData.h"
#include "../../common/data/MatrixBlockMultiData.h"
class UpdateRuleMatMul : public htgs::IRule<MatrixBlockData<data_ptr>, MatrixBlockMultiData>
{
 public:
  UpdateRuleMatMul(int totalBlocks) : totalBlocks(totalBlocks) {}

  virtual bool canTerminateRule(size_t pipelineId) override {
    return this->totalBlocks == 0;
  }

  virtual void applyRule(std::shared_ptr<MatrixBlockData<data_ptr>> data, size_t pipelineId) override {
//    int row = data->getRequest()->getRow();
//    int col = data->getRequest()->getCol();

//    std::cout << "Received " << row << ", " << col << " totalBlocks = " << totalBlocks << std::endl;
    // Notify a block has been updated
      totalBlocks--;

    auto result = new MatrixBlockMultiData(data, nullptr);
      addResult(result);

  }

  std::string getName() override {
    return "UpdateRuleMatMul";
  }

 private:
  int totalBlocks;
};


#endif //HTGS_TUTORIALS_UPDATERULEMATMUL_H

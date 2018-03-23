//
// Created by tjb3 on 7/29/16.
//

#ifndef HTGS_TUTORIALS_UPDATEFACTORRULE_H
#define HTGS_TUTORIALS_UPDATEFACTORRULE_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixPanelData.h"
class UpdateFactorRule : public htgs::IRule<MatrixPanelData, MatrixBlockData<double *>>
{
 public:
  UpdateFactorRule(size_t totalBlocks, size_t gridHeight, htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks) :
      totalBlocks(totalBlocks), gridHeight(gridHeight), matrixBlocks(matrixBlocks) {}

  virtual bool canTerminateRule(size_t pipelineId) override {
    return this->totalBlocks == 0;
  }

  virtual void applyRule(std::shared_ptr<MatrixPanelData> data, size_t pipelineId) override {

    // Notify a block has been updated
    totalBlocks--;
    int row = data->getPanelOperatingDiagonal();
    int col = data->getPanelCol();

    // Update only the top block of the updated panel, unless the next block is along the diagonal
    if (row+1 == col)
    {
      // Skip the diagonal and notify that the entire panel is ready to be factored
      for (row = row + 2; row < gridHeight; row++)
      {
        addResult(matrixBlocks->get(row, col));
      }
    }
    else {
      // Indicate only the top panel is ready to be factored
      addResult(matrixBlocks->get(row + 1, col));
    }
  }

  std::string getName() override {
    return "UpdateFactorRule";
  }

 private:
  size_t totalBlocks;
  size_t gridHeight;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks;
};


#endif //HTGS_TUTORIALS_UPDATEFACTORRULE_H

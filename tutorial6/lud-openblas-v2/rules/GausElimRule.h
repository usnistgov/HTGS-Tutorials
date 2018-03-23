//
// Created by tjb3 on 7/29/16.
//

#ifndef HTGS_TUTORIALS_GAUSELIMRULE_H
#define HTGS_TUTORIALS_GAUSELIMRULE_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixPanelData.h"
class GausElimRule : public htgs::IRule<MatrixPanelData, MatrixBlockData<double *>>
{
 public:

  GausElimRule(size_t  totalBlocksDiagonal, size_t gridHeight, size_t gridWidth, htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks) :
      totalBlocksDiagonal(totalBlocksDiagonal), matrixBlocks(matrixBlocks)
  {}

  virtual bool canTerminateRule(size_t pipelineId) override {
    return totalBlocksDiagonal == 0;
  }

  virtual void applyRule(std::shared_ptr<MatrixPanelData> data, size_t pipelineId) override {

    int col = data->getPanelCol();
    int operatingDiagonal = data->getPanelOperatingDiagonal();

    // If the current col is the operating diagonal, then indicate to do gaus elim
    if (operatingDiagonal+1 == col) {
      addResult(matrixBlocks->get(operatingDiagonal+1, col));
      totalBlocksDiagonal--;
    }

  }

  std::string getName() override {
    return "GausElimRule";
  }

 private:
  size_t totalBlocksDiagonal;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks;
};

#endif //HTGS_TUTORIALS_GAUSELIMRULE_H

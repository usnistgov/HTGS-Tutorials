//
// Created by tjb3 on 7/29/16.
//

#ifndef HTGS_TUTORIALS_GAUSELIMRULELOWER_H
#define HTGS_TUTORIALS_GAUSELIMRULELOWER_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixFactorData.h"
#include "../../../tutorial-utils/util-matrix.h"
class GausElimRuleLower : public htgs::IRule<MatrixBlockData<double *>, MatrixFactorData<double *>>
{

 public:

  GausElimRuleLower(htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks, int gridHeight, int gridWidth) :
      matrixBlocks(matrixBlocks), gridHeight(gridHeight), gridWidth(gridWidth)
  {
    this->lowerState = this->allocStateContainer<bool>(gridHeight, gridWidth, 0);

    // The first column and first row are ready as no updates required
    for (int r = 1; r < gridHeight; r++)
    {
      this->lowerState->assign(r, 0, true);
    }

    for (int c = 1; c < gridWidth; c++)
    {
      this->lowerState->assign(0, c, true);
    }

  }

  virtual ~GausElimRuleLower() {
    delete lowerState;
  }

  virtual void applyRule(std::shared_ptr<MatrixBlockData<double *>> data, size_t pipelineId) override {
    int row = data->getRequest()->getRow();
    int col = data->getRequest()->getCol();

    // Update the state for the data we received
    this->lowerState->assign(row, col, true);

    // Received Diagonal
    if (row == col)
    {
      // Check along diagonal for all blocks ready to be factored
      for (int r = row+1; r < gridHeight; r++)
      {
        bool nextBlockReady = this->lowerState->get(r, col);

        // if the state value is equal to the row value, then it is ready to be sent
        if (nextBlockReady)
        {
          auto matrix = matrixBlocks->get(r, col);
          addResult(new MatrixFactorData<double *>(data, matrix));
        }

      }
    }
      // Check only below the diagonal
    else if (row > col)
    {
      // Check if the current data is ready to be factored
      // Validate the diagonal has been processed
      bool diagonalState = this->lowerState->get(col, col);
      if (diagonalState) {

        // Diagonal and data are ready to be factored
        auto diagonal = matrixBlocks->get(col, col);
        addResult(new MatrixFactorData<double *>(diagonal, data));

      }
    }
//    std::cout << "Lower state: " << std::endl;
//    lowerState->printContents();
//    std::cout << std::endl;

  }

  std::string getName() override {
    return "GausElimRuleLower";
  }

 private:
  // Store state . . .
  htgs::StateContainer<bool> *lowerState;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks;

  int gridWidth;
  int gridHeight;
};


#endif //HTGS_TUTORIALS_GAUSELIMRULELOWER_H

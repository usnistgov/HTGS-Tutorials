//
// Created by tjb3 on 7/29/16.
//

#ifndef HTGS_TUTORIALS_GAUSELIMRULEUPPER_H
#define HTGS_TUTORIALS_GAUSELIMRULEUPPER_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixFactorData.h"
#include "../../../tutorial-utils/util-matrix.h"
#include "../data/MatrixPanelData.h"
class GausElimRuleUpper : public htgs::IRule<MatrixBlockData<double *>, MatrixPanelData>
{

 public:

  GausElimRuleUpper(htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks, int gridHeight, int gridWidth, int blockSize) :
      matrixBlocks(matrixBlocks), gridWidth(gridWidth), gridHeight(gridHeight), blockSize(blockSize)
  {
    this->upperState = this->allocStateContainer<bool>(gridHeight, gridWidth, 0);
    // The first column and first row are ready as no updates required
    for (int r = 1; r < gridHeight; r++)
    {
      this->upperState->assign(r, 0, true);
    }

    for (int c = 1; c < gridWidth; c++)
    {
      this->upperState->assign(0, c, true);
    }
  }

  virtual ~GausElimRuleUpper() {
    delete upperState;
  }

  virtual void applyRule(std::shared_ptr<MatrixBlockData<double *>> data, int pipelineId) {
    int row = data->getRequest()->getRow();
    int col = data->getRequest()->getCol();

    // Update the state for the data we received
    this->upperState->assign(row, col, true);

    // Received Diagonal
    if (row == col) {
      // Check all on same row beyond col
      for (int c = col + 1; c < gridWidth; c++) {
        bool nextBlockReady = this->upperState->get(row, c);

        // if the next block is ready to be factored
        if (nextBlockReady) {
          auto matrix = matrixBlocks->get(row, c);

          int panelCol = matrix->getRequest()->getCol();
          int panelRow = matrix->getRequest()->getRow();

          int activeDiagonal = data->getRequest()->getCol();

          long panelHeight = blockSize * (gridHeight-panelRow);

          MatrixPanelData *panel =
              new MatrixPanelData(panelHeight, blockSize, panelCol, activeDiagonal, PanelState::TOP_FACTORED);
          panel->setMemory(matrix->getMatrixData());

          addResult(panel);

        }
      }
    }
      // Check only above the diagonal
    else if (col > row)
    {
      // Check if the current data is ready to be factored
      // Validate the diagonal has been processed
      bool diagonalState = this->upperState->get(row, row);
      if (diagonalState) {

        // Diagonal and data are ready to be factored
        auto diagonal = matrixBlocks->get(row, row);

        int panelCol = data->getRequest()->getCol();
        int panelRow = data->getRequest()->getRow();

        int activeDiagonal = diagonal->getRequest()->getCol();

        long panelHeight = blockSize * (gridHeight-panelRow);

        MatrixPanelData
            *panel = new MatrixPanelData(panelHeight, blockSize, panelCol, activeDiagonal, PanelState::TOP_FACTORED);
        panel->setMemory(data->getMatrixData());

        addResult(panel);

      }
    }

//    std::cout << "Upper state: " << std::endl;
//    upperState->printContents();
//    std::cout << std::endl;
  }

  std::string getName() {
    return "GausElimRuleUpper";
  }
 private:
  // Store state . . .
  htgs::StateContainer<bool> *upperState;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks;

  int gridWidth;
  int gridHeight;
  int blockSize;

};


#endif //HTGS_TUTORIALS_GAUSELIMRULEUPPER_H

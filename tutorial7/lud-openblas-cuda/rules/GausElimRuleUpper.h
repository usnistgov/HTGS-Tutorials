//
// Created by tjb3 on 7/29/16.
//

#ifndef HTGS_TUTORIALS_GAUSELIMRULEUPPER_H
#define HTGS_TUTORIALS_GAUSELIMRULEUPPER_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixFactorData.h"
#include "../../../tutorial-utils/util-matrix.h"
#include "../data/MatrixBlockMultiData.h"
class GausElimRuleUpper : public htgs::IRule<MatrixBlockMultiData<double *>, MatrixFactorData<double *>>
{

 public:

  GausElimRuleUpper(htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks, int gridHeight, int gridWidth) :
      matrixBlocks(matrixBlocks), gridWidth(gridWidth), gridHeight(gridHeight)
  {
    this->upperState = this->allocStateContainer<int>(gridHeight, gridWidth, 0);
  }

  virtual ~GausElimRuleUpper() {
    delete upperState;
  }

  virtual void applyRule(std::shared_ptr<MatrixBlockMultiData<double *>> data, int pipelineId) {
    int row = data->getRequest()->getRow();
    int col = data->getRequest()->getCol();

    // Update the state for the data we received
    int dataState = this->upperState->get(row, col) + 1;

    this->upperState->set(row, col, dataState);

    // Received Diagonal
    if (row == col) {
      if (dataState == row+1) {
        // Check all on same row beyond col
        for (int c = col + 1; c < gridWidth; c++) {
          int stateVal = this->upperState->get(row, c);

          // if the state value is equal to the row value, then it is ready to be sent
          if (stateVal == row) {
            auto matrix = matrixBlocks->get(row, c);
            addResult(new MatrixFactorData<double *>(data->getMatrixBlockData(), matrix));
//            std::cout << "Sending " << row << ", " << col << " with " << row << ", " << c << std::endl;
          }
        }
      }
    }
      // Check only above the diagonal
    else if (col > row)
    {
      // Check if the current data is ready to be factored
      if (dataState == row) {
        // Validate the diagonal has been processed
        int diagonalState = this->upperState->get(row, row);
        if (diagonalState == row+1) {

          // Diagonal and data are ready to be factored
          auto diagonal = matrixBlocks->get(row, row);
          addResult(new MatrixFactorData<double *>(diagonal, data->getMatrixBlockData()));
//          std::cout << "Sending " << row << ", " << row << " with " << row << ", " << col << std::endl;
        }
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
  htgs::StateContainer<int> *upperState;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks;

  int gridWidth;
  int gridHeight;

};


#endif //HTGS_TUTORIALS_GAUSELIMRULEUPPER_H

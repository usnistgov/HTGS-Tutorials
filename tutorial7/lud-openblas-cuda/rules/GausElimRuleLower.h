//
// Created by tjb3 on 7/29/16.
//

#ifndef HTGS_TUTORIALS_GAUSELIMRULELOWER_H
#define HTGS_TUTORIALS_GAUSELIMRULELOWER_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixFactorData.h"
#include "../data/MatrixFactorCudaData.h"
#include "../../../tutorial-utils/util-matrix.h"
class GausElimRuleLower : public htgs::IRule<MatrixBlockMultiData<double *>, MatrixFactorCudaData<double *>>
{

 public:

  GausElimRuleLower(htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks, int gridHeight, int gridWidth) :
      matrixBlocks(matrixBlocks), gridHeight(gridHeight), gridWidth(gridWidth)
  {
    this->lowerState = this->allocStateContainer<int>(gridHeight, gridWidth, 0);
    this->matrixDiagonalBlocks = this->allocStateContainer<std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>>>(gridWidth, nullptr);
  }

  virtual ~GausElimRuleLower() {
    delete lowerState;
  }

  virtual void applyRule(std::shared_ptr<MatrixBlockMultiData<double *>> data, int pipelineId) {
    int row = data->getRequest()->getRow();
    int col = data->getRequest()->getCol();

    // Update the state for the data we received
    int dataState = this->lowerState->get(row, col) + 1;

    this->lowerState->set(row, col, dataState);

    // Received Diagonal
    if (row == col)
    {
      // Store the GPU memory for the diagonal to recover it later
      if (!this->matrixDiagonalBlocks->has(col))
      {
        std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> dataMem = data->getMatrixBlockMemoryData();
        this->matrixDiagonalBlocks->set(col, dataMem);
      }

      // When the diagonal dataState is equal to the col+1, then it is ready to be factored
      if (dataState == col+1)
      {
        for (int r = row+1; r < gridHeight; r++)
        {
          int stateVal = this->lowerState->get(r, col);

          // if the state value is equal to the row value, then it is ready to be sent
          if (stateVal == col)
          {
            auto matrix = matrixBlocks->get(r, col);
            std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> dataMem = data->getMatrixBlockMemoryData();
            addResult(new MatrixFactorCudaData<double *>(dataMem, matrix));
//            std::cout << "Sending " << row << ", " << col << " with " << r << ", " << col << std::endl;
          }
        }
      }

//      // Check all on same row beyond col
//      for (int r = row+1; r < gridHeight; r++)
//      {
//        int stateVal = this->lowerState->get(r, col);
//
//        // if the state value is equal to the row value, then it is ready to be sent
//        if (stateVal == col)
//        {
//          auto matrix = matrixBlocks->get(r, col);
//          addResult(new MatrixFactorData<double *>(data, matrix));
//        }
//      }
    }
      // Check only below the diagonal
    else if (row > col)
    {
//      std::cout << "Received block below diagonal" << std::endl;
      // Check if the current data is ready to be factored
      if (dataState == col) {
        // Validate the diagonal has been processed
        int diagonalState = this->lowerState->get(col, col);
        if (diagonalState == col+1) {

          // Diagonal and data are ready to be factored
          auto diagonal = matrixDiagonalBlocks->get(col);
          addResult(new MatrixFactorCudaData<double *>(diagonal, data->getMatrixBlockData()));
//          std::cout << "Sending " << col << ", " << col << " with " << row << ", " << col << std::endl;


        }
      }
    }

//    std::cout << "Lower state: " << std::endl;
//    lowerState->printContents();
//    std::cout << std::endl;

  }

  std::string getName() {
    return "GausElimRuleLower";
  }

 private:
  // Store state . . .
  htgs::StateContainer<int> *lowerState;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>>> *matrixDiagonalBlocks;

  int gridWidth;
  int gridHeight;
};


#endif //HTGS_TUTORIALS_GAUSELIMRULELOWER_H

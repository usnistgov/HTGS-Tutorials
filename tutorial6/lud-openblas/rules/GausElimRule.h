//
// Created by tjb3 on 7/29/16.
//

#ifndef HTGS_TUTORIALS_GAUSELIMRULE_H
#define HTGS_TUTORIALS_GAUSELIMRULE_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixBlockData.h"
class GausElimRule : public htgs::IRule<MatrixBlockData<double *>, MatrixBlockData<double *>>
{
 public:

  GausElimRule(int totalBlocksDiagonal, int gridHeight, int gridWidth) : totalBlocksDiagonal(totalBlocksDiagonal)
  {
    this->diagonalState = this->allocStateContainer<int>(gridHeight, gridWidth, 0);
  }

  ~GausElimRule() {
    delete diagonalState;
  }

  virtual bool canTerminateRule(size_t pipelineId) override {
    return totalBlocksDiagonal == 0;
  }

  virtual void applyRule(std::shared_ptr<MatrixBlockData<double *>> data, size_t pipelineId) override {

    // if it is the diagonal, then ship it
    int row = data->getRequest()->getRow();
    int col = data->getRequest()->getCol();

//    std::cout << "Received " << row << ", " << col << " totalBlocks = " << totalBlocksDiagonal << std::endl;

    // Send for gaus elim on diagonal blocks
    if (row == col) {

      int state = this->diagonalState->get(row, col) + 1;
      this->diagonalState->set(row, col, state);


      // Check to see if the diagonal entry is ready for gausElim
      if (state == row) {
        addResult(data);
        totalBlocksDiagonal--;
      }

//      std::cout << "Diagonal State" << std::endl;
//      this->diagonalState->printContents();
//      std::cout <<std::endl;

    }

  }

  std::string getName() override {
    return "GausElimRule";
  }

 private:
  int totalBlocksDiagonal;
  htgs::StateContainer<int> *diagonalState;
};

#endif //HTGS_TUTORIALS_GAUSELIMRULE_H

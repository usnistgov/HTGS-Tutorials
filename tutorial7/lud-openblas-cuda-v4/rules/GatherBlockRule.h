//
// Created by tjb3 on 8/12/16.
//

#ifndef HTGS_TUTORIALS_GATHERBLOCKRULE_H
#define HTGS_TUTORIALS_GATHERBLOCKRULE_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixPanelData.h"
class GatherBlockRule : public htgs::IRule<MatrixBlockData<double *>, MatrixPanelData>
{
  enum class GatherState
  {
    None,
    Done
  };


 public:

  GatherBlockRule(int numBlocksHeight, int numBlocksWidth, long blockSize,
                  htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks) :
      numBlocksHeight(numBlocksHeight), blockSize(blockSize), matrixBlocks(matrixBlocks)
  {
    gatherBlockState = this->allocStateContainer<GatherState>(numBlocksHeight, numBlocksWidth, GatherState::None);
  }

  virtual void applyRule(std::shared_ptr<MatrixBlockData<double *>> data, int pipelineId) {

    int row = data->getRequest()->getRow();
    int col = data->getRequest()->getCol();

    gatherBlockState->assign(row, col, GatherState::Done);

    bool sendPanel = true;

    for (int rowCheck = col+1; rowCheck < numBlocksHeight; rowCheck++)
    {
      // Check to see if all blocks in the column have been factored
      if (!gatherBlockState->has(rowCheck, col))
      {
        sendPanel = false;
      }
    }

    if (sendPanel)
    {
      int diagLocation = col;
      int startingRow = diagLocation;
      long height = blockSize * (numBlocksHeight - startingRow);

      // The factored panel has the diagonal generated
      // The memory is located below the diagonal
      MatrixPanelData *panel = new MatrixPanelData(height, blockSize, col, col, PanelState::ALL_FACTORED);
      panel->setMemory(matrixBlocks->get(startingRow, col)->getMatrixData());
      addResult(panel);
    }

//    std::cout << "Gather state" << std::endl;
//    gatherBlockState->printState();
//    std::cout << std::endl;


  }
  std::string getName() {
    return "GatherBlockRule";
  }

 private:
  htgs::StateContainer<GatherState> *gatherBlockState;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<double *>>> *matrixBlocks;

  int numBlocksHeight;
  long blockSize;

};

#endif //HTGS_TUTORIALS_GATHERBLOCKRULE_H


// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXLOADRULE_H
#define HTGS_MATRIXLOADRULE_H
#include <htgs/api/IRule.hpp>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixBlockMulData.h"

class MatrixLoadRule : public htgs::IRule<MatrixBlockData<MatrixMemoryData_t>, MatrixBlockMulData> {

 public:
  MatrixLoadRule(int blockWidth, int blockHeight) {
    this->blockWidth = blockWidth;
    this->blockHeight = blockHeight;

    this->matrixAState = this->allocStateContainer(blockHeight, blockWidth);
    this->matrixBState = this->allocStateContainer(blockHeight, blockWidth);
  }

  ~MatrixLoadRule() {
    delete matrixAState;
    delete matrixBState;
  }

  bool isRuleTerminated(int pipelineId) {
    return false;
  }

  void shutdownRule(int pipelineId) {}

  void applyRule(std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> data, int pipelineId) {
    std::shared_ptr<MatrixRequestData> request = data->getRequest();

    switch (request->getType()) {
      case MatrixType::MatrixA:
        this->matrixAState->set(request->getRow(), request->getCol(), data);

        if (this->matrixBState->has(request->getRow(), request->getCol())) {
          addResult(new MatrixBlockMulData(data, this->matrixBState->get(request->getRow(), request->getCol())));
        }
        break;
      case MatrixType::MatrixB:
        this->matrixBState->set(request->getRow(), request->getCol(), data);

        if (this->matrixAState->has(request->getRow(), request->getCol())) {
          addResult(new MatrixBlockMulData(this->matrixAState->get(request->getRow(), request->getCol()), data));
        }
        break;
      case MatrixType::MatrixC:
        break;
    }
  }

  std::string getName() {
    return "MatrixLoadRule";
  }

 private:
  int blockWidth;
  int blockHeight;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>>> *matrixAState;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>>> *matrixBState;
};
#endif //HTGS_MATRIXLOADRULE_H

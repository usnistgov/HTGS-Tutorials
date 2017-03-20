
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXLOADRULE_H
#define HTGS_MATRIXLOADRULE_H
#include <htgs/api/IRule.hpp>
#include "../../tutorial-utils/matrix-library/data/MatrixBlockData.h"
#include "../../tutorial-utils/matrix-library/data/MatrixBlockMulData.h"
#include "../../tutorial-utils/matrix-library/data/MatrixRequestData.h"

template <class Type>
class HadamardLoadRule : public htgs::IRule<MatrixBlockData<Type>, MatrixBlockMulData<Type>> {

 public:
  HadamardLoadRule(size_t blockWidth, size_t blockHeight) {
    this->matrixAState = this->allocStateContainer(blockHeight, blockWidth);
    this->matrixBState = this->allocStateContainer(blockHeight, blockWidth);
  }

  ~HadamardLoadRule() override {
    delete matrixAState;
    delete matrixBState;
  }

  void applyRule(std::shared_ptr<MatrixBlockData<Type>> data, size_t pipelineId) override {
    std::shared_ptr<MatrixRequestData> request = data->getRequest();

    switch (request->getType()) {
      case MatrixType::MatrixA:
        this->matrixAState->set(request->getRow(), request->getCol(), data);

        if (this->matrixBState->has(request->getRow(), request->getCol())) {
          this->addResult(new MatrixBlockMulData<Type>(data, this->matrixBState->get(request->getRow(), request->getCol()), nullptr));
        }
        break;
      case MatrixType::MatrixB:
        this->matrixBState->set(request->getRow(), request->getCol(), data);

        if (this->matrixAState->has(request->getRow(), request->getCol())) {
          this->addResult(new MatrixBlockMulData<Type>(this->matrixAState->get(request->getRow(), request->getCol()), data, nullptr));
        }
        break;
      case MatrixType::MatrixC:
        break;
    }
  }

  std::string getName() override {
    return "HadamardLoadRule";
  }

 private:
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<Type>>> *matrixAState;
  htgs::StateContainer<std::shared_ptr<MatrixBlockData<Type>>> *matrixBState;
};
#endif //HTGS_MATRIXLOADRULE_H

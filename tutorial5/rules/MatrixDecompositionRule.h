
// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.
// NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
// You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

//
// Created by tjb3 on 6/21/16.
//

#ifndef HTGS_TUTORIALS_MATRIXDECOMPOSITIONRULE_H
#define HTGS_TUTORIALS_MATRIXDECOMPOSITIONRULE_H

class MatrixDecompositionRule : public htgs::IRule<MatrixRequestData, MatrixRequestData> {
 public:
  MatrixDecompositionRule(size_t numGpus) : numGpus(numGpus) {}

  virtual void applyRule(std::shared_ptr<MatrixRequestData> data, size_t pipelineId) {

    MatrixType type = data->getType();
    int row = data->getRow();
    int col = data->getCol();
    size_t gpuId;
    switch (type) {
      case MatrixType::MatrixA:
        // One column per GPU
        gpuId = col % numGpus;

        if (pipelineId == gpuId) {
          addResult(data);
//          std::cout << "sending matrixA: " << row << ", " << col << " to gpu: " << pipelineId << std::endl;
        }

        break;
      case MatrixType::MatrixB:
        // One row per GPU
        gpuId = row % numGpus;

        if (pipelineId == gpuId) {
          addResult(data);
//          std::cout << "sending matrixB: " << row << ", " << col << " to gpu: " << pipelineId << std::endl;
        }
        break;

      case MatrixType::MatrixC:break;
      case MatrixType::MatrixAny:break;
    }

  }
  virtual std::string getName() {
    return "MatrixDecompositionRule";
  }

 private:
  size_t numGpus;
};

#endif //HTGS_TUTORIALS_MATRIXDECOMPOSITIONRULE_H

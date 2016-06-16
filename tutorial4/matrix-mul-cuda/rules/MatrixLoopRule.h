//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXLOOPRULE_H
#define HTGS_MATRIXLOOPRULE_H

#include <htgs/api/IRule.hpp>
#include "../data/MatrixBlockData.h"
#include "../data/MatrixBlockMulData.h"

class MatrixLoopRule : public htgs::IRule<MatrixBlockData<MatrixMemoryData_t>, MatrixRequestData> {

public:
    MatrixLoopRule(int loopIterations) {
        this->loopIterations = loopIterations;
        firstRun = false;
    }

    ~MatrixLoopRule() {
    }

    bool isRuleTerminated(int pipelineId) {
        return loopIterations == 0 && firstRun;
    }

    void shutdownRule(int pipelineId) {
    }

    void applyRule(std::shared_ptr<MatrixBlockData<MatrixMemoryData_t>> data, int pipelineId) {
        firstRun = true;

        if (this->loopIterations > 0) {
            addResult(data->getRequest());
            this->loopIterations--;
        }
    }

    std::string getName() {
        return "MatrixLoopRule";
    }

private:
    int loopIterations;
  bool firstRun;
};

#endif //HTGS_MATRIXLOOPRULE_H

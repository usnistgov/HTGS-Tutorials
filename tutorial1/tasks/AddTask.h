//
// Created by tjb3 on 2/19/16.
//
#ifndef HTGS_ADDTASK_H
#define HTGS_ADDTASK_H

#include "../data/InputData.h"
#include "../data/OutputData.h"

#include <htgs/api/ITask.hpp>

class AddTask : public htgs::ITask<InputData, OutputData>
{

public:
    AddTask() { }

    virtual ~AddTask() { }

    virtual void initialize(int pipelineId, int numPipeline) { }

    virtual void shutdown() { }

    virtual void executeTask(std::shared_ptr<InputData> data) {
        // Adds x + y
        int sum = data->getX() + data->getY();

        // Sends data along output edge
        this->addResult(new OutputData(sum));
    }

    virtual std::string getName() {
      return "x+y=z";
    }


    virtual AddTask *copy() {
        return new AddTask();
    }

    virtual bool isTerminated(std::shared_ptr<htgs::BaseConnector> inputConnector) {
        return inputConnector->isInputTerminated();
    }
};

#endif //HTGS_ADDTASK_H

//
// Created by tjb3 on 12/3/15.
//

#ifndef HTGS_CCFTASK_H
#define HTGS_CCFTASK_H


#include <htgs/api/VoidData.hpp>
#include <htgs/api/ITask.hpp>
#include "../data/CCFData.h"

class CCFTask: public htgs::ITask<CCFData, htgs::VoidData> {


 public:
  CCFTask(size_t numThreads) : ITask(numThreads) { }


  ~CCFTask() { }

  virtual void executeTask(std::shared_ptr<CCFData> data) override;

  virtual std::string getName() override;

  virtual htgs::ITask<CCFData, htgs::VoidData> *copy() override;

 private:
  std::list<ImageStitching::CorrelationTriple> multiCcfs;
};


#endif //HTGS_CCFTASK_H

//
// Created by tjb3 on 11/17/15.
//

#ifndef HTGS_READTASK_H
#define HTGS_READTASK_H


#include <fftw-image-tile.h>
#include <tile-grid-traverser.hpp>
#include <htgs/api/ITask.hpp>
#include "../data/FFTData.h"

class ReadTask: public htgs::ITask<FFTData, FFTData> {

 public:
  ReadTask(int startCol, int startRow, int extentWidth, int extentHeight) : ITask() {
    this->startCol = startCol;
    this->startRow = startRow;
    this->extentWidth = extentWidth;
    this->extentHeight = extentHeight;
  }

  ~ReadTask() { }

  virtual void executeTask(std::shared_ptr<FFTData> data) override;

  virtual std::string getName() override;

  virtual htgs::ITask<FFTData, FFTData> *copy() override;

 private:
  int startCol;
  int startRow;
  int extentWidth;
  int extentHeight;
  int pipelineId;
};


#endif //HTGS_READTASK_H

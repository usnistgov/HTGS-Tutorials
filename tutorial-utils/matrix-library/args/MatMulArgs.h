//
// Created by tjb3 on 3/21/17.
//

#ifndef HTGS_TUTORIALS_MATMULARGS_H
#define HTGS_TUTORIALS_MATMULARGS_H

#include <cstddef>
#include <string>
#include <iostream>
#include <sstream>
class MatMulArgs {
 public:

  MatMulArgs() {
    matrixAHeight = 1024;
    matrixBWidth = 1024;
    sharedDim = 1024;

    blockSize = 512;
    numReadThreads = 1;
    numMatMulThreads = 10;
    directory = "data";
    numGPUs = 1;
    gpuIds = "0";
    outputDir = directory;
    runSequential = false;
    validateResults = false;
  }

  size_t getMatrixAHeight() const {
    return matrixAHeight;
  }
  size_t getMatrixBWidth() const {
    return matrixBWidth;
  }
  size_t getSharedDim() const {
    return sharedDim;
  }
  size_t getBlockSize() const {
    return blockSize;
  }
  size_t getNumReadThreads() const {
    return numReadThreads;
  }
  size_t getNumMatMulThreads() const {
    return numMatMulThreads;
  }
  const std::string &getDirectory() const {
    return directory;
  }
  const std::string &getOutputDir() const {
    return outputDir;
  }
  bool isRunSequential() const {
    return runSequential;
  }
  bool isValidateResults() const {
    return validateResults;
  }
  size_t getNumGPUs() const {
    return numGPUs;
  }
  const std::string &getGpuIds() const {
    return gpuIds;
  }

  void copyGpuIds(int *idArr)
  {
    std::istringstream iss(gpuIds);
    std::string token;
    size_t count = 0;
    while(std::getline(iss, token, ','))
    {
      if (count >= numGPUs)
      {
        std::cerr << "Num GPUs specified = " << numGPUs << " too many GPU ids specified: " << gpuIds << std::endl;
        exit(1);
      }

      int gpuId = std::stoi(token);
      idArr[count] = gpuId;

      count++;
    }
  }

  void processArgs(int argc, char** argv);


 private:
  size_t matrixAHeight;
  size_t matrixBWidth;
  size_t sharedDim;
  size_t blockSize;
  size_t numReadThreads;
  size_t numMatMulThreads;
  std::string directory;
  std::string outputDir;

  size_t numGPUs;
  std::string gpuIds;

  bool runSequential;
  bool validateResults;
};

#endif //HTGS_TUTORIALS_MATMULARGS_H

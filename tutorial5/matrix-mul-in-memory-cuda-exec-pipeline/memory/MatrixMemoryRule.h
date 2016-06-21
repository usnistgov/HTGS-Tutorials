//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXMEMORYRULE_H
#define HTGS_MATRIXMEMORYRULE_H
#include <htgs/api/IMemoryReleaseRule.hpp>

class MatrixMemoryRule: public htgs::IMemoryReleaseRule {
 public:

  MatrixMemoryRule(int releaseCount) : releaseCount(releaseCount) {
  }

  void memoryUsed() {
    releaseCount--;
  }

  bool canReleaseMemory() {
    return releaseCount == 0;
  }

 private:
  int releaseCount;
};
#endif //HTGS_MATRIXMEMORYRULE_H

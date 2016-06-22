//
// Created by tjb3 on 2/19/16.
//

#ifndef HTGS_INPUTDATA_H
#define HTGS_INPUTDATA_H

#include <htgs/api/IData.hpp>

class InputData : public htgs::IData {
 public:
  InputData(int x, int y) : x(x), y(y) {}

  int getX() const { return x; }
  int getY() const { return y; }

 private:
  int x;
  int y;

};

#endif //HTGS_INPUTDATA_H

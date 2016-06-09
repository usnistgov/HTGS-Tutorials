//
// Created by tjb3 on 2/23/16.
//

#ifndef HTGS_MATRIXREQUESTDATA_H
#define HTGS_MATRIXREQUESTDATA_H

#include <htgs/api/IData.hpp>
#include "../../../tutorial-utils/enums/MatrixType.h"

class MatrixRequestData : public htgs::IData {
 public:
  MatrixRequestData(int row, int col, MatrixType type) : row(row), col(col), type(type) { }

  int getRow() const {
    return row;
  }
  int getCol() const {
    return col;
  }
  MatrixType getType() const {
    return type;
  }

 private:
  int row;
  int col;
  MatrixType type;
};

#endif //HTGS_MATRIXREQUESTDATA_H

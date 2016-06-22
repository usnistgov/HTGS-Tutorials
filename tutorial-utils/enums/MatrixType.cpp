//
// Created by tjb3 on 6/8/16.
//

#include <string>
#include "MatrixType.h"

std::string matrixTypeToString(MatrixType type) {
  switch (type) {
    case MatrixType::MatrixA: return "MatrixA";
    case MatrixType::MatrixB: return "MatrixB";
    case MatrixType::MatrixC: return "MatrixC";
  }
  return "";
}
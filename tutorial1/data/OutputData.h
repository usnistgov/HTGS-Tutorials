//
// Created by tjb3 on 2/19/16.
//

#ifndef HTGS_OUTPUTDATA_H
#define HTGS_OUTPUTDATA_H

#include <htgs/api/IData.hpp>

class OutputData : public htgs::IData
{
public:
  OutputData(int result) : IData(result),  result(result) {}

  int getResult() const { return result; }

 private:
  int result;
};

#endif //HTGS_OUTPUTDATA_H

//
// Created by tjb3 on 11/16/15.
//


#ifndef HTGS_FFTMEMORYRULE_H
#define HTGS_FFTMEMORYRULE_H

#include <fftw-image-tile.h>
#include <tile-grid.hpp>
#include <htgs/api/IMemoryReleaseRule.hpp>

class FFTMemoryRule: public htgs::IMemoryReleaseRule {
 public:
  FFTMemoryRule(ImageStitching::ImageTile *tile, TileGrid<ImageStitching::ImageTile> *grid) {
    this->releaseCount = getReleaseCount(grid->getExtentWidth(),
                                         grid->getExtentHeight(),
                                         grid->getStartRow(),
                                         grid->getStartCol(),
                                         tile->getRowIdx(),
                                         tile->getColIdx());
  }

  FFTMemoryRule(ImageStitching::ImageTile *tile, int extendWidth, int extentHeight, int startRow, int startCol) {
    this->releaseCount =
        getReleaseCount(extendWidth, extentHeight, startRow, startCol, tile->getRowIdx(), tile->getColIdx());
  }

  ~FFTMemoryRule() {

  }

  void memoryUsed() { this->releaseCount--; }
  bool canReleaseMemory() { return this->releaseCount == 0; }

 private:
  int getReleaseCount(int gridWidth, int gridHeight, int startRow, int startCol, int row, int col) {
    return 2 + (col - startCol < gridWidth - 1 && col > startCol ? 1 : 0)
        + (row - startRow < gridHeight - 1 && row > startRow ? 1 : 0);
  }

  int releaseCount;
};

#endif //HTGS_FFTMEMORYRULE_H

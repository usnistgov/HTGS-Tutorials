//
// Created by tjb3 on 1/22/16.
//

#ifndef CMAKE_IMAGE_STITCHING_STITCHINGPARAMS_H
#define CMAKE_IMAGE_STITCHING_STITCHINGPARAMS_H

#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <tclap/ValuesConstraint.h>
#include <tclap/CmdLine.h>
#include <iterator>
#include <sstream>
#include <tile-grid.hpp>
#include "ParamValidators.h"

enum class FftwMode {
    Measure,
    Exhaustive,
    Patient,
    Estimate
};

std::string getFftwModeString(FftwMode mode);
int getFftwMode(FftwMode mode);
FftwMode convertFftwModeString(std::string mode);

class StitchingParams {



public:

    StitchingParams(int argc, char **argv) {
        std::string argList;
        bool fileFound = false;
        for (int arg = 0; arg < argc; arg++) {
            if (argv[arg] == "--" + INPUT_PARAM_FILE && arg+1 < argc) {
                this->args = loadArgs(std::string(argv[arg+1]), argc, argv);
                arg++;
                fileFound = true;
                break;
            }
            else {
                argList += std::string(argv[arg]) + " ";
            }
        }

        if (!fileFound) {
            std::istringstream iss(argList);
            std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(),
                      std::back_inserter(this->args));
        }


    }

    const std::string &getOutputParamFile() const {
        return outputParamFile;
    }

    bool isSaveParams() const {
        return saveParams;
    }

    int getGridWidth() const {
        return gridWidth;
    }

    int getGridHeight() const {
        return gridHeight;
    }

    int getStartTile() const {
        return startTile;
    }

    int getStartRow() const {
        return startRow;
    }

    int getStartCol() const {
        return startCol;
    }

    int getExtentWidth() const {
        return extentWidth;
    }

    int getExtentHeight() const {
        return extentHeight;
    }

    const GridOrigin &getOrigin() const {
        return origin;
    }

    const GridNumbering &getNumbering() const {
        return numbering;
    }

    const string &getPlanFile() const {
        return planFile;
    }

    const FftwMode &getFftwMode() const {
        return fftwMode;
    }

    bool isSavePlan() const {
        return savePlan;
    }

    bool isLoadPlan() const {
        return loadPlan;
    }

    int getNumThreadsFFT() const {
        return numThreadsFFT;
    }

  int getNumThreadsPCIAM() const {
      return numThreadsPCIAM;
  }


    bool isUseGpu() const {
        return useGpu;
    }

    const vector<int> &getGpuIds() const {
        return gpuIds;
    }

    const string &getImageDir() const {
        return imageDir;
    }

    const string &getFilenamePattern() const {
        return filenamePattern;
    }


    const string &getOutputDir() const {
        return outputDir;
    }

    const string &getOutputFilePrefix() const {
        return outputFilePrefix;
    }

    int getStageRepeatability() const {
        return stageRepeatability;
    }

    double getHorizontalOverlap() const {
        return horizontalOverlap;
    }

    double getVerticalOverlap() const {
        return verticalOverlap;
    }

    double getOverlapUncertainty() const {
        return overlapUncertainty;
    }

  int getMemoryPoolSize() const {
    return memoryPoolSize;
  }

  void printArgs()
    {
        std::cout << "Program arguments:" << std::endl <<
                GRID_WIDTH << " = " << this->gridWidth << std::endl <<
                GRID_HEIGHT << " = " << this->gridHeight << std::endl <<
                START_TILE << " = " << this->startTile << std::endl <<
                START_ROW << " = " << this->startRow << std::endl <<
                START_COL << " = " << this->startCol << std::endl <<
                EXTENT_WIDTH << " = " << this->extentWidth << std::endl <<
                EXTENT_HEIGHT <<" = " << this->extentHeight << std::endl <<
                IMAGE_DIR << " = " << this->imageDir << std::endl <<
                FILENAME_PATTERN <<" = " << this->filenamePattern << std::endl <<
                GRID_ORIGIN << " = " << TileGrid<void>::getOriginString(origin) << std::endl <<
                NUMBERING_PATTERN << " = " << TileGrid<void>::getGridNumberingString(numbering) << std::endl <<
                FFTW_MODE << " = " << getFftwModeString(this->fftwMode) << std::endl <<
                PLAN_FILE << " = " << this->planFile << std::endl <<
                LOAD_PLAN << " = " << (this->loadPlan ? "true" : "false") << std::endl <<
                SAVE_PLAN << " = " << (this->savePlan ? "true" : "false") << std::endl <<
                NUM_THREADS_FFT << " = " << this->numThreadsFFT << std::endl <<
                NUM_THREADS_PCIAM << " = " << this->numThreadsPCIAM << std::endl <<
                OUTPUT_PARAM_FILE << " = " << this->outputParamFile << std::endl <<
                SAVE_PARAMS << " = " << (this->saveParams ? "true" : "false") << std::endl <<
                OUTPUT_DIR << " = " << this->outputDir << std::endl <<
                OUTPUT_FILE_PREFIX << " = " << this->outputFilePrefix << std::endl <<
                STAGE_REPEATABILITY << " = " << this->stageRepeatability << std::endl <<
                HORIZONTAL_OVERLAP << " = " << this->horizontalOverlap << std::endl <<
                VERTICAL_OVERLAP << " = " << this->verticalOverlap << std::endl <<
                OVERLAP_UNCERTAINTY << " = " << this->overlapUncertainty << std::endl <<
                MEMORY_POOL_SIZE << " = " << this->memoryPoolSize << std::endl;


#ifdef USE_CUDA
        std::cout <<USE_GPU << " = " << (this->useGpu ? "true" : "false") << std::endl;

        if (gpuIds.size() > 0)
        {
            for (int v : gpuIds)
            {
                std::cout<< GPU_ID << " = " << v << std::endl;
            }
        }
#endif

    }

    ~StitchingParams()
    { }

    int parseArgs();
    int saveArgs(std::string fileName);
    std::vector<std::string> loadArgs(std::string fileName, int argc, char **argv);

private:
    template <typename T>
    std::string getOutputValueString(std::string option, T value, T defVal)
    {
        if (value == defVal)
            return "";

        std::stringstream ss;
        ss << option << " = " << value << std::endl;

        return ss.str();
    }

    std::string outputParamFile;
    std::string inputParamFile;
    bool saveParams;

    int gridWidth;
    int gridHeight;
    int startTile;

    int startRow;
    int startCol;
    int extentWidth;
    int extentHeight;

    GridOrigin origin;
    GridNumbering numbering;

    FftwMode fftwMode;
    std::string planFile;
    bool savePlan;
    bool loadPlan;

    int numThreadsFFT;
    int numThreadsPCIAM;

    bool useGpu;
    std::vector<int> gpuIds;

    std::string imageDir;
    std::string filenamePattern;

    std::string outputDir;
    std::string outputFilePrefix;

    int stageRepeatability;
    double horizontalOverlap;
    double verticalOverlap;
    double overlapUncertainty;

  int memoryPoolSize;



    std::vector<std::string> args;

    static std::string GRID_WIDTH;
    static std::string GRID_HEIGHT;
    static std::string START_TILE;
    static std::string IMAGE_DIR;
    static std::string FILENAME_PATTERN;
    static std::string GRID_ORIGIN;
    static std::string NUMBERING_PATTERN;
    static std::string START_ROW;
    static std::string START_COL;
    static std::string EXTENT_WIDTH;
    static std::string EXTENT_HEIGHT;

    static std::string FFTW_MODE;
    static std::string PLAN_FILE;
    static std::string SAVE_PLAN;
    static std::string LOAD_PLAN;

    static std::string NUM_THREADS_FFT;
    static std::string NUM_THREADS_PCIAM;

    static std::string USE_GPU;
    static std::string GPU_ID;

    static std::string INPUT_PARAM_FILE;
    static std::string OUTPUT_PARAM_FILE;
    static std::string SAVE_PARAMS;

    static std::string OUTPUT_DIR;
    static std::string OUTPUT_FILE_PREFIX;

    static std::string STAGE_REPEATABILITY;
    static std::string HORIZONTAL_OVERLAP;
    static std::string VERTICAL_OVERLAP;
    static std::string OVERLAP_UNCERTAINTY;

  static std::string MEMORY_POOL_SIZE;
};


#endif //CMAKE_IMAGE_STITCHING_STITCHINGPARAMS_H

//
// Created by tjb3 on 1/22/16.
//

#include "StitchingParams.h"
#include "util-filesystem.h"
#include <tclap/CmdLine.h>
#include <fstream>
#include <thread>

std::string StitchingParams::GRID_WIDTH = "grid-width";
std::string StitchingParams::GRID_HEIGHT = "grid-height";
std::string StitchingParams::START_TILE = "start-tile";
std::string StitchingParams::IMAGE_DIR = "image-dir";
std::string StitchingParams::FILENAME_PATTERN = "filename-pattern";
std::string StitchingParams::GRID_ORIGIN = "grid-origin";
std::string StitchingParams::NUMBERING_PATTERN = "numbering-pattern";
std::string StitchingParams::START_ROW = "start-row";
std::string StitchingParams::START_COL = "start-col";
std::string StitchingParams::EXTENT_WIDTH = "extent-width";
std::string StitchingParams::EXTENT_HEIGHT = "extent-height";


std::string StitchingParams::FFTW_MODE = "fftw-plan-type";
std::string StitchingParams::PLAN_FILE = "fftw-plan-file";
std::string StitchingParams::SAVE_PLAN = "fftw-plan-save";
std::string StitchingParams::LOAD_PLAN = "fftw-plan-load";
std::string StitchingParams::NUM_THREADS_FFT = "num-threads-fft";
std::string StitchingParams::NUM_THREADS_PCIAM = "num-threads-pciam";

std::string StitchingParams::USE_GPU = "use-gpu";
std::string StitchingParams::GPU_ID = "gpu-id";

std::string StitchingParams::INPUT_PARAM_FILE = "input-param-file";
std::string StitchingParams::OUTPUT_PARAM_FILE = "output-param-file";
std::string StitchingParams::SAVE_PARAMS = "save-params";

std::string StitchingParams::OUTPUT_DIR = "output-dir";
std::string StitchingParams::OUTPUT_FILE_PREFIX = "out-file-prefix";

std::string StitchingParams::STAGE_REPEATABILITY = "stage-repeatability";
std::string StitchingParams::HORIZONTAL_OVERLAP = "horizontal-overlap";
std::string StitchingParams::VERTICAL_OVERLAP = "vertical-overlap";
std::string StitchingParams::OVERLAP_UNCERTAINTY = "overlap-uncertainty";
std::string StitchingParams::MEMORY_POOL_SIZE = "memory-pool-size";

int StitchingParams::parseArgs() {
    try {
        // Create range all values greater than 0
        IntRange range(1);

        // Creates range all values 0 or greater
        IntRange rangeZero(0);

        DoubleRange percRange(0.0, 100.0);

        // Checks if directory exists
        DirectoryValidator dirValid;

        // Checks for a valid file pattern based on 'p'
        FileNamePatternValidator fileNamePatternValidator('p');

        // Numbering accepted values
        std::vector<std::string> numberingAllowed;
        for (int numbering = (int) GridNumbering::Column;
             numbering <= (int) GridNumbering::ColumnChained;
             numbering++) {
            GridNumbering numb = static_cast<GridNumbering>(numbering);
            numberingAllowed.push_back(TileGrid<void>::getGridNumberingString(numb));
        }
        TCLAP::ValuesConstraint<std::string> allowedNumbering(numberingAllowed);

        // Origin accepted values
        std::vector<std::string> originAllowed;
        for (int origin = (int) GridOrigin::UpperLeft;
             origin <= (int) GridOrigin::LowerRight;
             origin++) {
            GridOrigin org = static_cast<GridOrigin>(origin);
            originAllowed.push_back(TileGrid<void>::getOriginString(org));
        }
        TCLAP::ValuesConstraint<std::string>allowedOrigin(originAllowed);

        // FFTW Plan mode values
        std::vector<std::string> fftwModeAllowed;
        for (int mode = (int) FftwMode::Measure;
                mode <= (int) FftwMode::Estimate;
                mode++) {
            FftwMode fftwMode = static_cast<FftwMode>(mode);
            fftwModeAllowed.push_back(getFftwModeString(fftwMode));
        }
        TCLAP::ValuesConstraint<std::string>allowedFftwMode(fftwModeAllowed);

        TCLAP::CmdLine cmd("Image Stitching", ' ', "1.0");

        TCLAP::ValueArg<int> gridWidthArg("", GRID_WIDTH, "Width of the grid", true, 4, &range, cmd);
        TCLAP::ValueArg<int> gridHeightArg("", GRID_HEIGHT, "Height of the grid", true, 4, &range, cmd);
        TCLAP::ValueArg<int> startTileArg("", START_TILE, "Start tile number", true, 0, "int", cmd);
        TCLAP::ValueArg<std::string> imageDirArg("", IMAGE_DIR, "The image directory", true, ".", &dirValid, cmd);
        TCLAP::ValueArg<std::string> filenamePatternArg("", FILENAME_PATTERN, "The filename pattern", true, "{pppp}.tif", &fileNamePatternValidator, cmd);
        TCLAP::ValueArg<std::string> gridNumberingArg("", NUMBERING_PATTERN, "The numbering pattern for the grid", true, "Row", &allowedNumbering, cmd);
        TCLAP::ValueArg<std::string> gridOriginArg("", GRID_ORIGIN, "The origin of the numbering grid", true, "UpperLeft", &allowedOrigin, cmd);
        TCLAP::ValueArg<int> startRowArg("", START_ROW, "The starting row for the sub-grid", false, 0, &rangeZero, cmd);
        TCLAP::ValueArg<int> startColArg("", START_COL, "The starting column for the sub-grid", false, 0, &rangeZero, cmd);
        TCLAP::ValueArg<int> extentWidthArg("", EXTENT_WIDTH, "The width for the sub-grid", false, 4, &range, cmd);
        TCLAP::ValueArg<int> extentHeightArg("", EXTENT_HEIGHT, "The height for the sub-grid", false, 4, &range, cmd);

        TCLAP::ValueArg<std::string> fftwPlanTypeArg("", FFTW_MODE, "The FFTW planning mode type", false, "Measure", &allowedFftwMode, cmd);
        TCLAP::ValueArg<std::string> fftwPlanFileArg("", PLAN_FILE, "The FFTW planning file to load/save (default = plan.dat)", false, "plan.dat", "string", cmd);
        TCLAP::SwitchArg fftwSavePlanArg("", SAVE_PLAN, "If specified, will attempt to save the plan to file", cmd, false);
        TCLAP::SwitchArg fftwLoadPlanArg("", LOAD_PLAN, "If specified, will attempt to load the plan from file", cmd, false);

        TCLAP::ValueArg<int> numThreadsFFTArg("", NUM_THREADS_FFT, "The number of threads to use for FFT execution", false, (std::thread::hardware_concurrency()/2), &range, cmd);
        TCLAP::ValueArg<int> numThreadsPCIAMArg("", NUM_THREADS_PCIAM, "The number of threads to use for PCIAM execution", false, std::thread::hardware_concurrency(), &range, cmd);
        TCLAP::SwitchArg useGpuArg("", USE_GPU, "If specified, will attempt to execute using GPU(s), must have gpu-id specified (default will attempt to use gpu-id 0)", cmd, false);
        TCLAP::MultiArg<int> gpuIdArg("", GPU_ID, "Specifies a GPU Id, if multiple are specified, stitching will attempt to execute using multiple GPUs", false, &rangeZero, cmd);

        TCLAP::ValueArg<std::string> inputParamFileArg("", INPUT_PARAM_FILE, "Input parameters stored in a file (one per line) in the format: <param> = <value> (<param> for flags)", false, "params.txt", "string", cmd);
        TCLAP::ValueArg<std::string> outputParamFileArg("", OUTPUT_PARAM_FILE, "File to write parameters to file: must specify --save-params to enable", false, "params.txt", "string", cmd);
        TCLAP::SwitchArg saveParamsArg("", SAVE_PARAMS, "If specified, will attempt to save the parameters to file", cmd, false);
        TCLAP::ValueArg<std::string> outputDirectoryArg("", OUTPUT_DIR, "Directory to save output", false, "output", "string", cmd);
        TCLAP::ValueArg<std::string> outputFilePrefixArg("", OUTPUT_FILE_PREFIX, "the prefix used for the output files", false, "out-file", "string", cmd);

        TCLAP::ValueArg<int> repeatabilityArg("", STAGE_REPEATABILITY, "The repeatability of the stage", false, -1, &range, cmd);
        TCLAP::ValueArg<double> horizOverlapArg("", HORIZONTAL_OVERLAP, "The horizontal overlap", false, -1.0, &percRange, cmd);
        TCLAP::ValueArg<double> vertOverlapArg("", VERTICAL_OVERLAP, "The vertical overlap", false, -1.0, &percRange, cmd);
        TCLAP::ValueArg<double> overlapUncertArg("", OVERLAP_UNCERTAINTY, "The overlap uncertainty", false, -1.0, &percRange, cmd);

        TCLAP::ValueArg<int> memoryPoolSizeArg("", MEMORY_POOL_SIZE, "The size of the memory pool", false, (0), &rangeZero, cmd);

        cmd.parse(this->args);

        this->gridWidth = gridWidthArg.getValue();
        this->gridHeight = gridHeightArg.getValue();
        this->startTile = startTileArg.getValue();
        this->imageDir = imageDirArg.getValue();
        this->filenamePattern = filenamePatternArg.getValue();
        this->numbering = TileGrid<void>::convertGridNumberingString(gridNumberingArg.getValue());
        this->origin = TileGrid<void>::convertGridOriginString(gridOriginArg.getValue());
        this->startRow = startRowArg.getValue();
        this->startCol = startColArg.getValue();
        this->extentWidth = extentWidthArg.isSet() ? extentWidthArg.getValue() : this->gridWidth;
        this->extentHeight = extentHeightArg.isSet() ? extentHeightArg.getValue() : this->gridHeight;
        this->fftwMode = convertFftwModeString(fftwPlanTypeArg.getValue());
        this->planFile = fftwPlanFileArg.getValue();
        this->loadPlan = fftwSavePlanArg.getValue();
        this->savePlan = fftwSavePlanArg.getValue();
        this->numThreadsFFT = numThreadsFFTArg.getValue();
        this->numThreadsPCIAM = numThreadsPCIAMArg.getValue();
        this->inputParamFile = inputParamFileArg.getValue();
        this->outputParamFile = outputParamFileArg.getValue();
        this->saveParams = saveParamsArg.getValue();
        this->outputDir = outputDirectoryArg.getValue();
        this->outputFilePrefix = outputFilePrefixArg.getValue();
        this->stageRepeatability = repeatabilityArg.getValue();
        this->horizontalOverlap = horizOverlapArg.getValue();
        this->verticalOverlap = vertOverlapArg.getValue();
        this->overlapUncertainty = overlapUncertArg.getValue();
        this->memoryPoolSize = memoryPoolSizeArg.getValue();

        if (!_dirExists(this->outputDir.c_str()))
            if (_mkdir((char *)this->outputDir.c_str(), S_IRWXU) != 0)
                return -1;


#ifdef USE_CUDA
        this->useGpu = useGpuArg.getValue();
        this->gpuIds = gpuIdArg.getValue();
#else
        this->useGpu = false;
#endif

    } catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << ": " << e.typeDescription() << std::endl;
        return -1;
    }



    return 0;
}



int StitchingParams::saveArgs(std::string fileName) {
    std::cout << "Saving Parameters to file: " << fileName << std::endl;
    std::ofstream outFile(fileName);
    outFile << GRID_WIDTH << " = " << this->gridWidth << std::endl <<
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
    (this->loadPlan ? LOAD_PLAN + "\n" : "") <<
    (this->savePlan ? SAVE_PLAN + "\n" : "") <<
    NUM_THREADS_FFT << " = " << this->numThreadsFFT << std::endl <<
    NUM_THREADS_PCIAM << " = " << this->numThreadsPCIAM << std::endl <<
    OUTPUT_PARAM_FILE << " = " << this->outputParamFile << std::endl <<
    (this->saveParams ? SAVE_PARAMS + "\n" : "") <<
    OUTPUT_DIR << " = " << this->outputDir << std::endl <<
    OUTPUT_FILE_PREFIX << " = " << this->outputFilePrefix << std::endl <<
    this->getOutputValueString(STAGE_REPEATABILITY, this->stageRepeatability, -1) <<
    this->getOutputValueString(VERTICAL_OVERLAP, this->verticalOverlap, -1.0) <<
    this->getOutputValueString(HORIZONTAL_OVERLAP, this->horizontalOverlap, -1.0) <<
    this->getOutputValueString(OVERLAP_UNCERTAINTY, this->overlapUncertainty, -1.0) <<
    MEMORY_POOL_SIZE << " = " << this->memoryPoolSize << std::endl;



#ifdef USE_CUDA
    outFile << (this->useGpu ? USE_GPU + "\n" : "");

    if (gpuIds.size() > 0)
    {
        for (int v : gpuIds)
        {
            outFile << GPU_ID << " = " << v << std::endl;
        }
    }
#endif

    outFile.flush();
    return 0;
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

std::vector<std::string> StitchingParams::loadArgs(std::string fileName, int argc, char **argv) {

    std::cout << " Loading args from file: " << fileName << std::endl;

    std::vector<std::pair<std::string, std::string>> newOptions;
    std::vector<std::string> boolOptions;

    // Append to argc and argv
    std::ifstream inFile(fileName);
    std::string line;
    while (std::getline(inFile, line))
    {
        unsigned long firstChar = line.find_first_not_of(' ');
        if (line.at(firstChar) == '#')
            continue;


        std::vector<std::string> elems = split(line, '=');

        // Parse key = value pair
        if (elems.size() == 2)
        {
            std::string newOption = "--" + elems[0];
            std::string newValue = elems[1];
            newOptions.push_back(std::pair<std::string, std::string>(newOption, newValue));
        }
            // Parse boolean flag value
        else if (elems.size() == 1)
        {
            std::string newOption = "--" + elems[0];
            boolOptions.push_back(newOption);
        }
        else
        {
            std::cerr << "Error parsing line: " << line << " invalid expression, should follow the format 'option = \"value\"'; value should not contain '='" << std::endl;
        }
    }

    std::string argList;
    for (int arg = 0; arg < argc; arg++)
    {
        if (argv[arg] == "--" + INPUT_PARAM_FILE)
        {
            arg++;
        }
        else {
            argList += std::string(argv[arg]) + " ";
        }
    }

    for (std::pair<std::string, std::string> v : newOptions)
    {
        argList += " " + v.first + " " + v.second;
    }

    for (std::string v : boolOptions)
    {
        argList += " " + v;
    }

    std::istringstream iss(argList);
    std::vector<std::string> tokens;
    std::copy(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>(), std::back_inserter(tokens));

//    for (std::string v : tokens)
//        std::cerr << v << std::endl;

    return tokens;
}

std::string getFftwModeString(FftwMode mode)
{
    switch (mode)
    {
        case FftwMode::Measure: return "Measure";
        case FftwMode::Exhaustive: return "Exhaustive";
        case FftwMode::Patient: return "Patient";
        case FftwMode::Estimate: return "Estimate";
    }
}

int getFftwMode(FftwMode mode)
{
    switch (mode)
    {
        case FftwMode::Measure: return FFTW_MEASURE;
        case FftwMode::Exhaustive: return FFTW_EXHAUSTIVE;
        case FftwMode::Patient: return FFTW_PATIENT;
        case FftwMode::Estimate: return FFTW_ESTIMATE;
    }
}

FftwMode convertFftwModeString(std::string mode)
{
    if (mode == "Measure")
        return FftwMode::Measure;
    else if (mode == "Exhaustive")
        return FftwMode::Exhaustive;
    else if (mode == "Patient")
        return FftwMode::Patient;
    else if (mode == "Estimate")
        return FftwMode::Estimate;
    else
        return FftwMode::Measure;
}
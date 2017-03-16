//
// Created by tjb3 on 1/27/16.
//

#ifndef CMAKE_IMAGE_STITCHING_PARAMVALIDATORS_H
#define CMAKE_IMAGE_STITCHING_PARAMVALIDATORS_H


#include <tuple>
#include <tile-grid.hpp>
#include <tclap/Constraint.h>
#include <sys/stat.h>
#include <limits.h>
#include <float.h>

class DoubleRange : public TCLAP::Constraint<double>
{
public:
    DoubleRange(double min) : min(min), max(DBL_MAX) {}
    DoubleRange(double min, double max) : min(min), max(max) {}


private:
    std::string description() const {
        return "double >= " + std::to_string(min) + (max == DBL_MAX ? "" : " and <= " + std::to_string(max));
    }

    std::string shortID() const {
        return "double >= " + std::to_string(min) + (max == DBL_MAX ? "" : " and <= " + std::to_string(max));
    }

    bool check(const double &value) const {
        return value >= min && value <= max;
    }

    double min;
    double max;
};

class IntRange : public TCLAP::Constraint<int>
{

public:
    IntRange(int min) : min(min), max(INT_MAX)
    { }
    IntRange(int min, int max) : min(min), max(max)
    {}

private:


    std::string description() const {
        return "int >= " + std::to_string(min) + (max == INT_MAX ? "" : " and <= " + std::to_string(max));
    }

    std::string shortID() const {
        return "int >= " + std::to_string(min) + (max == INT_MAX ? "" : " and <= " + std::to_string(max));
    }

    bool check(const int &value) const {
        return value >= min && value <= max;
    }
    int min;
    int max;
};

class DirectoryValidator : public TCLAP::Constraint<std::string>
{
public:
    std::string description() const {
        return "Valid image directory";
    }

    std::string shortID() const {
        return "Valid image directory";
    }

    bool check(const std::string &value) const {
        struct stat info;
        if (stat(value.c_str(), &info) != 0)
            return false;
        else if (info.st_mode & S_IFDIR)
            return true;
        else
            return false;
    }
};


class FileNamePatternValidator : public TCLAP::Constraint<std::string>
{
public:
    FileNamePatternValidator(char specialChar) : specialChar(specialChar)
    {

    }

    std::string description() const {
        return "Filename pattern must contain the format {pppp}, where p represents a digit: example F_0001.tif = F_{pppp}.tif";
    }

    std::string shortID() const {
        return "Filename pattern: {" + std::string(1, specialChar) + std::string(1, specialChar) + "}.tif";
    }

    bool check(const std::string &value) const {
        std::tuple<std::string, std::string, int> res = TileGrid<void>::parseFilePattern(specialChar, value);
        int numDigits = std::get<2>(res);
        return numDigits > 0;
    }
private:
    char specialChar;
};


#endif //CMAKE_IMAGE_STITCHING_PARAMVALIDATORS_H

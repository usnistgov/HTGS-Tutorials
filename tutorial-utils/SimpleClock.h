//
// Created by Hobbs on 2/15/2016.
//

#ifndef HTGS_SIMPLECLOCK_H
#define HTGS_SIMPLECLOCK_H

#include <chrono>

enum class TimeVal {
  MILLI,
  NANO,
  SEC,
  MICRO
};

class SimpleClock {

 public:

  SimpleClock() {
    duration = 0;
    count = 0;
  }
  void start() {
    startTime = std::chrono::high_resolution_clock::now();
  }

  void stopAndIncrement() {
    stop();
    incrementDuration();
  }

  void stop() {
    endTime = std::chrono::high_resolution_clock::now();
  }

  void incrementDuration() {
    this->duration = this->duration + std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
    count = count + 1;
  }

  long long int getDuration() {
    return duration;
  }

  long getCount() {
    return count;
  }

  double getAverageTime(TimeVal val) {
    double avg = (double) this->getDuration() / (double) this->getCount();

    switch (val) {
      case TimeVal::MILLI:
        return avg / 1000000;
      case TimeVal::NANO:
        return avg;
      case TimeVal::SEC:
        return avg / 1000000000;
      case TimeVal::MICRO:
        return avg / 1000;
    }

    return 0.0;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> startTime;
  std::chrono::time_point<std::chrono::system_clock> endTime;

  long long int duration;
  long count;
};

#endif //HTGS_SIMPLECLOCK_H

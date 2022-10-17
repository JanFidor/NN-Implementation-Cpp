#include <random>

#pragma once

class RandomValueGenerator {
    double range_start, range_end;

    std::random_device rand_dev;
    typedef std::mt19937 LocalGenerator;
    LocalGenerator generator;

    std::uniform_real_distribution<double> distribution;

    public:
        RandomValueGenerator(double start, double end);
        
        double generate();
};

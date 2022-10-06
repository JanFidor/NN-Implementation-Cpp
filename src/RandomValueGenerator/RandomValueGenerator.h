#include <random>

#ifndef RECIPE_H
#define RECIPE_H

class RandomValueGenerator {
    double range_start, range_end;

    std::random_device rand_dev;
    typedef std::mt19937 LocalGenerator;
    LocalGenerator generator;

    std::uniform_real_distribution<double> distribution;

    public:
        double value;
        double derivative;
        RandomValueGenerator(double start, double end);
        
        double generate();
};

#endif
#include <random>
#include "RandomValueGenerator.h"

RandomValueGenerator::RandomValueGenerator(
    double start, double end
) : range_start(start), range_end(end), generator(rand_dev()), distribution(std::uniform_real_distribution<>(start, end)){}
        
double RandomValueGenerator::generate(){
    return distribution(generator);
}

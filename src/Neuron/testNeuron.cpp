#include <catch2/catch_test_macros.hpp>
#include "Neuron.h"
#include "../ActivationFunction/SigmoidFunction.cpp"
#include "../utils/utils.h"


TEST_CASE("Neuron calculates output and derivative corectly"){
    const SigmoidFunction f;
    std::vector<double> weights;
    Neuron n(f, weights, 0.0);
    n.calculate(0);
    bool condition = aproximatelyEqual(n.getOutput(), 0.5, 0.0001) && aproximatelyEqual(n.getOutputDerivative(), 0.0, 0.0001);
    REQUIRE(condition);
}

TEST_CASE("Neuron adjusts weight corectly"){
    const SigmoidFunction f;
    std::vector<double> weights = {1, 2};
    Neuron n(f, weights, 0.0);
    n.adjustWeight(0, 2);
    bool condition = n.getWeights()[0] == 2;
    REQUIRE(condition);
}

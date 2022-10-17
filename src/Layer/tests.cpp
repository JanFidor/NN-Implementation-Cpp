#include <catch2/catch_test_macros.hpp>
#include "../utils/utils.h"
#include "../ActivationFunction/Linear.cpp"
#include "Layer.h"

TEST_CASE("Layer correctly assigns variables in constructor"){
    Layer layer = Layer(Linear(), { { 1,}});
    bool condition = layer.getSize() == 1;
    REQUIRE(condition);
}

TEST_CASE("Layer correctly sets derivatives"){
    Layer layer = Layer(Linear(), { { 1,}});
    layer.setTotalDerivatives({2});

    std::vector<double> derivatives = layer.getTotalDerivatives();

    bool condition = derivatives.size() == 1 && aproximatelyEqual(derivatives.front(), 2, 0.001);
    REQUIRE(condition);
}

TEST_CASE("Layer correctly updates derivatives for a single neuron"){
    Layer layer = Layer(Linear(), { { 2,}});
    layer.calculateInputs({1});     // update output derivative
    layer.updateTotalDerivatives({2});

    std::vector<double> derivatives = layer.getTotalDerivatives();

    bool condition = derivatives.size() == 1 && aproximatelyEqual(derivatives.front(), 4, 0.001);
    REQUIRE(condition);
}

TEST_CASE("Layer correctly adjusts weights for a single neuron"){
    Layer layer = Layer(Linear(), { { 1,}});
    double input = 2;
    double derivative = 2;
    layer.calculateInputs({input});     // update output derivative

    double alpha = 0.1;
    layer.adjustWeights({derivative}, alpha);

    double delta = input * derivative * alpha;
    double output = layer.getOutputs().front();
    std::vector<double> propagation = layer.forwardPropagation();

    bool condition = aproximatelyEqual(propagation.front(), output * (1 - delta), 0.001);
    REQUIRE(condition);
}


TEST_CASE("Layer correctly updates derivatives for multiple neurons"){
    Layer layer = Layer(Linear(), { {1, 2}, {3, 4}});
    layer.calculateInputs({1, 1});     // update output derivative
    layer.updateTotalDerivatives({2, 1});

    std::vector<double> derivativesActual = layer.getTotalDerivatives();
    std::vector<double> derivativesExpected = {4, 10};

    REQUIRE(derivativesActual.size() == 2);

    for(int i = 0; i < layer.getSize(); i++)
        REQUIRE(derivativesActual[i] == derivativesExpected[i]);
}

TEST_CASE("Layer correctly adjusts weights for multiple neurons"){
    Layer layer = Layer(Linear(), {{1, 2}, {3, 4}});
    layer.calculateInputs({1, 2});     // update output derivative
    
    double alpha = 0.1;
    layer.adjustWeights({2, 3}, alpha);

    std::vector<double> propagationActual = layer.forwardPropagation();
    std::vector<double> propagationExpected = {0.8 + 2.6 * 2 , 1.7 + 3.4 * 2};
    for(int i = 0; i < layer.getSize(); i++)
        REQUIRE(propagationActual[i] == propagationExpected[i]);
        
}

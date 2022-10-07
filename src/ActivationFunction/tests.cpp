#include <catch2/catch_test_macros.hpp>
#include "SigmoidFunction.cpp"

bool aproximatelyEqual(double a, double b, double epsilon) {
    return abs(a - b) < epsilon;
}

TEST_CASE("Sigmoid function correctly calculates output for 0"){
    SigmoidFunction f;
    bool condition = aproximatelyEqual(f.calculateOutput(0), 0.5, 0.0001);
    REQUIRE(condition);
}


TEST_CASE("Sigmoid function correctly calculates output for a big positive number"){
    SigmoidFunction f;
    bool condition = aproximatelyEqual(f.calculateOutput(99999), 1.0, 0.0001);
    REQUIRE(condition);
}

TEST_CASE("Sigmoid function correctly calculates output for a big negative number"){
    SigmoidFunction f;
    bool condition = aproximatelyEqual(f.calculateOutput(-99999), 0.0, 0.0001);
    REQUIRE(condition);
}
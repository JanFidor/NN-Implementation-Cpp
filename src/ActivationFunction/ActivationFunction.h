#pragma once

class ActivationFunction{
    public:
        virtual double calculateOutput(double input) const = 0;
        virtual double calculateDerivative(double input) const = 0;
};
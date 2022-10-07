#pragma once

#include <vector>
#include "../ActivationFunction/ActivationFunction.h"

class Neuron {
    private:
        std::vector<int> weights;
        double output, outputDerivative;

        ActivationFunction& function;
    public:
        double totalDerivative;
        Neuron(ActivationFunction& function, std::vector<int> weights);
        
        double getOutput() const;
        double getOutputDerivative() const;

        void calculate(double input);

};

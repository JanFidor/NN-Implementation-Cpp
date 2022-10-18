#pragma once

#include <vector>
#include "../ActivationFunction/ActivationFunction.h"

class Neuron {
    private:
        std::vector<double> weights;
        double output;      
        double outputDerivative;

        const ActivationFunction& function;
    public:
        double totalDerivative;
        
        // makes incorporating an additional Neuron as a bias trivial
        Neuron(const ActivationFunction& function, std::vector<double> weights, double output = 0);

        const std::vector<double>& getWeights() const;
        
        double getOutput() const;
        double getOutputDerivative() const;

        void calculate(double input);

        void adjustWeight(int weightId, double newValue);
};

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
        Neuron(const ActivationFunction&, std::vector<double>, double);

        const std::vector<double>& getWeights() const;
        
        double getOutput() const;
        double getOutputDerivative() const;

        void calculate(double);

        void adjustWeight(int, double);
};

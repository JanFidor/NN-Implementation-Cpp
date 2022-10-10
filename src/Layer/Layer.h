#pragma once

#include <vector>
#include <memory>
#include "../Neuron/Neuron.h"
#include "../ActivationFunction/ActivationFunction.h"


class Layer {
    private:
        std::vector<Neuron> neurons;
        const ActivationFunction& function;

        Neuron createNeuron(const std::vector<int>&) const;
    public: 
        Layer(const ActivationFunction&, const std::vector<std::vector<int>>&);

        double forwardPropagationForIndex(int) const;
        double totalDerivativeOverNeuronsInput(const Neuron&) const;
        double derivativeOverWeight(const Neuron&, int) const;
        void adjustWeight(int, int, double);

        void calculateInputs(const std::vector<double>&);
        std::vector<double> getOutputs() const;
};
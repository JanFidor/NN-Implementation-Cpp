#pragma once

#include <vector>
#include "../Neuron/Neuron.h"
#include "../ActivationFunction/ActivationFunction.h"


class Layer {
    private:
        int size;
        std::vector<Neuron> neurons;
        const ActivationFunction& function;
        bool hasBiasNeuron;

        double forwardPropagationForIndex(int index) const;

        double calculateTotalDerivative(const Neuron& neuron, const std::vector<double>& derivatives) const;
    public: 
        Layer(
            const ActivationFunction& function,
            const std::vector<std::vector<double>>& neuronsWeights,
            bool hasBiasNeuron = false
        );

        void setTotalDerivatives(const std::vector<double>& derivatives);        
        void updateTotalDerivatives(const std::vector<double>& derivatives);
        void adjustWeights(const std::vector<double>& derivatives, double alpha);

        void calculateInputs(const std::vector<double>& inputs);

        int getSize() const;
        std::vector<double> getOutputs() const;
        std::vector<double> getTotalDerivatives() const;

        std::vector<double> forwardPropagation() const;
};
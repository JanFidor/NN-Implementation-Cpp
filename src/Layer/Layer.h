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

        Neuron createNeuron(const std::vector<double>&) const;
        double forwardPropagationForIndex(int) const;

        double calculateTotalDerivative(const Neuron&, const std::vector<double>&) const;

        void adjustWeight(Neuron&, int, double);
    public: 
        Layer(const ActivationFunction&, const std::vector<std::vector<double>>&, bool=false);

        void setTotalDerivatives(const std::vector<double>&);        
        void updateTotalDerivatives(const std::vector<double>&);
        void adjustWeights(const std::vector<double>&, double);

        void calculateInputs(const std::vector<double>&);

        int getSize() const;
        std::vector<double> getOutputs() const;
        std::vector<double> getTotalDerivatives() const;

        std::vector<double> forwardPropagation() const;
};
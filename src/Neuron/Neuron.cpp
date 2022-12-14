#include <vector>
#include "Neuron.h"



Neuron::Neuron(
    const ActivationFunction& function, std::vector<double> weights, double output
) : function(function), weights(weights), output(output){}

const std::vector<double>& Neuron::getWeights() const{
    return weights;
}

double Neuron::getOutput() const {
    return output;
}

double Neuron::getOutputDerivative() const{
    return outputDerivative;
}

void Neuron::calculate(double input){
    output = function.calculateOutput(input);
    outputDerivative = function.calculateDerivative(input);
    
}

void Neuron::adjustWeight(int index, double newValue){
    weights[index] = newValue;
}
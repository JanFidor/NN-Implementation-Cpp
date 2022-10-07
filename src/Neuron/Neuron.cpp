#include <vector>
#include "Neuron.h"



Neuron::Neuron(ActivationFunction& function, std::vector<int> weights) : function(function), weights(weights){}

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
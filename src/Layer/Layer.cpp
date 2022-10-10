#include "Layer.h"
#include <algorithm>




Layer::Layer(const ActivationFunction& function, const std::vector<std::vector<int>>& neuronsWeights) : function(function){
    for(std::vector<int> weights : neuronsWeights)
        neurons.push_back(createNeuron(weights));
}

Neuron Layer::createNeuron(const std::vector<int>& weights) const{
    return Neuron(function, weights, 1.0);     // neuron can be used as bias
}

double Layer::forwardPropagationForIndex(int index) const{
    double sum = 0;

    for(const Neuron& neuron : neurons)
        sum += neuron.getOutput() * neuron.getWeights()[index];     // TODO 1 method for all weights vs additional for a single weight
    
    return sum;
}

double Layer::totalDerivativeOverNeuronsInput(const Neuron& neuron) const{
    double sum = 0;
    const std::vector<int> weights = neuron.getWeights();

    for(int i = 0; i < neurons.size(); i++)
        sum += weights[i] * neurons[i].totalDerivative;
    
    return sum * neuron.getOutputDerivative(); 
}

double Layer::derivativeOverWeight(const Neuron& neuron, int weightId) const {
    return neuron.getOutput() * neurons[weightId].totalDerivative;
}

void Layer::adjustWeight(int neuronId, int weightId, double newValue){
    neurons[neuronId].adjustWeight(weightId, newValue);
}

void Layer::calculateInputs(const std::vector<double>& inputs){
    for(int i = 0; i < neurons.size(); i++)
        neurons[i].calculate(inputs[i]);
}

std::vector<double> Layer::getOutputs() const{
    std::vector<double> outputs;
    for(int i = 0; i < neurons.size(); i++)
        outputs.emplace_back(neurons[i].getOutput());
    return outputs;
}

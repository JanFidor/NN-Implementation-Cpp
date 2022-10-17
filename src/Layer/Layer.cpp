#include "Layer.h"
#include <algorithm>


Layer::Layer(const ActivationFunction& function, const std::vector<std::vector<double>>& neuronsWeights) : function(function){
    size = neuronsWeights.size();

    for(std::vector<double> weights : neuronsWeights)
        neurons.push_back(createNeuron(weights));
}

Neuron Layer::createNeuron(const std::vector<double>& weights) const{
    return Neuron(function, weights, 1.0);     // neuron can be used as bias
}

std::vector<double> Layer::forwardPropagation() const{
    int layerSize = neurons.front().getWeights().size();
    std::vector<double> outputs;
    for(int i = 0; i < layerSize; i++)
        outputs.push_back(forwardPropagationForIndex(i));
    return outputs;
}

double Layer::forwardPropagationForIndex(int index) const{
    double sum = 0;

    for(const Neuron& neuron : neurons)
        sum += neuron.getOutput() * neuron.getWeights()[index];     // TODO 1 method for all weights vs additional for a single weight
    
    return sum;
}

// TODO add setter for output layer
void Layer::setTotalDerivatives(const std::vector<double>& derivatives){
    for(int i = 0; i < size; i++)
        neurons[i].totalDerivative = derivatives[i];
}

void Layer::updateTotalDerivatives(const std::vector<double>& derivatives){
    for(Neuron& neuron : neurons)
        neuron.totalDerivative = calculateTotalDerivative(neuron, derivatives);
}

void Layer::adjustWeights(const std::vector<double>& derivatives, double alpha){
    for(Neuron& neuron : neurons){
        for(int i = 0; i < neuron.getWeights().size(); i++){
            double derivativeOverWeight = neuron.getOutput() * derivatives[i];
            double adjustedWeight = neuron.getWeights()[i] - derivativeOverWeight * alpha; // TODO
            neuron.adjustWeight(i, adjustedWeight);
        }
    }
}

double Layer::calculateTotalDerivative(const Neuron& neuron, const std::vector<double>& derivatives) const{
    double sum = 0;
    const std::vector<double> weights = neuron.getWeights();
    for(int i = 0; i < neurons.size(); i++)
        sum += derivatives[i] * weights[i];
    
    return sum * neuron.getOutputDerivative(); 
}

double Layer::derivativeOverWeight(const Neuron& neuron, int weightId) const {
    return neuron.getOutput() * neurons[weightId].totalDerivative;
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

std::vector<double> Layer::getTotalDerivatives() const{
    std::vector<double> derivatives;
    for(int i = 0; i < neurons.size(); i++)
        derivatives.emplace_back(neurons[i].totalDerivative);
    return derivatives;
}

int Layer::getSize() const{
    return size;
}

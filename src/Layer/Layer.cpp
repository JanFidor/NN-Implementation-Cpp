#include "Layer.h"
#include <algorithm>


Layer::Layer(
    const ActivationFunction& function,
    const std::vector<std::vector<double>>& neuronsWeights,
    bool hasBiasNeuron
) : function(function), hasBiasNeuron(hasBiasNeuron){
    size = neuronsWeights.size();

    for(std::vector<double> weights : neuronsWeights)
        neurons.push_back(Neuron(function, weights, 1.0));
}

double Layer::forwardPropagationForIndex(int index) const{
    double sum = 0;

    for(const Neuron& neuron : neurons)
        sum += neuron.getOutput() * neuron.getWeights()[index];
    
    return sum;
}

void Layer::setTotalDerivatives(const std::vector<double>& derivatives){
    for(int i = 0; i < derivatives.size(); i++)
        neurons[i].totalDerivative = derivatives[i];
}

void Layer::updateTotalDerivatives(const std::vector<double>& derivatives){
    // calculating totalDerivative for Bias changes nothing
    for(Neuron& neuron : neurons)
        neuron.totalDerivative = calculateTotalDerivative(neuron, derivatives); 
}

double Layer::calculateTotalDerivative(const Neuron& neuron, const std::vector<double>& derivatives) const{
    double sum = 0;
    const std::vector<double> weights = neuron.getWeights();
    for(int i = 0; i < derivatives.size(); i++)
        sum += derivatives[i] * weights[i];
    
    return sum * neuron.getOutputDerivative(); 
}

void Layer::adjustWeights(const std::vector<double>& derivatives, double alpha){
    for(Neuron& neuron : neurons){
        for(int i = 0; i < neuron.getWeights().size(); i++){
            double derivativeOverWeight = neuron.getOutput() * derivatives[i];
            double adjustedWeight = neuron.getWeights()[i] - derivativeOverWeight * alpha;
            neuron.adjustWeight(i, adjustedWeight);
        }
    }
}

void Layer::calculateInputs(const std::vector<double>& inputs){
    for(int i = 0; i < inputs.size(); i++)
        neurons[i].calculate(inputs[i]);
}


int Layer::getSize() const{
    return size;
}

std::vector<double> Layer::getOutputs() const{
    std::vector<double> outputs;
    int d = neurons.size() + (hasBiasNeuron ? -1 : 0);

    for(int i = 0; i < d; i++)
        outputs.emplace_back(neurons[i].getOutput());
    return outputs;
}

std::vector<double> Layer::getTotalDerivatives() const{
    std::vector<double> derivatives;
    int d = neurons.size() + (hasBiasNeuron ? -1 : 0);

    for(int i = 0; i < d; i++)
        derivatives.emplace_back(neurons[i].totalDerivative);
    return derivatives;
}

std::vector<double> Layer::forwardPropagation() const{
    int layerSize = neurons.front().getWeights().size();
    std::vector<double> outputs;
    for(int i = 0; i < layerSize; i++)
        outputs.push_back(forwardPropagationForIndex(i));
    return outputs;
}

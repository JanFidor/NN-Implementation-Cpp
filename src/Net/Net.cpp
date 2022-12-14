#include "Net.h"
#include <algorithm>


Net::Net(
    const LossFunction& lossFunction, 
    const std::vector<LayerStructure>& netStructure, 
    const std::pair<double, double> weightInitialRange,
    double alpha
) : generator(weightInitialRange.first, weightInitialRange.second), lossFunction(lossFunction), alpha(alpha){
    for(int i = 0; i < netStructure.size(); i++){
        LayerStructure currStructure = netStructure[i];
        const ActivationFunction& func = currStructure.getFunction();
        bool isNotLast = i != netStructure.size() - 1;

        // add bias neuron if layer is not last
        int currStructureSize = currStructure.getSize() + (isNotLast ? 1 : 0);
        int nextStructureSize = isNotLast ? netStructure[i + 1].getSize() : 0;
            
        bool hasBiasNeuron = i != netStructure.size() - 1;

        Layer layer = generateLayer(
            func,
            currStructureSize,
            nextStructureSize, 
            hasBiasNeuron
        );
        layers.push_back(layer);
    }
}

Layer Net::generateLayer(
    const ActivationFunction& function, 
    int size, 
    int weightCountEach, 
    bool hasBiasNeuron
){
    std::vector<std::vector<double>> weights(size);
    std::generate(weights.begin(), weights.end(), [&]() { 
        return this->generateWeights(weightCountEach); 
    });
    return Layer(function, weights, hasBiasNeuron);
}

std::vector<double> Net::generateWeights(int weightsCount){
    std::vector<double> weights(weightsCount);
    std::generate(weights.begin(), weights.end(), [&]() { 
        return this->generator.generate(); 
    });
    return weights;
}

void Net::propagate(const std::vector<double>& input, const std::vector<double>& targets){
    setInput(input);
    propagateForward();
    calculateOutputDerivatives(targets);
    propagateBackward();
}

void Net::setInput(const std::vector<double>& input){
    layers[0].calculateInputs(input);
}

void Net::calculateOutputDerivatives(const std::vector<double>& targets){
    Layer& outputLayer = layers.back();
    const std::vector<double>& outputs = outputLayer.getOutputs();
    std::vector<double> derivatives = lossFunction.calculateDerivatives(targets, outputs);
    outputLayer.setTotalDerivatives(derivatives);
}

void Net::propagateForward(){
    for(int i = 0; i < layers.size() - 1; i++){
        Layer& currLayer = layers[i];
        Layer& nextLayer = layers[i + 1];
        
        std::vector<double> inputs = currLayer.forwardPropagation();
        nextLayer.calculateInputs(inputs);  
    }
}

void Net::propagateBackward(){
    int d = layers.size();

    for(int i = d - 1; i > 0; i--){
        Layer& curr = layers[i];
        Layer& prev = layers[i - 1];

        propagateBackwardForLayers(curr, prev);
    }
}

void Net::propagateBackwardForLayers(Layer& curr, Layer& prev){
    std::vector<double> currDerivatives = curr.getTotalDerivatives();
    prev.updateTotalDerivatives(currDerivatives);
    prev.adjustWeights(currDerivatives, alpha);
}

std::vector<double> Net::getOutputs(const std::vector<double>& input){
    setInput(input);
    propagateForward();
    return layers.back().getOutputs();
}
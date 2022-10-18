#pragma once

#include <vector>
#include <memory>
#include "../Layer/Layer.h"
#include "../ActivationFunction/ActivationFunction.h"
#include "../LayerStructure/LayerStructure.h"
#include "../RandomValueGenerator/RandomValueGenerator.h"
#include "../LossFunction/LossFunction.h"
#include <utility>

class Net {
    private:
        std::vector<Layer> layers;
        const LossFunction& lossFunction; 
        double alpha;
        
        RandomValueGenerator generator;

        std::vector<double> generateWeights(int weightsCount);

        Layer generateLayer(
            const ActivationFunction& function, 
            int size, 
            int weightCountEach, 
            bool hasBiasNeuron
        );

        void setInput(const std::vector<double>& input);

        void propagateForward();
        void calculateOutputDerivatives(const std::vector<double>& targets);
        void propagateBackward();

        void propagateBackwardForLayers(Layer& curr, Layer& prev);
    public:
        Net(
            const LossFunction& lossFunction, 
            const std::vector<LayerStructure>& netStructure, 
            const std::pair<double, double> weightInitialRange,
            double alpha
        );
        void propagate(const std::vector<double>& input, const std::vector<double>& targets);

        double getTotalError() const;
        std::vector<double> getOutputs(const std::vector<double>& input);
};
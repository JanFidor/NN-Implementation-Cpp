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

        std::vector<double> generateWeights(int);
        Layer generateLayer(const ActivationFunction&, int, int);

        void setInput(const std::vector<double>&);

        void propagateForward();
        void calculateOutputDerivatives(const std::vector<double>&);    // TODO
        void propagateBackward();

        void propagateBackwardForLayers(Layer&, Layer&);
        void updateLayerWeights(Layer&, const std::vector<double>);
    public:
        Net(const LossFunction&, const std::vector<LayerStructure>&, const std::pair<double, double>, double);
        void propagate(const std::vector<double>&, const std::vector<double>&);

        double getTotalError() const;
        std::vector<double> getOutputs(const std::vector<double>&);
};
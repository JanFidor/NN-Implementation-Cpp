#pragma once

#include "../ActivationFunction/ActivationFunction.h"

class LayerStructure{
    private:
        int size;
        const ActivationFunction& function;
    public:
        LayerStructure(int size, const ActivationFunction& function);

        int getSize() const;
        const ActivationFunction& getFunction() const;
};
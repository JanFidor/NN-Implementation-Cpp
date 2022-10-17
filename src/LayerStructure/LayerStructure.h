#pragma once

#include "../ActivationFunction/ActivationFunction.h"

class LayerStructure{
    private:
        int size;
        const ActivationFunction& function;
    public:
        LayerStructure(int, const ActivationFunction&);

        int getSize() const;
        const ActivationFunction& getFunction() const;
};
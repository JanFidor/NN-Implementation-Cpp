#include "LayerStructure.h"


LayerStructure::LayerStructure(int size, const ActivationFunction& function) : size(size), function(function){}

int LayerStructure::getSize() const{
    return size;
}

const ActivationFunction& LayerStructure::getFunction() const{
    return function;
}
#include "ActivationFunction.h"

class Linear: public ActivationFunction{
    public:
        double calculateOutput(double input) const{
            return input;
        }
        double calculateDerivative(double input) const{
            return 1;
        }
};

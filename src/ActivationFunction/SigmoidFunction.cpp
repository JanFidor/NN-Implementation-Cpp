#include "ActivationFunction.h"
#include <math.h>

class SigmoidFunction: public ActivationFunction{
    public:
        double calculateOutput(double input) const{
            return 1 / (1 + exp(-input));
        }
        double calculateDerivative(double input) const{
            return input * (1 - input);
        }
};

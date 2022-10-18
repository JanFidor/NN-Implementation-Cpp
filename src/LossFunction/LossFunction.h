#pragma once

#include <math.h>
#include <vector>

class LossFunction{
    public:
        virtual double calculateLoss(
            const std::vector<double>& targets,
            const std::vector<double>& outputs
        ) const = 0;
        virtual std::vector<double> calculateDerivatives(
            const std::vector<double>& targets,
            const std::vector<double>& outputs
        ) const = 0;
};
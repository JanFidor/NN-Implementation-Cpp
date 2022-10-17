#include "LossFunction.h"
#include "math.h"

class MeanSquaredError: public LossFunction{
    public:
        double calculateLoss(const std::vector<double>& targets, const std::vector<double>& outputs) const{
            
            double sum = 0;
            for(int i = 0; i < targets.size(); i++)
                sum += pow(targets[i] - outputs[i], 2);
            
            return sum;
        }
        std::vector<double> calculateDerivatives(const std::vector<double>& targets, const std::vector<double>& outputs) const{
            std::vector<double> derivatives;
            for(int i = 0; i < targets.size(); i++)
                derivatives.emplace_back(2 * (outputs[i] - targets[i]));
            return derivatives;
        }
};
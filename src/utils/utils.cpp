#include <math.h>
#include "utils.h"

bool aproximatelyEqual(double a, double b, double epsilon) {
    return abs(a - b) < epsilon;
}
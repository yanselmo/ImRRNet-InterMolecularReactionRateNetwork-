def inverse_minmaxnorm(data,
                maxval = 24.8248,
                minval = -58.7808):
    return data * (maxval - minval) + minval
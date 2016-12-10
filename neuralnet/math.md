---
title: "math"
author: "Dean Kilfoyle"
date: "10 December 2016"
output: html_document
---



Step 1: Feed Forward

Each neuron in layer l receives weighted input from every neuron in layer l-1

$$z(x) = \sum_{i=1}^nw_ix_i+b$$

The weight matrix is stored as w_oi (output, input) so that the above equation can be completed for each neuron in a single dot product

    I1  I2         
H1  w   w   dot  I1   =   H1z
H2  w   w        I2       H2z

The output z is then feed into an activation function of several choices.

Sigmoid function

$$\sigma = \frac{1}{1+e^{-x}}$$

This squeezes all outputs into range 0..1 and prevents extreme values from affecting the output.


```r
x =(-100:100)/10
data.frame(x=x, y=1/(1+exp(-x))) %>% 
  ggplot(aes(x=x,y=y)) +
  geom_line()
```

![plot of chunk unnamed-chunk-1](figure/unnamed-chunk-1-1.png)

The derivative of the sigmoid function is:

$$S'(x) = S(x) * ( 1.0 - S(x) )$$

Step 2: Back propogation

Starting with the output layer the error is calculated for each neuron (where i is the expected:

$$E = (actual-expected)$$

Gradient descent calculates the gradient of the Error function for the given set of weights.

$$\frac{ \partial E}{\partial w_{(ik)}} = \delta_k \cdot o_i$$
Where the delta's are:

$$\delta_i = \begin{cases}-E f'(z_i) & \mbox{, output nodes}\\ f'(z_i) \sum_k w_{ki}\delta_k & \mbox{, interier nodes}\\ \end{cases}$$

where
i is each neuron in the current layer
k is each neuron in the backwards prior layer (ie current layer + 1)

Note interior deltas are calculated from the delta in the backwards prior layer.

Weights are updated to go down the gradient (ie to minimize the error)

$$w_k = w_k-\eta \sum_j \frac{\partial E_{X_j}}{\partial w_k} \label{deltaw}$$


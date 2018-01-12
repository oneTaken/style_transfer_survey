Content:
+ [project page](https://cs.stanford.edu/people/jcjohns/eccv16/)
+ [reference](#reference)
+ [background](#background)
+ [contribution](#contribution)
+ [intuition](#intuition)
+ [method](#method)
+ [shortcomings](#shortcomings)

#### reference 
+ [translation](https://www.jianshu.com/p/b728752a70e9)
+ [pytorch official example](https://github.com/pytorch/examples/tree/master/fast_neural_style)

#### background
Recent works use *per-pixel* loss between output and the ground-truth. Parallel work has shown
*perceptual* loss based on high-level features extracted from pretrained networks works well.

#### contribution
Similar results to optimization-based method, but three orders of magnitude faster, 
can be real-time. Also on super resolution problem.

#### intuition
Combine the benefits of these two approaches, one is *perceptual loss function*, the other
one is optimization. Train a feed-neural network and can run in real-time when inference.
The key insight of these methods is that convolutional neural networks pretrained for image
classification have already learned to encode the perceptual and semantic information.

#### method
Use fractionally-strided convolution to downsampling and upsampling. One benefit is low
computation, the other one is bigger receptive field. And some residual connections in the 
middle layers.
y is the transferred output, given the input image x.

Feature Reconstruction Loss:

MSELoss between the high dimensional representation of the pretrained model of x and y.

Style Reconstruction Loss:
Perform style reconstruction from a set of layers J, for each j in the set:

`loss_j = MSELoss(Gram(fj(x)), Gram(fj(y)))`

`fj` is the `j-th` layer representation in the layer set J of the pretrained model. 

#### shortcomings
Fast but one model for one style.

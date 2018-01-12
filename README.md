# style_transfer_survey
A survey on style_transfer from the original fantasy paper till now.

+ A Neural Algorithm of Artistic Style
    + arxiv: [1508.06576](https://arxiv.org/abs/1508.06576)
    + github: https://github.com/jcjohnson/neural-style
    + translation: https://www.jianshu.com/p/9f03b61fdeac
+ Texture Networks: Feed-forward Synthesis of Textures and Stylized Images
    + arxiv: [1603.03417](https://arxiv.org/abs/1603.03417), also [ICML 2016]()
    + github: [author torch](https://github.com/DmitryUlyanov/texture_nets), 
    + translation: https://www.jianshu.com/p/1187049ae1ad
+ Perceptual Losses for Real-Time Style Transfer and Super-Resolution
    + arxiv: [1603.08155](https://arxiv.org/abs/1603.08155), also [ECCV 2016]()
    + github: [author torch](https://github.com/jcjohnson/fast-neural-style), [third pytorch](https://github.com/abhiskk/fast-neural-style)
    + translation: https://www.jianshu.com/p/b728752a70e9
    + webpage: [project page](https://cs.stanford.edu/people/jcjohns/eccv16/)
+ Instance Normalization: The missing Ingredient for Fast Stylization
    + arxiv: [1607.08022](https://arxiv.org/abs/1607.08022)
    + github: [author torch](https://github.com/DmitryUlyanov/texture_nets)
    + translation: https://www.jianshu.com/p/d77b6273b990
+ Image Style Transfer Using Convolutional Neural Networks
    + arxiv: [CVPR16](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)
    + github: [author pytorch](https://github.com/leongatys/PytorchNeuralStyleTransfer)
+ A Learned Representation For Artistic Style
    + arxiv: [1610.07629](https://arxiv.org/abs/1610.07629), also [ICLR 2017](https://openreview.net/forum?id=BJO-BuT1g&noteId=BJO-BuT1g)
    + github: [tf](https://github.com/tensorflow/magenta/tree/master/magenta/models/image_stylization)
+ Controlling Perceptual Factors in Neural Style Transfer
    + arxiv: [1611.07865](https://arxiv.org/abs/1611.07865), also [CVPR 2017]()
    + github: [author pytorch](https://github.com/leongatys/PytorchNeuralStyleTransfer)
+ Fast Patch-based Style Transfer of Arbitrary Style
    + arxiv: [1612.04337](https://arxiv.org/abs/1612.04337)
    + github: [author torch](https://github.com/rtqichen/style-swap)
    + reference
        + http://blog.csdn.net/wyl1987527/article/details/70476044
        + http://blog.csdn.net/Hungryof/article/details/61195783
        + http://mathworld.wolfram.com/FrobeniusNorm.html
+ Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization 
    + arxiv: [1703.06868](https://arxiv.org/abs/1703.06868), also [ICCV 2017]()
    + github: [author torch](https://github.com/xunhuang1995/AdaIN-style), [third pytorch](https://github.com/naoto0804/pytorch-AdaIN)
    + webpage: [project](http://www.cs.cornell.edu/~xhuang/publication/adain/)
+ Deep Photo Style Transfer
    + arxiv: [1703.07511](https://arxiv.org/abs/1703.07511)
    + github: [author torch](https://github.com/luanfujun/deep-photo-styletransfer), https://github.com/LouieYang/deep-photo-styletransfer-tf
    + translation: http://blog.csdn.net/cicibabe/article/details/70868746
+ Universal Style Transfer via Feature Transforms
    + arxiv: [1705.08086](https://arxiv.org/abs/1705.08086), also [NIPS 2017]()
    + github: [author torch](https://github.com/Yijunmaverick/UniversalStyleTransfer), [third pytorch](https://github.com/sunshineatnoon/PytorchWCT)
    + 
    
# practice
First, pytorch has a official example [fast_neural_style](https://github.com/pytorch/examples/tree/master/fast_neural_style).

points:
$$loss =\alpha*loss_{content} + \beta * loss_{style}$$
$$y=TransferNet(x)$$
$$loss_{content}=MSELoss(\Gamma(x),\Gamma(y))$$ 

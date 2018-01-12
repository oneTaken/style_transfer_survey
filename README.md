# style_transfer_survey
A survey on style_transfer from the original fantasy paper till now.

# paper
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

Points:

+ training phase
    + content image **x**
    + style image **s**
    + pretrained model **F**[1,2,3,4], different middle-level feature representations on high dimension, 
    (VGG16). Freezed weight parameters
    + Style Transfer Model **T** , a FCN model with size invariant
    + `loss = weight_content * loss_content + weight_style * loss_style`
+ evaluating phase
    + content image x
    + trained Style Transfer Model 
    + styled image y = **T(x)**

More Details:
+ primary criterion is MSELoss.
+ loss_content is criterion(F2(x), F2(y))
+ GramMatrix **G** is :
```python
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
```
+ `gm_s = [G(F1(s)), G(F2(s)), G(F3(s)), G(F4(s))]`
+ `gm_y = [G(F1(y)), G(F2(y)), G(F3(y)), G(F4(y))]`
+ `loss_style = sum([MSELoss(gm_s[i], gm_y[i]) for i in range(len(gm_s))])`
+ padding is reflection, not constant 0

Size Analysis:
+ `x.shape=(m1, n1, 3)`
+ `s.shape=(m2, n2, 3)`
+ `batch_size = b`
+ T downsample two times, both `int(ceil(x/2))`, this will bring size difference. 
For example, input image is size`(3,33,33)`, output size is `(3,36,36)`. Saying proper.
+ `gm_s[i]` size is `(b, ch[i], ch[i])`

Think About The Model:
+  The VGG16 is just a representation on high dimension. It can be replaced by any other 
similar pretrained model.
+ The four middle-level representations can also be chosen as other.

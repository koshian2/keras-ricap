# A Keras/TensorFlow implementation of RICAP
This repository is Keras's implementation of RICAP(Random Image Cropping And Patching), Data Augmentation algorithm which achieved SoTA with CIFAR-10 at error rate 2.19%. We applied to [animeface-character-dataset](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/).

# How to use
It is easy to inherit from ImageDataGenerator of Keras like this:

```python
from keras.preprocessing.image import ImageDataGenerator
from ricap import ricap

# RICAP Generator
class RICAPGenerator(ImageDataGenerator):
    def __init__(self, ricap_beta=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ricap_beta = ricap_beta

    def flow_from_directory(self, *args, **kwargs):
        for batch_X, batch_y in super().flow_from_directory(*args, **kwargs):
            ricap_X, ricap_y = ricap(batch_X, batch_y, self.ricap_beta)
            yield ricap_X, ricap_y
```

# Use same random numbers with beta distribution in batch vs. Use different random numbers for each sample
There are 2 versions here.

1. Use same random numbers in batch (**default, same as the paper**)
1. Use different random numbers for each sample (made it on our own)

If you want to try differnt random numbers version, please specify ```use_same_random_value_on_batch=False```

## Same version
![](https://github.com/koshian2/keras-ricap/blob/master/images/ricap_same.gif)

## Different version
![](https://github.com/koshian2/keras-ricap/blob/master/images/ricap_diffenet.gif)

## Results
|              Conditions             | Test Accuracy |
|:-----------------------------------:|--------------:|
|               baseline              |      91.5039% |
| batchwise RICAP (original) beta=0.1 |      91.7236% |
| batchwise RICAP (original) beta=0.3 |      92.3340% |
| batchwise RICAP (original) beta=0.5 |      91.6748% |
| samplewise RICAP (changed) beta=0.1 |      92.0898% |
| samplewise RICAP (changed) beta=0.3 |      91.9922% |
| samplewise RICAP (changed) beta=0.5 |      91.6504% |

![](https://github.com/koshian2/keras-ricap/blob/master/images/ricap_03.png)
![](https://github.com/koshian2/keras-ricap/blob/master/images/ricap_04.png)

Accuracy is better for batchwise(the original), but samplewise is smoother in error.

## Beta scheduling
Also, we tried a method to schedule and change the beta value (a method close to [the pseudo label of semi supervised learning](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf)).

**example:**
* beta=0.01 if epoch<=20
* increase beta value linearly if 20<epoch<=90
* beta=1 if epoch>90 (i.e. uniform distribution)

![](https://github.com/koshian2/keras-ricap/blob/master/images/ricap_05.png)

On beta scheduling, it seems that samplewise RICAP is better.

# Refence
Ryo Takahashi, Takashi Matsubara, Kuniaki Uehara. [Data Augmentation using Random Image Cropping and Patching for Deep CNNs](https://arxiv.org/abs/1811.09030). ACML2018. 2018

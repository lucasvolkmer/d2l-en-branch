# Network in Network (NiN)
:label:`sec_nin`

LeNet, AlexNet e VGG compartilham um padrão de design comum:
extrair recursos que exploram a estrutura *espacial*
por meio de uma sequência de camadas de convolução e agrupamento
e, em seguida, pós-processar as representações por meio de camadas totalmente conectadas.
As melhorias no LeNet por AlexNet e VGG residem principalmente
em como essas redes posteriores ampliam e aprofundam esses dois módulos.
Alternativamente, pode-se imaginar o uso de camadas totalmente conectadas
no início do processo.
No entanto, um uso descuidado de camadas densas pode desistir da
estrutura espacial da representação inteiramente,
Os blocos *rede em rede* (*NiN*) oferecem uma alternativa.
Eles foram propostos com base em uma visão muito simples:
para usar um MLP nos canais para cada pixel separadamente :cite:`Lin.Chen.Yan.2013`.


## Blocos NiN 

Recall that the inputs and outputs of convolutional layers
consist of four-dimensional tensors with axes
corresponding to the example, channel, height, and width.
Also recall that the inputs and outputs of fully-connected layers
are typically two-dimensional tensors corresponding to the example and feature.
The idea behind NiN is to apply a fully-connected layer
at each pixel location (for each height and  width).
If we tie the weights across each spatial location,
we could think of this as a $1\times 1$ convolutional layer
(as described in :numref:`sec_channels`)
or as a fully-connected layer acting independently on each pixel location.
Another way to view this is to think of each element in the spatial dimension
(height and width) as equivalent to an example
and a channel as equivalent to a feature.

:numref:`fig_nin` illustrates the main structural differences
between VGG and NiN, and their blocks.
The NiN block consists of one convolutional layer
followed by two $1\times 1$ convolutional layers that act as
per-pixel fully-connected layers with ReLU activations.
The convolution window shape of the first layer is typically set by the user.
The subsequent window shapes are fixed to $1 \times 1$.

Lembre-se de que as entradas e saídas das camadas convolucionais
consistem em tensores quadridimensionais com eixos
correspondendo ao exemplo, canal, altura e largura.
Lembre-se também de que as entradas e saídas de camadas totalmente conectadas
são tipicamente tensores bidimensionais correspondentes ao exemplo e ao recurso.
A ideia por trás do NiN é aplicar uma camada totalmente conectada
em cada localização de pixel (para cada altura e largura).
Se amarrarmos os pesos em cada localização espacial,
poderíamos pensar nisso como uma camada convolucional $ 1 \ vezes 1 $
(conforme descrito em: numref: `sec_channels`)
ou como uma camada totalmente conectada agindo de forma independente em cada localização de pixel.
Outra maneira de ver isso é pensar em cada elemento na dimensão espacial
(altura e largura) como equivalente a um exemplo
e um canal como equivalente a um recurso.

:numref:`fig_nin` illustrates the main structural differences
between VGG and NiN, and their blocks.
The NiN block consists of one convolutional layer
followed by two $1\times 1$ convolutional layers that act as
per-pixel fully-connected layers with ReLU activations.
The convolution window shape of the first layer is typically set by the user.
The subsequent window shapes are fixed to $1 \times 1$.

: numref: `fig_nin` ilustra as principais diferenças estruturais
entre VGG e NiN, e seus blocos.
O bloco NiN consiste em uma camada convolucional
seguido por duas camadas convolucionais $ 1 \ vezes 1 $ que atuam como
camadas totalmente conectadas por pixel com ativações ReLU.
A forma da janela de convolução da primeira camada é normalmente definida pelo usuário.
As formas de janela subsequentes são fixadas em $ 1 \ vezes 1 $.

![Comparing architectures of VGG and NiN, and their blocks.](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides,
                               padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                               activation='relu')])
```

## NiN Model

The original NiN network was proposed shortly after AlexNet
and clearly draws some inspiration.
NiN uses convolutional layers with window shapes
of $11\times 11$, $5\times 5$, and $3\times 3$,
and the corresponding numbers of output channels are the same as in AlexNet. Each NiN block is followed by a maximum pooling layer
with a stride of 2 and a window shape of $3\times 3$.

A rede NiN original foi proposta logo após AlexNet
e claramente tira alguma inspiração.
NiN usa camadas convolucionais com formas de janela
de $ 11 \ vezes 11 $, $ 5 \ vezes 5 $ e $ 3 \ vezes 3 $,
e os números correspondentes de canais de saída são iguais aos do AlexNet. Cada bloco NiN é seguido por uma camada de pooling máxima
com um passo de 2 e uma forma de janela de $ 3 \ vezes 3 $.

One significant difference between NiN and AlexNet
is that NiN avoids fully-connected layers altogether.
Instead, NiN uses an NiN block with a number of output channels equal to the number of label classes, followed by a *global* average pooling layer,
yielding a vector of logits.
One advantage of NiN's design is that it significantly
reduces the number of required model parameters.
However, in practice, this design sometimes requires
increased model training time.

Uma diferença significativa entre NiN e AlexNet
é que o NiN evita camadas totalmente conectadas.
Em vez disso, NiN usa um bloco NiN com um número de canais de saída igual ao número de classes de rótulo, seguido por uma camada de pooling média * global *,
produzindo um vetor de logits.
Uma vantagem do design da NiN é que
reduz o número de parâmetros de modelo necessários.
No entanto, na prática, esse design às vezes requer
aumento do tempo de treinamento do modelo.

```{.python .input}
net = nn.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding=1),
        # The global average pooling layer automatically sets the window shape
        # to the height and width of the input
        nn.GlobalAvgPool2D(),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        nn.Flatten())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # There are 10 label classes
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # Transform the four-dimensional output into two-dimensional output with a
    # shape of (batch size, 10)
    nn.Flatten())
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Dropout(0.5),
        # There are 10 label classes
        nin_block(10, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 1, 10)),
        # Transform the four-dimensional output into two-dimensional output
        # with a shape of (batch size, 10)
        tf.keras.layers.Flatten(),
        ])
```

We create a data example to see the output shape of each block.

Criamos um exemplo de dados para ver a forma de saída de cada bloco.

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## Training

As before we use Fashion-MNIST to train the model.
NiN's training is similar to that for AlexNet and VGG.

Como antes, usamos o Fashion-MNIST para treinar a modelo.
O treinamento de NiN é semelhante ao de AlexNet e VGG.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Summary

* NiN uses blocks consisting of a convolutional layer and multiple $1\times 1$ convolutional layers. This can be used within the convolutional stack to allow for more per-pixel nonlinearity.
* NiN removes the fully-connected layers and replaces them with global average pooling (i.e., summing over all locations) after reducing the number of channels to the desired number of outputs (e.g., 10 for Fashion-MNIST).
* Removing the fully-connected layers reduces overfitting. NiN has dramatically fewer parameters.
* The NiN design influenced many subsequent CNN designs.

* NiN usa blocos que consistem em uma camada convolucional e várias camadas convolucionais $ 1 \ vezes 1 $. Isso pode ser usado dentro da pilha convolucional para permitir mais não linearidade por pixel.
* NiN remove as camadas totalmente conectadas e as substitui pelo agrupamento médio global (ou seja, somando todos os locais) depois de reduzir o número de canais para o número desejado de saídas (por exemplo, 10 para Fashion-MNIST).
* A remoção das camadas totalmente conectadas reduz o ajuste excessivo. NiN tem muito menos parâmetros.
* O design NiN influenciou muitos designs subsequentes da CNN.

## Exercises

1. Tune the hyperparameters to improve the classification accuracy.
1. Why are there two $1\times 1$ convolutional layers in the NiN block? Remove one of them, and then observe and analyze the experimental phenomena.
1. Calculate the resource usage for NiN.
    1. What is the number of parameters?
    1. What is the amount of computation?
    1. What is the amount of memory needed during training?
    1. What is the amount of memory needed during prediction?
1. What are possible problems with reducing the $384 \times 5 \times 5$ representation to a $10 \times 5 \times 5$ representation in one step?

1. Ajuste os hiperparâmetros para melhorar a precisão da classificação.
1. Por que existem duas camadas convolucionais $ 1 \ times 1 $ no bloco NiN? Remova um deles e então observe e analise os fenômenos experimentais.
1. Calcule o uso de recursos para NiN.
     1. Qual é o número de parâmetros?
     1. Qual é a quantidade de computação?
     1. Qual é a quantidade de memória necessária durante o treinamento?
     1. Qual é a quantidade de memória necessária durante a previsão?
1. Quais são os possíveis problemas com a redução da representação $ 384 \ vezes 5 \ vezes 5 $ para uma representação $ 10 \ vezes 5 \ vezes 5 $ em uma etapa?
:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/332)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjAwNDUxNDgyXX0=
-->
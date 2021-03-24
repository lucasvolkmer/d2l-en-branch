# Densely Connected Networks (DenseNet)

ResNet significantly changed the view of how to parametrize the functions in deep networks. *DenseNet* (dense convolutional network) is to some extent the logical extension of this :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
To understand how to arrive at it, let us take a small detour to mathematics.

A ResNet mudou significativamente a visão de como parametrizar as funções em redes profundas. * DenseNet * (rede convolucional densa) é, até certo ponto, a extensão lógica disso: cite: `Huang.Liu.Van-Der-Maaten.ea.2017`.
Para entender como chegar a isso, façamos um pequeno desvio para a matemática.


## From ResNet to DenseNet

## De ResNet para DenseNet

Recall the Taylor expansion for functions. For the point $x = 0$ it can be written as
Lembre-se da expansão de Taylor para funções. Para o ponto $ x = 0 $, pode ser escrito como

$$f(x) = f(0) + f'(0) x + \frac{f''(0)}{2!}  x^2 + \frac{f'''(0)}{3!}  x^3 + \ldots.$$


The key point is that it decomposes a function into increasingly higher order terms. In a similar vein, ResNet decomposes functions into

O ponto principal é que ele decompõe uma função em termos de ordem cada vez mais elevados. De maneira semelhante, o ResNet decompõe funções em

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

That is, ResNet decomposes $f$ into a simple linear term and a more complex
nonlinear one.
What if we want to capture (not necessarily add) information beyond two terms?
One solution was 

Ou seja, o ResNet decompõe $ f $ em um termo linear simples e um termo mais complexo
não linear.
E se quisermos capturar (não necessariamente adicionar) informações além de dois termos?
Uma solução foi
DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.

![The main difference between ResNet (left) and DenseNet (right) in cross-layer connections: use of addition and use of concatenation. ](../img/densenet-block.svg)
:label:`fig_densenet_block`

As shown in :numref:`fig_densenet_block`, the key difference between ResNet and DenseNet is that in the latter case outputs are *concatenated* (denoted by $[,]$) rather than added.
As a result, we perform a mapping from $\mathbf{x}$ to its values after applying an increasingly complex sequence of functions:

Conforme mostrado em: numref: `fig_densenet_block`, a principal diferença entre ResNet e DenseNet é que, no último caso, as saídas são * concatenadas * (denotadas por $ [,] $) em vez de adicionadas.
Como resultado, realizamos um mapeamento de $ \ mathbf {x} $ para seus valores após aplicar uma sequência cada vez mais complexa de funções:

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2([\mathbf{x}, f_1(\mathbf{x})]), f_3([\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})])]), \ldots\right].$$

In the end, all these functions are combined in MLP to reduce the number of features again. In terms of implementation this is quite simple:
rather than adding terms, we concatenate them. The name DenseNet arises from the fact that the dependency graph between variables becomes quite dense. The last layer of such a chain is densely connected to all previous layers. The dense connections are shown in :numref:`fig_densenet`.

No final, todas essas funções são combinadas no MLP para reduzir o número de recursos novamente. Em termos de implementação, isso é bastante simples:
em vez de adicionar termos, nós os concatenamos. O nome DenseNet surge do fato de o gráfico de dependência entre as variáveis se tornar bastante denso. A última camada de tal cadeia está densamente conectada a todas as camadas anteriores. As conexões densas são mostradas em: numref: `fig_densenet`.

![Dense connections in DenseNet.](../img/densenet.svg)
:label:`fig_densenet`


The main components that compose a DenseNet are *dense blocks* and *transition layers*. The former define how the inputs and outputs are concatenated, while the latter control the number of channels so that it is not too large.

Os principais componentes que compõem uma DenseNet são * blocos densos * e * camadas de transição *. O primeiro define como as entradas e saídas são concatenadas, enquanto o último controla o número de canais para que não seja muito grande.

## Dense Blocks

DenseNet uses the modified "batch normalization, activation, and convolution"
structure of ResNet (see the exercise in :numref:`sec_resnet`).
First, we implement this convolution block structure.

A DenseNet usa a "normalização, ativação e convolução em lote" modificada
estrutura do ResNet (veja o exercício em: numref: `sec_resnet`).
Primeiro, implementamos essa estrutura de bloco de convolução.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

A *dense block* consists of multiple convolution blocks, each using the same number of output channels. In the forward propagation, however, we concatenate the input and output of each convolution block on the channel dimension.

Um * bloco denso * consiste em vários blocos de convolução, cada um usando o mesmo número de canais de saída. Na propagação direta, entretanto, concatenamos a entrada e a saída de cada bloco de convolução na dimensão do canal.

```{.python .input}
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
#@tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
#@tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

In the following example,
we define a `DenseBlock` instance with 2 convolution blocks of 10 output channels.
When using an input with 3 channels, we will get an output with  $3+2\times 10=23$ channels. The number of convolution block channels controls the growth in the number of output channels relative to the number of input channels. This is also referred to as the *growth rate*.

No exemplo a seguir,
definimos uma instância `DenseBlock` com 2 blocos de convolução de 10 canais de saída.
Ao usar uma entrada com 3 canais, obteremos uma saída com $ 3 + 2 \ vezes 10 = 23 $ canais. O número de canais de bloco de convolução controla o crescimento do número de canais de saída em relação ao número de canais de entrada. Isso também é conhecido como * taxa de crescimento *.

```{.python .input}
blk = DenseBlock(2, 10)
blk.initialize()
X = np.random.uniform(size=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab pytorch
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = DenseBlock(2, 10)
X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

## Transition Layers

Since each dense block will increase the number of channels, adding too many of them will lead to an excessively complex model. A *transition layer* is used to control the complexity of the model. It reduces the number of channels by using the $1\times 1$ convolutional layer and halves the height and width of the average pooling layer with a stride of 2, further reducing the complexity of the model.

Uma vez que cada bloco denso aumentará o número de canais, adicionar muitos deles levará a um modelo excessivamente complexo. Uma * camada de transição * é usada para controlar a complexidade do modelo. Ele reduz o número de canais usando a camada convolucional $ 1 \ vezes 1 $ e divide pela metade a altura e a largura da camada de pooling média com uma distância de 2, reduzindo ainda mais a complexidade do modelo.

```{.python .input}
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
#@tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

Apply a transition layer with 10 channels to the output of the dense block in the previous example.  This reduces the number of output channels to 10, and halves the height and width.

Aplique uma camada de transição com 10 canais à saída do bloco denso no exemplo anterior. Isso reduz o número de canais de saída para 10 e divide a altura e a largura pela metade.

```{.python .input}
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
#@tab pytorch
blk = transition_block(23, 10)
blk(Y).shape
```

```{.python .input}
#@tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

## DenseNet Model

Next, we will construct a DenseNet model. DenseNet first uses the same single convolutional layer and maximum pooling layer as in ResNet.

A seguir, construiremos um modelo DenseNet. A DenseNet usa primeiro a mesma camada convolucional única e camada máxima de pooling que no ResNet.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def block_1():
    return tf.keras.Sequential([
       tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
       tf.keras.layers.BatchNormalization(),
       tf.keras.layers.ReLU(),
       tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

Then, similar to the four modules made up of residual blocks that ResNet uses,
DenseNet uses four dense blocks.
Similar to ResNet, we can set the number of convolutional layers used in each dense block. Here, we set it to 4, consistent with the ResNet-18 model in :numref:`sec_resnet`. Furthermore, we set the number of channels (i.e., growth rate) for the convolutional layers in the dense block to 32, so 128 channels will be added to each dense block.

In ResNet, the height and width are reduced between each module by a residual block with a stride of 2. Here, we use the transition layer to halve the height and width and halve the number of channels.

Então, semelhante aos quatro módulos compostos de blocos residuais que o ResNet usa,
A DenseNet usa quatro blocos densos.
Semelhante ao ResNet, podemos definir o número de camadas convolucionais usadas em cada bloco denso. Aqui, nós o definimos como 4, consistente com o modelo ResNet-18 em: numref: `sec_resnet`. Além disso, definimos o número de canais (ou seja, taxa de crescimento) para as camadas convolucionais no bloco denso para 32, de modo que 128 canais serão adicionados a cada bloco denso.

No ResNet, a altura e a largura são reduzidas entre cada módulo por um bloco residual com uma distância de 2. Aqui, usamos a camada de transição para reduzir pela metade a altura e a largura e pela metade o número de canais.

```{.python .input}
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))
```

```{.python .input}
#@tab pytorch
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

```{.python .input}
#@tab tensorflow
def block_2():
    net = block_1()
    # `num_channels`: the current number of channels
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # This is the number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net
```

Similar to ResNet, a global pooling layer and a fully-connected layer are connected at the end to produce the output.


Semelhante ao ResNet, uma camada de pooling global e uma camada totalmente conectada são conectadas na extremidade para produzir a saída.

```{.python .input}
net.add(nn.BatchNorm(),
        nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10))
    return net
```

## Training

Since we are using a deeper network here, in this section, we will reduce the input height and width from 224 to 96 to simplify the computation.

Como estamos usando uma rede mais profunda aqui, nesta seção, reduziremos a altura e largura de entrada de 224 para 96 para simplificar o cálculo.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Summary

* In terms of cross-layer connections, unlike ResNet, where inputs and outputs are added together, DenseNet concatenates inputs and outputs on the channel dimension.
* The main components that compose DenseNet are dense blocks and transition layers.
* We need to keep the dimensionality under control when composing the network by adding transition layers that shrink the number of channels again.

* Em termos de conexões entre camadas, ao contrário do ResNet, onde entradas e saídas são adicionadas, o DenseNet concatena entradas e saídas na dimensão do canal.
* Os principais componentes que compõem o DenseNet são blocos densos e camadas de transição.
* Precisamos manter a dimensionalidade sob controle ao compor a rede, adicionando camadas de transição que reduzem o número de canais novamente.

## Exercises

1. Why do we use average pooling rather than maximum pooling in the transition layer?
1. One of the advantages mentioned in the DenseNet paper is that its model parameters are smaller than those of ResNet. Why is this the case?
1. One problem for which DenseNet has been criticized is its high memory consumption.
    1. Is this really the case? Try to change the input shape to $224\times 224$ to see the actual GPU memory consumption.
    1. Can you think of an alternative means of reducing the memory consumption? How would you need to change the framework?
1. Implement the various DenseNet versions presented in Table 1 of the DenseNet paper :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
1. Design an MLP-based model by applying the DenseNet idea. Apply it to the housing price prediction task in :numref:`sec_kaggle_house`.

1. Por que usamos pooling médio em vez de pooling máximo na camada de transição?
1. Uma das vantagens mencionadas no artigo da DenseNet é que seus parâmetros de modelo são menores que os do ResNet. Por que isso acontece?
1. Um problema pelo qual a DenseNet foi criticada é o alto consumo de memória.
     1. Este é realmente o caso? Tente alterar a forma de entrada para $ 224 \ vezes 224 $ para ver o consumo real de memória da GPU.
     1. Você consegue pensar em um meio alternativo de reduzir o consumo de memória? Como você precisa mudar a estrutura?
1. Implemente as várias versões da DenseNet apresentadas na Tabela 1 do artigo da DenseNet: cite: `Huang.Liu.Van-Der-Maaten.ea.2017`.
1. Projete um modelo baseado em MLP aplicando a ideia DenseNet. Aplique-o à tarefa de previsão do preço da habitação em: numref: `sec_kaggle_house`.
:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/331)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI1OTU3OTUzNV19
-->
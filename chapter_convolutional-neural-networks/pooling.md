# *Pooling*
:label:`sec_pooling`



Muitas vezes, conforme processamos imagens, queremos gradualmente
reduzir a resolução espacial de nossas representações ocultas,
agregando informações para que
quanto mais alto subimos na rede,
maior o campo receptivo (na entrada)
ao qual cada nó oculto é sensível.

Muitas vezes, nossa tarefa final faz alguma pergunta global sobre a imagem,
por exemplo, *contém um gato?*
Então, normalmente, as unidades de nossa camada final devem ser sensíveis
para toda a entrada.
Ao agregar informações gradualmente, produzindo mapas cada vez mais grosseiros,
alcançamos esse objetivo de, em última análise, aprendendo uma representação global,
enquanto mantém todas as vantagens das camadas convolucionais nas camadas intermediárias de processamento.

Além disso, ao detectar recursos de nível inferior, como bordas
(conforme discutido em :numref:`sec_conv_layer`),
frequentemente queremos que nossas representações sejam um tanto invariáveis ​​à tradução.
Por exemplo, se pegarmos a imagem `X`
com uma delimitação nítida entre preto e branco
e deslocarmos a imagem inteira em um pixel para a direita,
ou seja, `Z [i, j] = X [i, j + 1]`,
então a saída para a nova imagem `Z` pode ser muito diferente.
A borda terá deslocado um pixel.
Na realidade, os objetos dificilmente ocorrem exatamente no mesmo lugar.
Na verdade, mesmo com um tripé e um objeto estacionário,
a vibração da câmera devido ao movimento do obturador
pode mudar tudo em um pixel ou mais
(câmeras de última geração são carregadas com recursos especiais para resolver esse problema).

Esta seção apresenta *camadas de pooling*,
que servem ao duplo propósito de
mitigando a sensibilidade das camadas convolucionais à localização
e de representações de *downsampling* espacialmente.

## *Pooling* Máximo e *Pooling* Médio


Como camadas convolucionais, operadores de *pooling*
consistem em uma janela de formato fixo que é deslizada
todas as regiões na entrada de acordo com seu passo,
computando uma única saída para cada local percorrido
pela janela de formato fixo (também conhecida como *janela de pooling*).
No entanto, ao contrário do cálculo de correlação cruzada
das entradas e grãos na camada convolucional,
a camada de *pooling* não contém parâmetros (não há *kernel*).
Em vez disso, os operadores de *pooling* são determinísticos,
normalmente calculando o valor máximo ou médio
dos elementos na janela de *pooling*.
Essas operações são chamadas de *pooling máximo* (*pooling máximo* para breve)
e *pooling médio*, respectivamente.


Em ambos os casos, como com o operador de correlação cruzada,
podemos pensar na janela de *pooling*
começando da parte superior esquerda do tensor de entrada
e deslizando pelo tensor de entrada da esquerda para a direita e de cima para baixo.
Em cada local que atinge a janela de *pooling*,
ele calcula o máximo ou o médio
valor do subtensor de entrada na janela,
dependendo se o *pooling* máximo ou médio é empregado.

![Pooling máximo com uma forma de janela de pool de $2\times 2$. As partes sombreadas são o primeiro elemento de saída, bem como os elementos tensores de entrada usados para o cálculo de saída: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

The output tensor in :numref:`fig_pooling`  has a height of 2 and a width of 2.
The four elements are derived from the maximum value in each pooling window:

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

A pooling layer with a pooling window shape of $p \times q$
is called a $p \times q$ pooling layer.
The pooling operation is called $p \times q$ pooling.

Let us return to the object edge detection example
mentioned at the beginning of this section.
Now we will use the output of the convolutional layer
as the input for $2\times 2$ maximum pooling.
Set the convolutional layer input as `X` and the pooling layer output as `Y`. Whether or not the values of `X[i, j]` and `X[i, j + 1]` are different,
or `X[i, j + 1]` and `X[i, j + 2]` are different,
the pooling layer always outputs `Y[i, j] = 1`.
That is to say, using the $2\times 2$ maximum pooling layer,
we can still detect if the pattern recognized by the convolutional layer
moves no more than one element in height or width.

In the code below, we implement the forward propagation
of the pooling layer in the `pool2d` function.
This function is similar to the `corr2d` function
in :numref:`sec_conv_layer`.
However, here we have no kernel, computing the output
as either the maximum or the average of each region in the input.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

We can construct the input tensor `X` in :numref:`fig_pooling` to validate the output of the two-dimensional maximum pooling layer.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

Also, we experiment with the average pooling layer.

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## Padding and Stride

As with convolutional layers, pooling layers
can also change the output shape.
And as before, we can alter the operation to achieve a desired output shape
by padding the input and adjusting the stride.
We can demonstrate the use of padding and strides
in pooling layers via the built-in two-dimensional maximum pooling layer from the deep learning framework.
We first construct an input tensor `X` whose shape has four dimensions,
where the number of examples (batch size) and number of channels are both 1.

:begin_tab:`tensorflow`
It is important to note that tensorflow
prefers and is optimized for *channels-last* input.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```


```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

By default, the stride and the pooling window in the instance from the framework's built-in class
have the same shape.
Below, we use a pooling window of shape `(3, 3)`,
so we get a stride shape of `(3, 3)` by default.

```{.python .input}
pool2d = nn.MaxPool2D(3)
# Because there are no model parameters in the pooling layer, we do not need
# to call the parameter initialization function
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
pool2d(X)
```

The stride and padding can be manually specified.

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```



:begin_tab:`mxnet`
Of course, we can specify an arbitrary rectangular pooling window
and specify the padding and stride for height and width, respectively.
:end_tab:

:begin_tab:`pytorch`
Of course, we can specify an arbitrary rectangular pooling window
and specify the padding and stride for height and width, respectively.
For `nn.MaxPool2D` padding should be smaller than half of the kernel_size.
If the condition is not met, we can first pad the input using
`nn.functional.pad` and then pass it to the pooling layer.
:end_tab:

:begin_tab:`tensorflow`
Of course, we can specify an arbitrary rectangular pooling window
and specify the padding and stride for height and width, respectively.
In TensorFlow, to implement a padding of 1 all the way around the tensor, a function designed for padding 
must be invoked using `tf.pad`. This will implement the required padding and allow the aforementioned (3, 3) pooling with a (2, 2) stride to perform
similar to those in PyTorch and MXNet. When padding in this way, the built-in `padding` variable must be set to `valid`.
:end_tab:

```{.python .input}
pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab pytorch
X_pad = nn.functional.pad(X, (2, 2, 1, 1))
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3))
pool2d(X_pad)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1, 1], [2, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2,3))
pool2d(X_padded)
```


## Multiple Channels

When processing multi-channel input data,
the pooling layer pools each input channel separately,
rather than summing the inputs up over channels
as in a convolutional layer.
This means that the number of output channels for the pooling layer
is the same as the number of input channels.
Below, we will concatenate tensors `X` and `X + 1`
on the channel dimension to construct an input with 2 channels. 

:begin_tab:`tensorflow`
Note that this will require a 
concatenation along the last dimension for TensorFlow due to the channels-last syntax.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.concat([X, X + 1], 3)  # Concatenate along `dim=3` due to channels-last syntax
```

As we can see, the number of output channels is still 2 after pooling.

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)

```

:begin_tab:`tensorflow`
Note that the output for the tensorflow pooling appears at first glance to be different, however 
numerically the same results are presented as MXNet and PyTorch.
The difference lies in the dimensionality, and reading the 
output vertically yields the same output as the other implementations. 
:end_tab:

## Summary

* Taking the input elements in the pooling window, the maximum pooling operation assigns the maximum value as the output and the average pooling operation assigns the average value as the output.
* One of the major benefits of a pooling layer is to alleviate the excessive sensitivity of the convolutional layer to location.
* We can specify the padding and stride for the pooling layer.
* Maximum pooling, combined with a stride larger than 1 can be used to reduce the spatial dimensions (e.g., width and height).
* The pooling layer's number of output channels is the same as the number of input channels.


## Exercises

1. Can you implement average pooling as a special case of a convolution layer? If so, do it.
1. Can you implement max pooling as a special case of a convolution layer? If so, do it.
1. What is the computational cost of the pooling layer? Assume that the input to the pooling layer is of size $c\times h\times w$, the pooling window has a shape of $p_h\times p_w$ with a padding of $(p_h, p_w)$ and a stride of $(s_h, s_w)$.
1. Why do you expect maximum pooling and average pooling to work differently?
1. Do we need a separate minimum pooling layer? Can you replace it with another operation?
1. Is there another operation between average and maximum pooling that you could consider (hint: recall the softmax)? Why might it not be so popular?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3MjMwOTM1ODQsLTEzMDk5MTE4Nl19
-->
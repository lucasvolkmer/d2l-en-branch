# Canais de Múltiplas Entradas e Saídas
:label:`sec_channels`


Embora tenhamos descrito os vários canais
que compõem cada imagem (por exemplo, imagens coloridas têm os canais RGB padrão
para indicar a quantidade de vermelho, verde e azul) e camadas convolucionais para vários canais em :numref:`subsec_why-conv-channels`,
até agora, simplificamos todos os nossos exemplos numéricos
trabalhando com apenas uma única entrada e um único canal de saída.
Isso nos permitiu pensar em nossas entradas, *kernels* de convolução,
e saídas, cada um como tensores bidimensionais.

Quando adicionamos canais a isto,
nossas entradas e representações ocultas
ambas se tornam tensores tridimensionais.
Por exemplo, cada imagem de entrada RGB tem a forma $3\times h\times w$.
Referimo-nos a este eixo, com um tamanho de 3, como a dimensão do *canal*.
Nesta seção, daremos uma olhada mais detalhada
em núcleos de convolução com múltiplos canais de entrada e saída.

## Canais de Entrada Múltiplos


Quando os dados de entrada contêm vários canais,
precisamos construir um *kernel* de convolução
com o mesmo número de canais de entrada que os dados de entrada,
para que possa realizar correlação cruzada com os dados de entrada.
Supondo que o número de canais para os dados de entrada seja $c_i$,
o número de canais de entrada do *kernel* de convolução também precisa ser $c_i$. Se a forma da janela do nosso kernel de convolução é $k_h\times k_w$,
então quando $c_i=1$, podemos pensar em nosso kernel de convolução
apenas como um tensor bidimensional de forma $k_h\times k_w$.

No entanto, quando $c_i>1$, precisamos de um kernel
que contém um tensor de forma $k_h\times k_w$ para *cada* canal de entrada. Concatenando estes $c_i$ tensores juntos
produz um kernel de convolução de forma $c_i\times k_h\times k_w$.
Uma vez que o *kernel* de entrada e convolução tem cada um $c_i$ canais,
podemos realizar uma operação de correlação cruzada
no tensor bidimensional da entrada
e o tensor bidimensional do núcleo de convolução
para cada canal, adicionando os resultados $c_i$ juntos
(somando os canais)
para produzir um tensor bidimensional.
Este é o resultado de uma correlação cruzada bidimensional
entre uma entrada multicanal e
um *kernel* de convolução com vários canais de entrada.

Em :numref:`fig_conv_multi_in`, demonstramos um exemplo
de uma correlação cruzada bidimensional com dois canais de entrada.
As partes sombreadas são o primeiro elemento de saída
bem como os elementos tensores de entrada e kernel usados ​​para o cálculo de saída:
$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

![Cálculo de correlação cruzada com 2 canais de entrada.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`


Para ter certeza de que realmente entendemos o que está acontecendo aqui,
podemos implementar operações de correlação cruzada com vários canais de entrada.
Observe que tudo o que estamos fazendo é realizar uma operação de correlação cruzada
por canal e depois somando os resultados.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab mxnet, pytorch
def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

Podemos construir o tensor de entrada `X` e o tensor do kernel` K`
correspondendo aos valores em :numref:`fig_conv_multi_in`
para validar a saída da operação de correlação cruzada.

```{.python .input}
#@tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## Canais de Saída Múltiplos


Independentemente do número de canais de entrada,
até agora acabamos sempre com um canal de saída.
No entanto, como discutimos em :numref:`subsec_why-conv-channels`,
é essencial ter vários canais em cada camada.
Nas arquiteturas de rede neural mais populares,
na verdade, aumentamos a dimensão do canal
à medida que subimos na rede neural,
normalmente reduzindo as amostras para compensar a resolução espacial
para maior *profundidade do canal*.
Intuitivamente, você pode pensar em cada canal
como respondendo a algum conjunto diferente de *features*.
A realidade é um pouco mais complicada do que as interpretações mais ingênuas dessa intuição, uma vez que as representações não são aprendidas de forma independente, mas sim otimizadas para serem úteis em conjunto.
Portanto, pode não ser que um único canal aprenda um detector de bordas, mas sim que alguma direção no espaço do canal corresponde à detecção de bordas.


Denote por $c_i$ e $c_o$ o número
dos canais de entrada e saída, respectivamente,
e sejam $k_h$ e $k_w$ a altura e a largura do *kernel*.
Para obter uma saída com vários canais,
podemos criar um tensor de kernel
da forma $c_i\times k_h\times k_w$
para *cada* canal de saída.
Nós os concatenamos na dimensão do canal de saída,
de modo que a forma do núcleo de convolução
é $c_o\times c_i\times k_h\times k_w$.
Em operações de correlação cruzada,
o resultado em cada canal de saída é calculado
do *kernel* de convolução correspondente a esse canal de saída
e recebe a entrada de todos os canais no tensor de entrada.

Implementamos uma função de correlação cruzada
para calcular a saída de vários canais, conforme mostrado abaixo.

```{.python .input}
#@tab all
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of `K`, and each time, perform
    # cross-correlation operations with input `X`. All of the results are
    # stacked together
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

Construímos um *kernel* de convolução com 3 canais de saída
concatenando o tensor do kernel `K` com` K + 1`
(mais um para cada elemento em `K`) e` K + 2`.

```{.python .input}
#@tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

Below, we perform cross-correlation operations
on the input tensor `X` with the kernel tensor `K`.
Now the output contains 3 channels.
The result of the first channel is consistent
with the result of the previous input tensor `X`
and the multi-input channel,
single-output channel kernel.

```{.python .input}
#@tab all
corr2d_multi_in_out(X, K)
```

## $1\times 1$ Convolutional Layer

At first, a $1 \times 1$ convolution, i.e., $k_h = k_w = 1$,
does not seem to make much sense.
After all, a convolution correlates adjacent pixels.
A $1 \times 1$ convolution obviously does not.
Nonetheless, they are popular operations that are sometimes included
in the designs of complex deep networks.
Let us see in some detail what it actually does.

Because the minimum window is used,
the $1\times 1$ convolution loses the ability
of larger convolutional layers
to recognize patterns consisting of interactions
among adjacent elements in the height and width dimensions.
The only computation of the $1\times 1$ convolution occurs
on the channel dimension.

:numref:`fig_conv_1x1` shows the cross-correlation computation
using the $1\times 1$ convolution kernel
with 3 input channels and 2 output channels.
Note that the inputs and outputs have the same height and width.
Each element in the output is derived
from a linear combination of elements *at the same position*
in the input image.
You could think of the $1\times 1$ convolutional layer
as constituting a fully-connected layer applied at every single pixel location
to transform the $c_i$ corresponding input values into $c_o$ output values.
Because this is still a convolutional layer,
the weights are tied across pixel location.
Thus the $1\times 1$ convolutional layer requires $c_o\times c_i$ weights
(plus the bias).


![The cross-correlation computation uses the $1\times 1$ convolution kernel with 3 input channels and 2 output channels. The input and output have the same height and width.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

Let us check whether this works in practice:
we implement a $1 \times 1$ convolution
using a fully-connected layer.
The only thing is that we need to make some adjustments
to the data shape before and after the matrix multiplication.

```{.python .input}
#@tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    Y = d2l.matmul(K, X)  # Matrix multiplication in the fully-connected layer
    return d2l.reshape(Y, (c_o, h, w))
```

When performing $1\times 1$ convolution,
the above function is equivalent to the previously implemented cross-correlation function `corr2d_multi_in_out`.
Let us check this with some sample data.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
```

```{.python .input}
#@tab all
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## Summary

* Multiple channels can be used to extend the model parameters of the convolutional layer.
* The $1\times 1$ convolutional layer is equivalent to the fully-connected layer, when applied on a per pixel basis.
* The $1\times 1$ convolutional layer is typically used to adjust the number of channels between network layers and to control model complexity.


## Exercises

1. Assume that we have two convolution kernels of size $k_1$ and $k_2$, respectively (with no nonlinearity in between).
    1. Prove that the result of the operation can be expressed by a single convolution.
    1. What is the dimensionality of the equivalent single convolution?
    1. Is the converse true?
1. Assume an input of shape $c_i\times h\times w$ and a convolution kernel of shape $c_o\times c_i\times k_h\times k_w$, padding of $(p_h, p_w)$, and stride of $(s_h, s_w)$.
    1. What is the computational cost (multiplications and additions) for the forward propagation?
    1. What is the memory footprint?
    1. What is the memory footprint for the backward computation?
    1. What is the computational cost for the backpropagation?
1. By what factor does the number of calculations increase if we double the number of input channels $c_i$ and the number of output channels $c_o$? What happens if we double the padding?
1. If the height and width of a convolution kernel is $k_h=k_w=1$, what is the computational complexity of the forward propagation?
1. Are the variables `Y1` and `Y2` in the last example of this section exactly the same? Why?
1. How would you implement convolutions using matrix multiplication when the convolution window is not $1\times 1$?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/69)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/70)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/273)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjAyMzI3MDY2MSwtNDA1NDAzOTA0LC0xMT
M4NTU2NzQsMjIzNjY5NjMzLDEwODQ3OTUxOTcsMTA5NjM5ODc2
NV19
-->
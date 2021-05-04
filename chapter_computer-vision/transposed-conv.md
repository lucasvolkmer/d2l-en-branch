# Convolução Transposta
:label:`sec_transposed_conv`

As camadas que apresentamos até agora para redes neurais convolucionais, incluindo camadas convolucionais (:numref:`sec_conv_layer`) e camadas de pooling (:numref:`sec_pooling`), geralmente reduzem a largura e altura de entrada ou as mantêm inalteradas. Aplicativos como segmentação semântica (:numref:`sec_semantic_segmentation`) e redes adversárias geradoras (:numref:`sec_dcgan`), no entanto, exigem prever valores para cada pixel e, portanto, precisam aumentar a largura e altura de entrada. A convolução transposta, também chamada de convolução fracionada :cite:`Dumoulin.Visin.2016` ou deconvolução :cite:`Long.Shelhamer.Darrell.2015`, serve a este propósito.

```{.python .input}
from mxnet import np, npx, init
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from d2l import torch as d2l
```

## Convolução Transposta 2D Básica

Vamos considerar um caso básico em que os canais de entrada e saída são 1, com 0 preenchimento e 1 passo. :numref:`fig_trans_conv` ilustra como a convolução transposta com um *kernel* $2\times 2$ é calculada na matriz de entrada $2\times 2$.

![Camada de convolução transposta com um *kernel* $2\times 2$.](../img/trans-conv.svg)
:label:`fig_trans_conv`

Podemos implementar essa operação fornecendo o *kernel* da matriz $K$ e a entrada da matriz $X$.

```{.python .input}
#@tab all
def trans_conv(X, K):
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```


Lembre-se de que a convolução calcula os resultados por `Y[i, j] = (X[i: i + h, j: j + w] * K).sum()` (consulte `corr2d` em :numref:`sec_conv_layer`), que resume os valores de entrada por meio do *kernel*. Enquanto a convolução transposta transmite valores de entrada por meio do *kernel*, o que resulta em uma forma de saída maior.

Verifique os resultados em :numref:`fig_trans_conv`.

```{.python .input}
#@tab all
X = d2l.tensor([[0., 1], [2, 3]])
K = d2l.tensor([[0., 1], [2, 3]])
trans_conv(X, K)
```

:begin_tab:`mxnet`
Ou podemos usar `nn.Conv2DTranspose` para obter os mesmos resultados. Como `nn.Conv2D`, tanto a entrada quanto o *kernel* devem ser tensores 4-D.
:end_tab:

:begin_tab:`pytorch`
Ou podemos usar `nn.ConvTranspose2d` para obter os mesmos resultados. Como `nn.Conv2d`, tanto a entrada quanto o *kernel* devem ser tensores 4-D.
:end_tab:

```{.python .input}
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```

## Preenchimento, Passos e Canais

Aplicamos elementos de preenchimento à entrada em convolução, enquanto eles são aplicados à saída em convolução transposta. Um preenchimento $1\times 1$ significa que primeiro calculamos a saída como normal e, em seguida, removemos as primeiras/últimas linhas e colunas.

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
```

Similarly, strides are applied to outputs as well.

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
```

The multi-channel extension of the transposed convolution is the same as the convolution. When the input has multiple channels, denoted by $c_i$, the transposed convolution assigns a $k_h\times k_w$ kernel matrix to each input channel. If the output has a channel size $c_o$, then we have a $c_i\times k_h\times k_w$ kernel for each output channel.


As a result, if we feed $X$ into a convolutional layer $f$ to compute $Y=f(X)$ and create a transposed convolution layer $g$ with the same hyperparameters as $f$ except for the output channel set to be the channel size of $X$, then $g(Y)$ should has the same shape as $X$. Let us verify this statement.

```{.python .input}
X = np.random.uniform(size=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```

## Analogy to Matrix Transposition

The transposed convolution takes its name from the matrix transposition. In fact, convolution operations can also be achieved by matrix multiplication. In the example below, we define a $3\times 3$ input $X$ with a $2\times 2$ kernel $K$, and then use `corr2d` to compute the convolution output.

```{.python .input}
#@tab all
X = d2l.arange(9.0).reshape(3, 3)
K = d2l.tensor([[0, 1], [2, 3]])
Y = d2l.corr2d(X, K)
Y
```

Next, we rewrite convolution kernel $K$ as a matrix $W$. Its shape will be $(4, 9)$, where the $i^\mathrm{th}$ row present applying the kernel to the input to generate the $i^\mathrm{th}$ output element.

```{.python .input}
#@tab all
def kernel2matrix(K):
    k, W = d2l.zeros(5), d2l.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```

Then the convolution operator can be implemented by matrix multiplication with proper reshaping.

```{.python .input}
Y == np.dot(W, X.reshape(-1)).reshape(2, 2)
```

```{.python .input}
#@tab pytorch
Y == torch.mv(W, X.reshape(-1)).reshape(2, 2)
```

We can implement transposed convolution as a matrix multiplication as well by reusing `kernel2matrix`. To reuse the generated $W$, we construct a $2\times 2$ input, so the corresponding weight matrix will have a shape $(9, 4)$, which is $W^\top$. Let us verify the results.

```{.python .input}
X = np.array([[0, 1], [2, 3]])
Y = trans_conv(X, K)
Y == np.dot(W.T, X.reshape(-1)).reshape(3, 3)
```

```{.python .input}
#@tab pytorch
X = torch.tensor([[0.0, 1], [2, 3]])
Y = trans_conv(X, K)
Y == torch.mv(W.T, X.reshape(-1)).reshape(3, 3)
```

## Summary

* Compared to convolutions that reduce inputs through kernels, transposed convolutions broadcast inputs.
* If a convolution layer reduces the input width and height by $n_w$ and $h_h$ time, respectively. Then a transposed convolution layer with the same kernel sizes, padding and strides will increase the input width and height by $n_w$ and $n_h$, respectively.
* We can implement convolution operations by the matrix multiplication, the corresponding transposed convolutions can be done by transposed matrix multiplication.

## Exercises

1. Is it efficient to use matrix multiplication to implement convolution operations? Why?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/376)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1450)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0MzgzMjU5MzUsMTU0MjgyNDE5Ml19
-->
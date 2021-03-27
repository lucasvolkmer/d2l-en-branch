# Preenchimento e Saltos
:label:`sec_padding`



No exemplo anterior de :numref:`fig_correlation`,
nossa entrada tinha altura e largura de 3
e nosso núcleo de convolução tinha altura e largura de 2,
produzindo uma representação de saída com dimensão $2\times2$.
Como generalizamos em :numref:`sec_conv_layer`,
assumindo que
a forma de entrada é $n_h\times n_w$
e a forma do kernel de convolução é $k_h\times k_w$,
então a forma de saída será
$(n_h-k_h+1) \times (n_w-k_w+1)$.
Portanto, a forma de saída da camada convolucional
é determinada pela forma da entrada
e a forma do núcleo de convolução.

Em vários casos, incorporamos técnicas,
incluindo preenchimento e convoluções com saltos,
que afetam o tamanho da saída.
Como motivação, note que uma vez que os *kernels* geralmente
têm largura e altura maiores que $1$,
depois de aplicar muitas convoluções sucessivas,
tendemos a acabar com resultados que são
consideravelmente menor do que nossa entrada.
Se começarmos com uma imagem de $240 \times 240$ pixels,
$10$ camadas de $5 \times 5$ convoluções
reduzem a imagem para $200 \times 200$ pixels,
cortando $30 \%$ da imagem e com ela
obliterando qualquer informação interessante
nos limites da imagem original.
*Preenchimento* é a ferramenta mais popular para lidar com esse problema.

In other cases, we may want to reduce the dimensionality drastically,
e.g., if we find the original input resolution to be unwieldy.
*Strided convolutions* are a popular technique that can help in these instances.

## Preenchimento

Conforme descrito acima, um problema complicado ao aplicar camadas convolucionais
é que tendemos a perder pixels no perímetro de nossa imagem.
Uma vez que normalmente usamos pequenos *kernels*,
para qualquer convolução dada,
podemos perder apenas alguns pixels,
mas isso pode somar conforme aplicamos
muitas camadas convolucionais sucessivas.
Uma solução direta para este problema
é adicionar pixels extras de preenchimento ao redor do limite de nossa imagem de entrada,
aumentando assim o tamanho efetivo da imagem.
Normalmente, definimos os valores dos pixels extras para zero.
Em :numref:`img_conv_pad`, preenchemos uma entrada $3 \times 3$,
aumentando seu tamanho para $5 \times 5$.
A saída correspondente então aumenta para uma matriz $4 \times 4$
As partes sombreadas são o primeiro elemento de saída, bem como os elementos tensores de entrada e kernel usados para o cálculo de saída: $0\times0+0\times1+0\times2+0\times3=0$.

![Correlação cruzada bidimensional com preenchimento.](../img/conv-pad.svg)
:label:`img_conv_pad`

Em geral, se adicionarmos um total de $p_h$ linhas de preenchimento
(cerca de metade na parte superior e metade na parte inferior)
e um total de $p_w$ colunas de preenchimento
(cerca de metade à esquerda e metade à direita),
a forma de saída será

$$(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1).$$


Isso significa que a altura e largura da saída
aumentará em $p_h$ e $p_w$, respectivamente.

Em muitos casos, queremos definir $p_h=k_h-1$ e $p_w=k_w-1$
para dar à entrada e saída a mesma altura e largura.
Isso tornará mais fácil prever a forma de saída de cada camada
ao construir a rede.
Supondo que $k_h$ seja estranho aqui,
vamos preencher $p_h/2$ linhas em ambos os lados da altura.
Se $k_h$ for par, uma possibilidade é
juntar $\lceil p_h/2\rceil$ linhas no topo da entrada
e $\lfloor p_h/2\rfloor$ linhas na parte inferior.
Vamos preencher ambos os lados da largura da mesma maneira.


CNNs geralmente usam *kernels* de convolução
com valores de altura e largura ímpares, como 1, 3, 5 ou 7.
Escolher tamanhos ímpares de *kernel* tem o benefício
que podemos preservar a dimensionalidade espacial
enquanto preenche com o mesmo número de linhas na parte superior e inferior,
e o mesmo número de colunas à esquerda e à direita.

Além disso, esta prática de usar *kernels* estranhos
e preenchimento para preservar precisamente a dimensionalidade
oferece um benefício administrativo.
Para qualquer tensor bidimensional `X`,
quando o tamanho do *kernel* é estranho
e o número de linhas e colunas de preenchimento
em todos os lados são iguais,
produzindo uma saída com a mesma altura e largura da entrada,
sabemos que a saída `Y [i, j]` é calculada
por correlação cruzada do kernel de entrada e convolução
com a janela centralizada em `X [i, j]`.

No exemplo a seguir, criamos uma camada convolucional bidimensional
com altura e largura de 3
e aplique 1 pixel de preenchimento em todos os lados.
Dada uma entrada com altura e largura de 8,
descobrimos que a altura e a largura da saída também é 8.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# For convenience, we define a function to calculate the convolutional layer.
# This function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])

# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return tf.reshape(Y, Y.shape[1:3])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

When the height and width of the convolution kernel are different,
we can make the output and input have the same height and width
by setting different padding numbers for height and width.

```{.python .input}
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape
```

## Stride

When computing the cross-correlation,
we start with the convolution window
at the top-left corner of the input tensor,
and then slide it over all locations both down and to the right.
In previous examples, we default to sliding one element at a time.
However, sometimes, either for computational efficiency
or because we wish to downsample,
we move our window more than one element at a time,
skipping the intermediate locations.

We refer to the number of rows and columns traversed per slide as the *stride*.
So far, we have used strides of 1, both for height and width.
Sometimes, we may want to use a larger stride.
:numref:`img_conv_stride` shows a two-dimensional cross-correlation operation
with a stride of 3 vertically and 2 horizontally.
The shaded portions are the output elements as well as the input and kernel tensor elements used for the output computation: $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$.
We can see that when the second element of the first column is outputted,
the convolution window slides down three rows.
The convolution window slides two columns to the right
when the second element of the first row is outputted.
When the convolution window continues to slide two columns to the right on the input,
there is no output because the input element cannot fill the window
(unless we add another column of padding).

![Cross-correlation with strides of 3 and 2 for height and width, respectively.](../img/conv-stride.svg)
:label:`img_conv_stride`

In general, when the stride for the height is $s_h$
and the stride for the width is $s_w$, the output shape is

$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$

If we set $p_h=k_h-1$ and $p_w=k_w-1$,
then the output shape will be simplified to
$\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$.
Going a step further, if the input height and width
are divisible by the strides on the height and width,
then the output shape will be $(n_h/s_h) \times (n_w/s_w)$.

Below, we set the strides on both the height and width to 2,
thus halving the input height and width.

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

Next, we will look at a slightly more complicated example.

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

For the sake of brevity, when the padding number
on both sides of the input height and width are $p_h$ and $p_w$ respectively, we call the padding $(p_h, p_w)$.
Specifically, when $p_h = p_w = p$, the padding is $p$.
When the strides on the height and width are $s_h$ and $s_w$, respectively,
we call the stride $(s_h, s_w)$.
Specifically, when $s_h = s_w = s$, the stride is $s$.
By default, the padding is 0 and the stride is 1.
In practice, we rarely use inhomogeneous strides or padding,
i.e., we usually have $p_h = p_w$ and $s_h = s_w$.

## Summary

* Padding can increase the height and width of the output. This is often used to give the output the same height and width as the input.
* The stride can reduce the resolution of the output, for example reducing the height and width of the output to only $1/n$ of the height and width of the input ($n$ is an integer greater than $1$).
* Padding and stride can be used to adjust the dimensionality of the data effectively.

## Exercises

1. For the last example in this section, use mathematics to calculate the output shape to see if it is consistent with the experimental result.
1. Try other padding and stride combinations on the experiments in this section.
1. For audio signals, what does a stride of 2 correspond to?
1. What are the computational benefits of a stride larger than 1?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/272)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2MDkxNzUxMjMsLTE0ODMxNjY2MTAsMT
U1Njk1NzM0OCwxOTc0NTA0ODkyLC05MDQ4MzczNl19
-->
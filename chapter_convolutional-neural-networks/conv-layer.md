# Convolução para Imagens
:label:`sec_conv_layer`

Agora que entendemos como as camadas convolucionais funcionam na teoria,
estamos prontos para ver como eles funcionam na prática.
Com base na nossa motivação de redes neurais convolucionais
como arquiteturas eficientes para explorar a estrutura em dados de imagem,
usamos imagens como nosso exemplo de execução.


## A Operação de Correlação Cruzada


Lembre-se de que, estritamente falando, as camadas convolucionais
são um nome impróprio, uma vez que as operações que elas expressam
são descritos com mais precisão como correlações cruzadas.
Com base em nossas descrições de camadas convolucionais em :numref:`sec_why-conv`,
em tal camada, um tensor de entrada
e um tensor de *kernel* são combinados
para produzir um tensor de saída por meio de uma operação de correlação cruzada.

Vamos ignorar os canais por enquanto e ver como isso funciona
com dados bidimensionais e representações ocultas.
Em :numref:`fig_correlation`,
a entrada é um tensor bidimensional
com altura de 3 e largura de 3.
Marcamos a forma do tensor como $3 \times 3$ or ($3$, $3$).
A altura e a largura do *kernel* são 2.
A forma da *janela do kernel* (ou *janela de convolução*)
é dada pela altura e largura do *kernel*
(aqui é $2 \times 2$).

![Operação de correlação cruzada bidimensional. As partes sombreadas são o primeiro elemento de saída, bem como os elementos tensores de entrada e *kernel* usados para o cálculo de saída: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

Na operação de correlação cruzada bidimensional,
começamos com a janela de convolução posicionada
no canto superior esquerdo do tensor de entrada
e o deslizamos pelo tensor de entrada,
ambos da esquerda para a direita e de cima para baixo.
Quando a janela de convolução desliza para uma determinada posição,
o subtensor de entrada contido nessa janela
e o tensor do *kernel* são multiplicados elemento a elemento
e o tensor resultante é resumido
produzindo um único valor escalar.
Este resultado fornece o valor do tensor de saída
no local correspondente.
Aqui, o tensor de saída tem uma altura de 2 e largura de 2
e os quatro elementos são derivados de
a operação de correlação cruzada bidimensional:

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

Observe que ao longo de cada eixo, o tamanho da saída
é ligeiramente menor que o tamanho de entrada.
Como o *kernel* tem largura e altura maiores que um,
só podemos calcular corretamente a correlação cruzada
para locais onde o *kernel* se encaixa totalmente na imagem,
o tamanho da saída é dado pelo tamanho da entrada $n_h \times n_w$
menos o tamanho do *kernel* de convolução $k_h \times k_w$
através da

$$(n_h-k_h+1) \times (n_w-k_w+1).$$

Este é o caso, pois precisamos de espaço suficiente
para "deslocar" o *kernel* de convolução na imagem.
Mais tarde, veremos como manter o tamanho inalterado
preenchendo a imagem com zeros em torno de seu limite
para que haja espaço suficiente para mudar o *kernel*.
Em seguida, implementamos este processo na função `corr2d`,
que aceita um tensor de entrada `X` e um tensor de *kernel* `K`
e retorna um tensor de saída `Y`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
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
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

Podemos construir o tensor de entrada `X` e o tensor do kernel` K`
from :numref:`fig_correlation`
para validar o resultado da implementação acima
da operação de correlação cruzada bidimensional.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## Camadas Convolucionais


Uma camada convolucional correlaciona a entrada e o *kernel*
e adiciona um *bias* escalar para produzir uma saída.
Os dois parâmetros de uma camada convolucional
são o *kernel* e o *bias* escalar.
Ao treinar modelos com base em camadas convolucionais,
normalmente inicializamos os *kernels* aleatoriamente,
assim como faríamos com uma camada totalmente conectada.

Agora estamos prontos para implementar uma camada convolucional bidimensional
com base na função `corr2d` definida acima.
Na função construtora `__init__`,
declaramos `weight` e` bias` como os dois parâmetros do modelo.
A função de propagação direta
chama a função `corr2d` e adiciona o viés.

```{.python .input}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
#@tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
#@tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

Na convolução
$h \times w$ 
ou um *kernel* de convolução $h \times w$
a altura e a largura do *kernel* de convolução são $h$ e $w$, respectivamente.
Também nos referimos a
uma camada convolucional com um kernel de convolução $h \times w$
simplesmente como uma camada convolucional $h \times w$


## Detecção de Borda de Objeto em Imagens

Vamos analisar uma aplicação simples de uma camada convolucional:
detectar a borda de um objeto em uma imagem
encontrando a localização da mudança de pixel.
Primeiro, construímos uma "imagem" de $6\times 8$ pixels.
As quatro colunas do meio são pretas (0) e as demais são brancas (1).

```{.python .input}
#@tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
#@tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

Em seguida, construímos um kernel `K` com uma altura de 1 e uma largura de 2.
Quando realizamos a operação de correlação cruzada com a entrada,
se os elementos horizontalmente adjacentes forem iguais,
a saída é 0. Caso contrário, a saída é diferente de zero.

```{.python .input}
#@tab all
K = d2l.tensor([[1.0, -1.0]])
```

Estamos prontos para realizar a operação de correlação cruzada
com os argumentos `X` (nossa entrada) e` K` (nosso kernel).
Como você pode ver, detectamos 1 para a borda do branco ao preto
e -1 para a borda do preto ao branco.
Todas as outras saídas assumem o valor 0.

```{.python .input}
#@tab all
Y = corr2d(X, K)
Y
```

Agora podemos aplicar o kernel à imagem transposta.
Como esperado, ele desaparece. O kernel `K` detecta apenas bordas verticais.

```{.python .input}
#@tab all
corr2d(d2l.transpose(X), K)
```

## Aprendendo um Kernel


Projetar um detector de borda por diferenças finitas `[1, -1]` é legal
se sabemos que é exatamente isso que estamos procurando.
No entanto, quando olhamos para *kernels* maiores,
e considere camadas sucessivas de convoluções,
pode ser impossível especificar
exatamente o que cada filtro deve fazer manualmente.

Agora vamos ver se podemos aprender o *kernel* que gerou `Y` de` X`
olhando apenas para os pares de entrada--saída.
Primeiro construímos uma camada convolucional
e inicializamos seu *kernel* como um tensor aleatório.
A seguir, em cada iteração, usaremos o erro quadrático
para comparar `Y` com a saída da camada convolucional.
Podemos então calcular o gradiente para atualizar o *kernel*.
Por uma questão de simplicidade,
na sequência
nós usamos a classe embutida
para camadas convolucionais bidimensionais
e ignorar o *bias*.

```{.python .input}
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Update the kernel
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'batch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
#@tab pytorch
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
#@tab tensorflow
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(3e-2, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'batch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

Observe que o erro caiu para um valor pequeno após 10 iterações. Agora daremos uma olhada no tensor do *kernel* que aprendemos.

```{.python .input}
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
#@tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
#@tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

Indeed, the learned kernel tensor is remarkably close
to the kernel tensor `K` we defined earlier.

## Correlação Cruzada e Convolução


Lembre-se de nossa observação de :numref:`sec_why-conv` da correspondência
entre as operações de correlação cruzada e convolução.
Aqui, vamos continuar a considerar as camadas convolucionais bidimensionais.
E se essas camadas
realizar operações de convolução estritas
conforme definido em :eqref:`eq_2d-conv-discrete`
em vez de correlações cruzadas?
Para obter a saída da operação de *convolução* estrita, precisamos apenas inverter o tensor do *kerne*l bidimensional tanto horizontal quanto verticalmente e, em seguida, executar a operação de *correlação cruzada* com o tensor de entrada.

É digno de nota que, uma vez que os *kernels* são aprendidos a partir de dados no aprendizado profundo,
as saídas das camadas convolucionais permanecem inalteradas
não importa se tais camadas
executam
as operações de convolução estrita
ou as operações de correlação cruzada.

To illustrate this, suppose that a convolutional layer performs *cross-correlation* and learns the kernel in :numref:`fig_correlation`, which is denoted as the matrix $\mathbf{K}$ here.
Assuming that other conditions remain unchanged, 
when this layer performs strict *convolution* instead,
the learned kernel $\mathbf{K}'$ will be the same as $\mathbf{K}$
after $\mathbf{K}'$ is 
flipped both horizontally and vertically.
That is to say,
when the convolutional layer
performs strict *convolution*
for the input in :numref:`fig_correlation`
and $\mathbf{K}'$,
the same output in :numref:`fig_correlation`
(cross-correlation of the input and $\mathbf{K}$)
will be obtained.

In keeping with standard terminology with deep learning literature,
we will continue to refer to the cross-correlation operation
as a convolution even though, strictly-speaking, it is slightly different.
Besides,
we use the term *element* to refer to
an entry (or component) of any tensor representing a layer representation or a convolution kernel.


## Feature Map and Receptive Field

As described in :numref:`subsec_why-conv-channels`,
the convolutional layer output in
:numref:`fig_correlation`
is sometimes called a *feature map*,
as it can be regarded as
the learned representations (features)
in the spatial dimensions (e.g., width and height)
to the subsequent layer.
In CNNs,
for any element $x$ of some layer,
its *receptive field* refers to
all the elements (from all the previous layers)
that may affect the calculation of $x$
during the forward propagation.
Note that the receptive field
may be larger than the actual size of the input.

Let us continue to use :numref:`fig_correlation` to explain the receptive field.
Given the $2 \times 2$ convolution kernel,
the receptive field of the shaded output element (of value $19$)
is
the four elements in the shaded portion of the input.
Now let us denote the $2 \times 2$
output as $\mathbf{Y}$
and consider a deeper CNN
with an additional $2 \times 2$ convolutional layer that takes $\mathbf{Y}$
as its input, outputting
a single element $z$.
In this case,
the receptive field of $z$
on $\mathbf{Y}$ includes all the four elements of $\mathbf{Y}$,
while
the receptive field
on the input includes all the nine input elements.
Thus, 
when any element in a feature map
needs a larger receptive field
to detect input features over a broader area,
we can build a deeper network.




## Summary

* The core computation of a two-dimensional convolutional layer is a two-dimensional cross-correlation operation. In its simplest form, this performs a cross-correlation operation on the two-dimensional input data and the kernel, and then adds a bias.
* We can design a kernel to detect edges in images.
* We can learn the kernel's parameters from data.
* With kernels learned from data, the outputs of convolutional layers remain unaffected regardless of such layers' performed operations (either strict convolution or cross-correlation).
* When any element in a feature map needs a larger receptive field to detect broader features on the input, a deeper network can be considered.


## Exercises

1. Construct an image `X` with diagonal edges.
    1. What happens if you apply the kernel `K` in this section to it?
    1. What happens if you transpose `X`?
    1. What happens if you transpose `K`?
1. When you try to automatically find the gradient for the `Conv2D` class we created, what kind of error message do you see?
1. How do you represent a cross-correlation operation as a matrix multiplication by changing the input and kernel tensors?
1. Design some kernels manually.
    1. What is the form of a kernel for the second derivative?
    1. What is the kernel for an integral?
    1. What is the minimum size of a kernel to obtain a derivative of degree $d$?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/271)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2Nzc4MzEzNDcsLTE5ODg2NDM0NzUsLT
E4MTg0NzQ3MzUsLTE0OTQ2MDExMTIsLTI0MjE5ODI3NywtNjc0
ODUxNzY5LDU1OTEzNTQ1MCwxOTg0OTc5Nzk3XX0=
-->
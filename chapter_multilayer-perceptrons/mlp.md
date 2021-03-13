# *Perceptrons* Multicamada
:label:`sec_mlp`

Em :numref:`chap_linear`,, apresentamos
regressão *softmax* (:numref:`sec_softmax`),
implementando o algoritmo do zero
(:numref:`sec_softmax_scratch`) e usando APIs de alto nível
(:numref:`sec_softmax_concise`),
e classificadores de treinamento para reconhecer
10 categorias de roupas a partir de imagens de baixa resolução.
Ao longo do caminho, aprendemos como organizar dados,
coagir nossos resultados em uma distribuição de probabilidade válida,
aplicar uma função de perda apropriada,
e minimizá-la em relação aos parâmetros do nosso modelo.
Agora que dominamos essa mecânica
no contexto de modelos lineares simples,
podemos lançar nossa exploração de redes neurais profundas,
a classe comparativamente rica de modelos
com o qual este livro se preocupa principalmente.

## Camadas Ocultas

Descrevemos a transformação afim em
:numref:`subsec_linear_model`,
que é uma transformação linear adicionada por um *bias.
Para começar, relembre a arquitetura do modelo
correspondendo ao nosso exemplo de regressão *softmax*,
ilustrado em :numref:`fig_softmaxreg`.
Este modelo mapeou nossas entradas diretamente para nossas saídas
por meio de uma única transformação afim,
seguido por uma operação *softmax*.
Se nossos *labels* realmente estivessem relacionados
aos nossos dados de entrada por uma transformação afim,
então esta abordagem seria suficiente.
Mas a linearidade nas transformações afins é uma suposição *forte*.

### Modelos Lineares Podem Dar Errado


Por exemplo, linearidade implica a *mais fraca*
suposição de *monotonicidade*:
que qualquer aumento em nosso recurso deve
sempre causar um aumento na saída do nosso modelo
(se o peso correspondente for positivo),
ou sempre causa uma diminuição na saída do nosso modelo
(se o peso correspondente for negativo).
Às vezes, isso faz sentido.
Por exemplo, se estivéssemos tentando prever
se um indivíduo vai pagar um empréstimo,
podemos razoavelmente imaginar que mantendo tudo o mais igual,
um candidato com uma renda maior
sempre estaria mais propenso a retribuir
do que um com uma renda mais baixa.
Embora monotônico, esse relacionamento provavelmente
não está linearmente associado à probabilidade de
reembolso. Um aumento na receita de 0 a 50 mil
provavelmente corresponde a um aumento maior
em probabilidade de reembolso
do que um aumento de 1 milhão para 1,05 milhão.
Uma maneira de lidar com isso pode ser pré-processar
nossos dados de forma que a linearidade se torne mais plausível,
digamos, usando o logaritmo da receita como nosso recurso.


Observe que podemos facilmente encontrar exemplos
que violam a monotonicidade.
Digamos, por exemplo, que queremos prever a probabilidade
de morte com base na temperatura corporal.
Para indivíduos com temperatura corporal
acima de 37 ° C (98,6 ° F),
temperaturas mais altas indicam maior risco.
No entanto, para indivíduos com temperatura corporal
abaixo de 37 ° C, temperaturas mais altas indicam risco menor!
Também neste caso, podemos resolver o problema
com algum pré-processamento inteligente.
Ou seja, podemos usar a distância de 37 ° C como nossa *feature*.


Mas que tal classificar imagens de cães e gatos?
Aumentar a intensidade
do pixel no local (13, 17) deveria
sempre aumentar (ou sempre diminuir)
a probabilidade de que a imagem retrate um cachorro?
A confiança em um modelo linear corresponde à implícita
suposição de que o único requisito
para diferenciar gatos vs. cães é avaliar
o brilho de pixels individuais.
Esta abordagem está fadada ao fracasso em um mundo
onde inverter uma imagem preserva a categoria.


E ainda, apesar do aparente absurdo da linearidade aqui,
em comparação com nossos exemplos anteriores,
é menos óbvio que poderíamos resolver o problema
com uma correção de pré-processamento simples.
Isso ocorre porque o significado de qualquer pixel
depende de maneiras complexas de seu contexto
(os valores dos pixels circundantes).
Embora possa existir uma representação de nossos dados
isso levaria em consideração
as interações relevantes entre nossas características,
no topo das quais um modelo linear seria adequado, nós
simplesmente não sabemos como calculá-lo à mão.
Com redes neurais profundas, usamos dados observacionais
para aprender conjuntamente uma representação por meio de camadas ocultas
e um preditor linear que atua sobre essa representação.


### Incorporando Camadas Ocultas


Podemos superar essas limitações dos modelos lineares
e lidar com uma classe mais geral de funções
incorporando uma ou mais camadas ocultas.
A maneira mais fácil de fazer isso é empilhar
muitas camadas totalmente conectadas umas sobre as outras.
Cada camada alimenta a camada acima dela,
até gerarmos resultados.
Podemos pensar nas primeiras $L-1$ camadas
como nossa representação e a camada final
como nosso preditor linear.
Esta arquitetura é comumente chamada
um *perceptron multicamadas*,
frequentemente abreviado como *MLP*.
Abaixo, representamos um MLP em diagrama (:numref:`fig_mlp`).

![Um MLP com uma camada oculta de 5 unidades ocultas. ](../ img / mlp.svg) :label:`fig_mlp`

Este MLP tem 4 entradas, 3 saídas,
e sua camada oculta contém 5 unidades ocultas.
Uma vez que a camada de entrada não envolve nenhum cálculo,
produzindo saídas com esta rede
requer a implementação dos cálculos
para as camadas ocultas e de saída;
assim, o número de camadas neste MLP é 2.
Observe que essas camadas estão totalmente conectadas.
Cada entrada influencia cada neurônio na camada oculta,
e cada um deles, por sua vez, influencia
cada neurônio na camada de saída.
No entanto, conforme sugerido por :numref:`subsec_parameterization-cost-fc-layers`,
o custo de parametrização de MLPs
com camadas totalmente conectadas
pode ser proibitivamente alto,
o que pode motivar
compensação entre o salvamento do parâmetro e a eficácia do modelo, mesmo sem alterar o tamanho de entrada ou saída :cite:`Zhang.Tay.Zhang.ea.2021`.


### De Linear Para não Linear


Como antes, pela matriz $\mathbf{X} \in \mathbb{R}^{n \times d}$,
denotamos um *minibatch* de $n$ exemplos em que cada exemplo tem $d$ entradas (*features*).
Para um MLP de uma camada oculta, cuja camada oculta tem $h$ unidades ocultas,
denotamos por $\mathbf{H} \in \mathbb{R}^{n \times h}$
as saídas da camada oculta, que são
*representações ocultas*.
Em matemática ou código, $\mathbf{H}$ também é conhecido como uma *variável de camada oculta* ou uma *variável oculta*.
Uma vez que as camadas ocultas e de saída estão totalmente conectadas,
temos pesos de camada oculta $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ e *bias* $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$
e pesos da camada de saída $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ e *bias* $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$.
Formalmente, calculamos as saídas $\mathbf{O} \in \mathbb{R}^{n \times q}$
do MLP de uma camada oculta da seguinte maneira:

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$



Note that after adding the hidden layer,
our model now requires us to track and update
additional sets of parameters.
So what have we gained in exchange?
You might be surprised to find out
that---in the model defined above---*we
gain nothing for our troubles*!
The reason is plain.
The hidden units above are given by
an affine function of the inputs,
and the outputs (pre-softmax) are just
an affine function of the hidden units.
An affine function of an affine function
is itself an affine function.
Moreover, our linear model was already
capable of representing any affine function.


We can view the equivalence formally
by proving that for any values of the weights,
we can just collapse out the hidden layer,
yielding an equivalent single-layer model with parameters
$\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ and $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$:

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$


In order to realize the potential of multilayer architectures,
we need one more key ingredient: a
nonlinear *activation function* $\sigma$
to be applied to each hidden unit
following the affine transformation.
The outputs of activation functions
(e.g., $\sigma(\cdot)$)
are called *activations*.
In general, with activation functions in place,
it is no longer possible to collapse our MLP into a linear model:


$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

Since each row in $\mathbf{X}$ corresponds to an example in the minibatch,
with some abuse of notation, we define the nonlinearity
$\sigma$ to apply to its inputs in a rowwise fashion,
i.e., one example at a time.
Note that we used the notation for softmax
in the same way to denote a rowwise operation in :numref:`subsec_softmax_vectorization`.
Often, as in this section, the activation functions
that we apply to hidden layers are not merely rowwise,
but elementwise.
That means that after computing the linear portion of the layer,
we can calculate each activation
without looking at the values taken by the other hidden units.
This is true for most activation functions.


To build more general MLPs, we can continue stacking
such hidden layers,
e.g., $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$
and $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$,
one atop another, yielding ever more expressive models.

### Universal Approximators

MLPs can capture complex interactions
among our inputs via their hidden neurons,
which depend on the values of each of the inputs.
We can easily design hidden nodes
to perform arbitrary computation,
for instance, basic logic operations on a pair of inputs.
Moreover, for certain choices of the activation function,
it is widely known that MLPs are universal approximators.
Even with a single-hidden-layer network,
given enough nodes (possibly absurdly many),
and the right set of weights,
we can model any function,
though actually learning that function is the hard part.
You might think of your neural network
as being a bit like the C programming language.
The language, like any other modern language,
is capable of expressing any computable program.
But actually coming up with a program
that meets your specifications is the hard part.

Moreover, just because a single-hidden-layer network
*can* learn any function
does not mean that you should try
to solve all of your problems
with single-hidden-layer networks.
In fact, we can approximate many functions
much more compactly by using deeper (vs. wider) networks.
We will touch upon more rigorous arguments in subsequent chapters.


## Activation Functions
:label:`subsec_activation-functions`

Activation functions decide whether a neuron should be activated or not by
calculating the weighted sum and further adding bias with it.
They are differentiable operators to transform input signals to outputs,
while most of them add non-linearity.
Because activation functions are fundamental to deep learning,
(**let us briefly survey some common activation functions**).

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

### ReLU Function

The most popular choice,
due to both simplicity of implementation and
its good performance on a variety of predictive tasks,
is the *rectified linear unit* (*ReLU*).
[**ReLU provides a very simple nonlinear transformation**].
Given an element $x$, the function is defined
as the maximum of that element and $0$:

$$\operatorname{ReLU}(x) = \max(x, 0).$$

Informally, the ReLU function retains only positive
elements and discards all negative elements
by setting the corresponding activations to 0.
To gain some intuition, we can plot the function.
As you can see, the activation function is piecewise linear.

```{.python .input}
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

When the input is negative,
the derivative of the ReLU function is 0,
and when the input is positive,
the derivative of the ReLU function is 1.
Note that the ReLU function is not differentiable
when the input takes value precisely equal to 0.
In these cases, we default to the left-hand-side
derivative and say that the derivative is 0 when the input is 0.
We can get away with this because
the input may never actually be zero.
There is an old adage that if subtle boundary conditions matter,
we are probably doing (*real*) mathematics, not engineering.
That conventional wisdom may apply here.
We plot the derivative of the ReLU function plotted below.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

The reason for using ReLU is that
its derivatives are particularly well behaved:
either they vanish or they just let the argument through.
This makes optimization better behaved
and it mitigated the well-documented problem
of vanishing gradients that plagued
previous versions of neural networks (more on this later).

Note that there are many variants to the ReLU function,
including the *parameterized ReLU* (*pReLU*) function :cite:`He.Zhang.Ren.ea.2015`.
This variation adds a linear term to ReLU,
so some information still gets through,
even when the argument is negative:

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### Sigmoid Function

[**The *sigmoid function* transforms its inputs**],
for which values lie in the domain $\mathbb{R}$,
(**to outputs that lie on the interval (0, 1).**)
For that reason, the sigmoid is
often called a *squashing function*:
it squashes any input in the range (-inf, inf)
to some value in the range (0, 1):

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

In the earliest neural networks, scientists
were interested in modeling biological neurons
which either *fire* or *do not fire*.
Thus the pioneers of this field,
going all the way back to McCulloch and Pitts,
the inventors of the artificial neuron,
focused on thresholding units.
A thresholding activation takes value 0
when its input is below some threshold
and value 1 when the input exceeds the threshold.


When attention shifted to gradient based learning,
the sigmoid function was a natural choice
because it is a smooth, differentiable
approximation to a thresholding unit.
Sigmoids are still widely used as
activation functions on the output units,
when we want to interpret the outputs as probabilities
for binary classification problems
(you can think of the sigmoid as a special case of the softmax).
However, the sigmoid has mostly been replaced
by the simpler and more easily trainable ReLU
for most use in hidden layers.
In later chapters on recurrent neural networks,
we will describe architectures that leverage sigmoid units
to control the flow of information across time.

Below, we plot the sigmoid function.
Note that when the input is close to 0,
the sigmoid function approaches
a linear transformation.

```{.python .input}
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

The derivative of the sigmoid function is given by the following equation:

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$


The derivative of the sigmoid function is plotted below.
Note that when the input is 0,
the derivative of the sigmoid function
reaches a maximum of 0.25.
As the input diverges from 0 in either direction,
the derivative approaches 0.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

### Tanh Function

Like the sigmoid function, [**the tanh (hyperbolic tangent)
function also squashes its inputs**],
transforming them into elements on the interval (**between -1 and 1**):

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

We plot the tanh function below.
Note that as the input nears 0, the tanh function approaches a linear transformation. Although the shape of the function is similar to that of the sigmoid function, the tanh function exhibits point symmetry about the origin of the coordinate system.

```{.python .input}
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

The derivative of the tanh function is:

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

The derivative of tanh function is plotted below.
As the input nears 0,
the derivative of the tanh function approaches a maximum of 1.
And as we saw with the sigmoid function,
as the input moves away from 0 in either direction,
the derivative of the tanh function approaches 0.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients.
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

In summary, we now know how to incorporate nonlinearities
to build expressive multilayer neural network architectures.
As a side note, your knowledge already
puts you in command of a similar toolkit
to a practitioner circa 1990.
In some ways, you have an advantage
over anyone working in the 1990s,
because you can leverage powerful
open-source deep learning frameworks
to build models rapidly, using only a few lines of code.
Previously, training these networks
required researchers to code up
thousands of lines of C and Fortran.

## Summary

* MLP adds one or multiple fully-connected hidden layers between the output and input layers and transforms the output of the hidden layer via an activation function.
* Commonly-used activation functions include the ReLU function, the sigmoid function, and the tanh function.


## Exercises

1. Compute the derivative of the pReLU activation function.
1. Show that an MLP using only ReLU (or pReLU) constructs a continuous piecewise linear function.
1. Show that $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$.
1. Assume that we have a nonlinearity that applies to one minibatch at a time. What kinds of problems do you expect this to cause?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/226)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTYzNzgxMzU4MywtNTEzNjM2Mzc5LC0xND
g0Mzk5ODc4LC0yMDYwMTAxNTM0LDg0OTk5NTcxMCw3MTMxMTU5
NDVdfQ==
-->
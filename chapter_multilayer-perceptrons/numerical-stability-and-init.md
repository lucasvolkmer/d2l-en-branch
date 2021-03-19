# Estabilidade Numérica e Inicialização
:label:`sec_numerical_stability`


Até agora, todos os modelos que implementamos
exigiram que inicializássemos seus parâmetros
de acordo com alguma distribuição pré-especificada.
Até agora, considerávamos o esquema de inicialização garantido,
encobrindo os detalhes de como essas escolhas são feitas.
Você pode até ter ficado com a impressão de que essas escolhas
não são especialmente importantes.
Pelo contrário, a escolha do esquema de inicialização
desempenha um papel significativo na aprendizagem da rede neural,
e pode ser crucial para manter a estabilidade numérica.
Além disso, essas escolhas podem ser amarradas de maneiras interessantes
com a escolha da função de ativação não linear.
Qual função escolhemos e como inicializamos os parâmetros
pode determinar a rapidez com que nosso algoritmo de otimização converge.
Escolhas ruins aqui podem nos fazer encontrar
explosões ou desaparecimento de gradientes durante o treinamento.
Nesta seção, nos aprofundamos nesses tópicos com mais detalhes
e discutimos algumas heurísticas úteis
que você achará útil
ao longo de sua carreira em *deep learning*.

## Explosão e Desaparecimento  de Gradientes 

Considere uma rede profunda com $ L $ camadas,
entrada $ \ mathbf {x} $ e saída $\mathbf{o}$.
Com cada camada $l$ definida por uma transformação $f_l$
parametrizada por pesos $\mathbf{W}^{(l)}$,
cuja variável oculta é $\mathbf{h}^{(l)}$ (com $\mathbf{h}^{(0)} = \mathbf{x}$),
nossa rede pode ser expressa como:

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \text{ e portanto } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x}).$$

Se todas as variáveis ocultas e a entrada forem vetores,
podemos escrever o gradiente de $\mathbf{o}$ em relação a
qualquer conjunto de parâmetros $\mathbf{W}^{(l)}$ da seguinte forma:

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.$$


Em outras palavras, este gradiente é
o produto das matrizes $L-l$
$\mathbf{M}^{(L)} \cdot \ldots \cdot \mathbf{M}^{(l+1)}$
e o vetor gradiente $\mathbf{v}^{(l)}$.
Assim, somos suscetíveis aos mesmos
problemas de *underflow* numérico que muitas vezes surgem
ao multiplicar muitas probabilidades.
Ao lidar com probabilidades, um truque comum é
mudar para o espaço de registro, ou seja, mudar
pressão da mantissa para o expoente
da representação numérica.
Infelizmente, nosso problema acima é mais sério:
inicialmente as matrizes $\mathbf{M}^{(l)}$ podem ter uma grande variedade de autovalores.
Eles podem ser pequenos ou grandes e
seu produto pode ser *muito grande* ou *muito pequeno*.

Os riscos apresentados por gradientes instáveis
vão além da representação numérica.
Gradientes de magnitude imprevisível
também ameaçam a estabilidade de nossos algoritmos de otimização.
Podemos estar enfrentando atualizações de parâmetros que são
(i) excessivamente grandes, destruindo nosso modelo
(o problema da *explosão do gradiente*);
ou (ii) excessivamente pequeno
(o problema do *desaparecimento do gradiente*),
tornando a aprendizagem impossível como parâmetros
dificilmente se move a cada atualização.


### (**Desaparecimento do Gradiente**)

Um culpado frequente que causa o problema do desaparecimento de gradiente 
é a escolha da função de ativação $\sigma$
que é anexada após as operações lineares de cada camada.
Historicamente, a função sigmóide
$1/(1 + \exp(-x))$ (introduzida em :numref:`sec_mlp`)
era popular porque se assemelha a uma função de limiar.
Como as primeiras redes neurais artificiais foram inspiradas
por redes neurais biológicas,
a ideia de neurônios que disparam *totalmente* ou *nem um pouco*
(como neurônios biológicos) parecia atraente.
Vamos dar uma olhada mais de perto no sigmóide
para ver por que isso pode causar desaparecimento de gradientes.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

Como você pode ver, (**o gradiente do sigmóide desaparece
tanto quando suas entradas são grandes quanto quando são pequenas**).
Além disso, ao retropropagar através de muitas camadas,
a menos que estejamos na zona Cachinhos Dourados, onde
as entradas para muitos dos sigmóides são próximas de zero,
os gradientes do produto geral podem desaparecer.
Quando nossa rede possui muitas camadas,
a menos que tenhamos cuidado, o gradiente
provavelmente será cortado em alguma camada.
Na verdade, esse problema costumava atormentar o treinamento profundo da rede.
Consequentemente, ReLUs, que são mais estáveis
(mas menos neuralmente plausíveis),
surgiram como a escolha padrão para os profissionais.


### [**Explosão de Gradiente**]

O problema oposto, quando os gradientes explodem,
pode ser igualmente irritante.
Para ilustrar isso um pouco melhor,
desenhamos 100 matrizes aleatórias Gaussianas
e multiplicamos-nas com alguma matriz inicial.
Para a escala que escolhemos
(a escolha da variação $\sigma^2=1$),
o produto da matriz explode.
Quando isso acontece devido à inicialização
de uma rede profunda, não temos chance de obter
um otimizador de gradiente descendente capaz de convergir.

```{.python .input}
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))

print('after multiplying 100 matrices', M)
```

```{.python .input}
#@tab pytorch
M = torch.normal(0, 1, size=(4,4))
print('a single matrix \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('after multiplying 100 matrices\n', M)
```

```{.python .input}
#@tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))

print('after multiplying 100 matrices\n', M.numpy())
```

### Quebrando a Simetria

Another problem in neural network design
is the symmetry inherent in their parametrization.
Assume that we have a simple MLP
with one hidden layer and two units.
In this case, we could permute the weights $\mathbf{W}^{(1)}$
of the first layer and likewise permute
the weights of the output layer
to obtain the same function.
There is nothing special differentiating
the first hidden unit vs. the second hidden unit.
In other words, we have permutation symmetry
among the hidden units of each layer.

This is more than just a theoretical nuisance.
Consider the aforementioned one-hidden-layer MLP
with two hidden units.
For illustration,
suppose that the output layer transforms the two hidden units into only one output unit.
Imagine what would happen if we initialized
all of the parameters of the hidden layer
as $\mathbf{W}^{(1)} = c$ for some constant $c$.
In this case, during forward propagation
either hidden unit takes the same inputs and parameters,
producing the same activation,
which is fed to the output unit.
During backpropagation,
differentiating the output unit with respect to parameters $\mathbf{W}^{(1)}$ gives a gradient whose elements all take the same value.
Thus, after gradient-based iteration (e.g., minibatch stochastic gradient descent),
all the elements of $\mathbf{W}^{(1)}$ still take the same value.
Such iterations would
never *break the symmetry* on its own
and we might never be able to realize
the network's expressive power.
The hidden layer would behave
as if it had only a single unit.
Note that while minibatch stochastic gradient descent would not break this symmetry,
dropout regularization would!


## Parameter Initialization

One way of addressing---or at least mitigating---the
issues raised above is through careful initialization.
Additional care during optimization
and suitable regularization can further enhance stability.


### Default Initialization

In the previous sections, e.g., in :numref:`sec_linear_concise`,
we used a normal distribution
to initialize the values of our weights.
If we do not specify the initialization method, the framework will
use a default random initialization method, which often works well in practice
for moderate problem sizes.






### Xavier Initialization
:label:`subsec_xavier`

Let us look at the scale distribution of
an output (e.g., a hidden variable) $o_{i}$ for some fully-connected layer
*without nonlinearities*.
With $n_\mathrm{in}$ inputs $x_j$
and their associated weights $w_{ij}$ for this layer,
an output is given by

$$o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$$

The weights $w_{ij}$ are all drawn
independently from the same distribution.
Furthermore, let us assume that this distribution
has zero mean and variance $\sigma^2$.
Note that this does not mean that the distribution has to be Gaussian,
just that the mean and variance need to exist.
For now, let us assume that the inputs to the layer $x_j$
also have zero mean and variance $\gamma^2$
and that they are independent of $w_{ij}$ and independent of each other.
In this case, we can compute the mean and variance of $o_i$ as follows:

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

One way to keep the variance fixed
is to set $n_\mathrm{in} \sigma^2 = 1$.
Now consider backpropagation.
There we face a similar problem,
albeit with gradients being propagated from the layers closer to the output.
Using the same reasoning as for forward propagation,
we see that the gradients' variance can blow up
unless $n_\mathrm{out} \sigma^2 = 1$,
where $n_\mathrm{out}$ is the number of outputs of this layer.
This leaves us in a dilemma:
we cannot possibly satisfy both conditions simultaneously.
Instead, we simply try to satisfy:

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

This is the reasoning underlying the now-standard
and practically beneficial *Xavier initialization*,
named after the first author of its creators :cite:`Glorot.Bengio.2010`.
Typically, the Xavier initialization
samples weights from a Gaussian distribution
with zero mean and variance
$\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$.
We can also adapt Xavier's intuition to
choose the variance when sampling weights
from a uniform distribution.
Note that the uniform distribution $U(-a, a)$ has variance $\frac{a^2}{3}$.
Plugging $\frac{a^2}{3}$ into our condition on $\sigma^2$
yields the suggestion to initialize according to

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$

Though the assumption for nonexistence of nonlinearities
in the above mathematical reasoning
can be easily violated in neural networks,
the Xavier initialization method
turns out to work well in practice.


### Beyond

The reasoning above barely scratches the surface
of modern approaches to parameter initialization.
A deep learning framework often implements over a dozen different heuristics.
Moreover, parameter initialization continues to be
a hot area of fundamental research in deep learning.
Among these are heuristics specialized for
tied (shared) parameters, super-resolution,
sequence models, and other situations.
For instance,
Xiao et al. demonstrated the possibility of training
10000-layer neural networks without architectural tricks
by using a carefully-designed initialization method :cite:`Xiao.Bahri.Sohl-Dickstein.ea.2018`.

If the topic interests you we suggest
a deep dive into this module's offerings,
reading the papers that proposed and analyzed each heuristic,
and then exploring the latest publications on the topic.
Perhaps you will stumble across or even invent
a clever idea and contribute an implementation to deep learning frameworks.


## Summary

* Vanishing and exploding gradients are common issues in deep networks. Great care in parameter initialization is required to ensure that gradients and parameters remain well controlled.
* Initialization heuristics are needed to ensure that the initial gradients are neither too large nor too small.
* ReLU activation functions mitigate the vanishing gradient problem. This can accelerate convergence.
* Random initialization is key to ensure that symmetry is broken before optimization.
* Xavier initialization suggests that, for each layer, variance of any output is not affected by the number of inputs, and variance of any gradient is not affected by the number of outputs.

## Exercises

1. Can you design other cases where a neural network might exhibit symmetry requiring breaking besides the permutation symmetry in an MLP's layers?
1. Can we initialize all weight parameters in linear regression or in softmax regression to the same value?
1. Look up analytic bounds on the eigenvalues of the product of two matrices. What does this tell you about ensuring that gradients are well conditioned?
1. If we know that some terms diverge, can we fix this after the fact? Look at the paper on layer-wise adaptive rate scaling  for inspiration :cite:`You.Gitman.Ginsburg.2017`.


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0NjMzNzk3MDIsLTE3NjIwNDgyNzcsMT
YyMDMxMjM4OF19
-->
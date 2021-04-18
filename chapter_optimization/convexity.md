# Convexidade
:label:`sec_convexity`

A convexidade desempenha um papel vital no projeto de algoritmos de otimização.
Em grande parte, isso se deve ao fato de que é muito mais fácil analisar e testar algoritmos em tal contexto.
Em outras palavras,
se o algoritmo funcionar mal, mesmo na configuração convexa,
normalmente, não devemos esperar ver grandes resultados de outra forma.
Além disso, embora os problemas de otimização no aprendizado profundo sejam geralmente não convexos, eles costumam exibir algumas propriedades dos convexos próximos aos mínimos locais. Isso pode levar a novas variantes de otimização interessantes, como :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

## Definições

Antes da análise convexa,
precisamos definir *conjuntos convexos* e *funções convexas*.
Eles levam a ferramentas matemáticas que são comumente aplicadas ao aprendizado de máquina.


### Convex Sets

Os conjuntos são a base da convexidade. Simplificando, um conjunto $\mathcal{X}$ em um espaço vetorial é *convexo* se para qualquer $a, b \in \mathcal{X}$ o segmento de linha conectando $a$ e $b$ também estiver em $\mathcal{X}$. Em termos matemáticos, isso significa que para todos $\lambda \in [0, 1]$ temos

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \text{ whenever } a, b \in \mathcal{X}.$$

Isso soa um pouco abstrato. Considere :numref:`fig_pacman`. O primeiro conjunto não é convexo, pois existem segmentos de linha que não estão contidos nele.
Os outros dois conjuntos não sofrem esse problema.

![O primeiro conjunto é não convexo e os outros dois são convexos.](../img/pacman.svg)
:label:`fig_pacman`

As definições por si só não são particularmente úteis, a menos que você possa fazer algo com elas.
Neste caso, podemos olhar as interseções como mostrado em :numref:`fig_convex_intersect`.
Suponha que $\mathcal{X}$ e $\mathcal{Y}$ são conjuntos convexos. Então $\mathcal{X} \cap \mathcal{Y}$ também é convexo. Para ver isso, considere qualquer $a, b \in \mathcal{X} \cap \mathcal{Y}$. Como $\mathcal{X}$ e $\mathcal{Y}$ são convexos, os segmentos de linha que conectam $a$ e $b$ estão contidos em $\mathcal{X}$ e $\mathcal{Y}$. Dado isso, eles também precisam estar contidos em $\mathcal{X} \cap \mathcal{Y}$, provando assim nosso teorema.

![A interseção entre dois conjuntos convexos é convexa.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

Podemos reforçar este resultado com pouco esforço: dados os conjuntos convexos $\mathcal{X}_i$, sua interseção $\cap_{i} \mathcal{X}_i$ é convexa.
Para ver que o inverso não é verdadeiro, considere dois conjuntos disjuntos $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Agora escolha $a \in \mathcal{X}$ e $b \in \mathcal{Y}$. O segmento de linha em :numref:`fig_nonconvex` conectando $a$ e $b$ precisa conter alguma parte que não está em $\mathcal{X}$ nem em $\mathcal{Y}$, uma vez que assumimos que $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Consequentemente, o segmento de linha também não está em $\mathcal{X} \cup \mathcal{Y}$, provando assim que, em geral, as uniões de conjuntos convexos não precisam ser convexas.

![A união de dois conjuntos convexos não precisa ser convexa.](../img/nonconvex.svg)
:label:`fig_nonconvex`

Normalmente, os problemas de aprendizado profundo são definidos em conjuntos convexos. Por exemplo, $\mathbb{R}^d$,
o conjunto de vetores $d$-dimensionais de números reais,
é um conjunto convexo (afinal, a linha entre quaisquer dois pontos em $\mathbb{R}^d$ permanece em $\mathbb{R}^d$). Em alguns casos, trabalhamos com variáveis de comprimento limitado, como bolas de raio $r$ conforme definido por $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ e } \|\mathbf{x}\| \leq r\}$.

### Função Convexa

Agora que temos conjuntos convexos, podemos introduzir *funções convexas* $f$.
Dado um conjunto convexo $\mathcal{X}$, uma função $f: \mathcal{X} \to \mathbb{R}$ é *convexa* se para todos $x, x' \in \mathcal{X}$ e para todos $\lambda \in [0, 1]$ temos

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

Para ilustrar isso, vamos representar graficamente algumas funções e verificar quais satisfazem o requisito.
Abaixo definimos algumas funções, convexas e não convexas.

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: d2l.cos(np.pi * x)  # Nonconvex
h = lambda x: d2l.exp(0.5 * x)  # Convex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

Como esperado, a função cosseno é *não convexa*, enquanto a parábola e a função exponencial são. Observe que o requisito de que $\mathcal{X}$ seja um conjunto convexo é necessário para que a condição faça sentido. Caso contrário, o resultado de $f(\lambda x + (1-\lambda) x')$ pode não ser bem definido.


### Desigualdades de Jensen

Dada uma função convexa $f$,
uma das ferramentas matemáticas mais úteis
é a *desigualdade de Jensen*.
Isso equivale a uma generalização da definição de convexidade:

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \text{ and }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$

onde $\alpha_i$ são números reais não negativos tais que $\sum_i \alpha_i = 1$ e $X$ é uma variável aleatória.
Em outras palavras, a expectativa de uma função convexa não é menos do que a função convexa de uma expectativa, onde a última é geralmente uma expressão mais simples.
Para provar a primeira desigualdade, aplicamos repetidamente a definição de convexidade a um termo da soma de cada vez.

Uma das aplicações comuns da desigualdade de Jensen é
para ligar uma expressão mais complicada por uma mais simples.
Por exemplo,
sua aplicação pode ser
no que diz respeito ao log-verossimilhança de variáveis aleatórias parcialmente observadas. Ou seja, usamos

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

uma vez que $\int P(Y) P(X \mid Y) dY = P(X)$.
Isso pode ser usado em métodos variacionais. Aqui $Y$ é normalmente a variável aleatória não observada, $P(Y)$ é a melhor estimativa de como ela pode ser distribuída e $P(X)$ é a distribuição com $Y$ integrada. Por exemplo, no agrupamento $Y$ podem ser os rótulos de cluster e $P(X \mid Y)$ é o modelo gerador ao aplicar rótulos de cluster.

## Propriedades 

As funções convexas têm algumas propriedades úteis. Nós os descrevemos abaixo.


### Mínimos locais são mínimos globais

Em primeiro lugar, os mínimos locais das funções convexas também são os mínimos globais.
Podemos provar isso por contradição como segue.

Considere uma função convexa $f$ definida em um conjunto convexo $\mathcal{X}$.
Suponha que $x^{\ast} \in \mathcal{X}$ seja um mínimo local:
existe um pequeno valor positivo $p$ de forma que para $x \in \mathcal{X}$ que satisfaz $0 < |x - x^{\ast}| \leq p$ temos $f(x^{\ast}) < f(x)$.

Assume that the local minimum $x^{\ast}$
is not the global minumum of $f$:
there exists $x' \in \mathcal{X}$ for which $f(x') < f(x^{\ast})$. 
There also exists 
$\lambda \in [0, 1)$ such as $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$
so that
$0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$. 

However,
according to the definition of convex functions, we have

Suponha que o mínimo local $x^{\ast}$
não é o mínimo global de $f$:
existe $ x '\ in \ mathcal {X} $ para o qual $ f (x') <f (x ^ {\ ast}) $.
Também existe
$ \ lambda \ in [0, 1) $ como $ \ lambda = 1 - \ frac {p} {| x ^ {\ ast} - x '|} $
de modo a
$ 0 <| \ lambda x ^ {\ ast} + (1- \ lambda) x '- x ^ {\ ast} | \ leq p $.

Contudo,
de acordo com a definição de funções convexas, temos

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

which contradicts with our statement that $x^{\ast}$ is a local minimum.
Therefore, there does not exist $x' \in \mathcal{X}$ for which $f(x') < f(x^{\ast})$. The local minimum $x^{\ast}$ is also the global minimum.

For instance, the convex function $f(x) = (x-1)^2$ has a local minimum at $x=1$, which is also the global minimum.

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

The fact that the local minima for convex functions are also the global minima is very convenient. 
It means that if we minimize functions we cannot "get stuck". 
Note, though, that this does not mean that there cannot be more than one global minimum or that there might even exist one. For instance, the function $f(x) = \mathrm{max}(|x|-1, 0)$ attains its minimum value over the interval $[-1, 1]$. Conversely, the function $f(x) = \exp(x)$ does not attain a minimum value on $\mathbb{R}$: for $x \to -\infty$ it asymptotes to $0$, but there is no $x$ for which $f(x) = 0$.

### Below Sets of Convex Functions Are Convex

We can conveniently 
define convex sets 
via *below sets* of convex functions.
Concretely,
given a convex function $f$ defined on a convex set $\mathcal{X}$,
any below set

$$\mathcal{S}_b := \{x | x \in \mathcal{X} \text{ and } f(x) \leq b\}$$

is convex. 

Let us prove this quickly. Recall that for any $x, x' \in \mathcal{S}_b$ we need to show that $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$ as long as $\lambda \in [0, 1]$. 
Since $f(x) \leq b$ and $f(x') \leq b$,
by the definition of convexity we have 

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$


### Convexity and Second Derivatives

Whenever the second derivative of a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ exists it is very easy to check whether $f$ is convex. 
All we need to do is check whether the Hessian of $f$ is positive semidefinite: $\nabla^2f \succeq 0$, i.e., 
denoting the Hessian matrix $\nabla^2f$ by $\mathbf{H}$,
$\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$
for all $\mathbf{x} \in \mathbb{R}^n$.
For instance, the function $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$ is convex since $\nabla^2 f = \mathbf{1}$, i.e., its Hessian is an identity matrix.


Formally, any twice-differentiable one-dimensional function $f: \mathbb{R} \rightarrow \mathbb{R}$ is convex
if and only if its second derivative $f'' \geq 0$. For any twice-differentiable multi-dimensional function $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$,
it is convex if and only if its Hessian $\nabla^2f \succeq 0$.

First, we need to prove the one-dimensional case.
To see that 
convexity of $f$ implies 
$f'' \geq 0$  we use the fact that

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

Since the second derivative is given by the limit over finite differences it follows that

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

To see that 
$f'' \geq 0$ implies that $f$ is convex
we use the fact that $f'' \geq 0$ implies that $f'$ is a monotonically nondecreasing function. Let $a < x < b$ be three points in $\mathbb{R}$,
where $x = (1-\lambda)a + \lambda b$ and $\lambda \in (0, 1)$.
According to the mean value theorem,
there exist $\alpha \in [a, x]$ and $\beta \in [x, b]$
such that

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \text{ and } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$


By monotonicity $f'(\beta) \geq f'(\alpha)$, hence

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

Since $x = (1-\lambda)a + \lambda b$,
we have

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

thus proving convexity.

Second, we need a lemma before 
proving the multi-dimensional case:
$f: \mathbb{R}^n \rightarrow \mathbb{R}$
is convex if and only if for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$

$$g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \text{ where } z \in [0,1]$$ 

is convex.

To prove that convexity of $f$ implies that $g$ is convex,
we can show that for all $a, b, \lambda \in [0, 1]$ (thus
$0 \leq \lambda a + (1-\lambda) b \leq 1$)

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

To prove the converse,
we can show that for 
all $\lambda \in [0, 1]$ 

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) g(\mathbf{y}).
\end{aligned}$$


Finally,
using the lemma above and the result of the one-dimensional case,
the multi-dimensional case
can be proven as follows.
A multi-dimensional function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is convex
if and only if for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ $g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$, where $z \in [0,1]$,
is convex.
According to the one-dimensional case,
this holds if and only if
$g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$)
for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$,
which is equivalent to $\mathbf{H} \succeq 0$
per the definition of positive semidefinite matrices.



## Constraints

One of the nice properties of convex optimization is that it allows us to handle constraints efficiently. That is, it allows us to solve problems of the form:

$$\begin{aligned} \mathop{\mathrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, N\}.
\end{aligned}$$

Here $f$ is the objective and the functions $c_i$ are constraint functions. To see what this does consider the case where $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$. In this case the parameters $\mathbf{x}$ are constrained to the unit ball. If a second constraint is $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$, then this corresponds to all $\mathbf{x}$ lying on a halfspace. Satisfying both constraints simultaneously amounts to selecting a slice of a ball as the constraint set.

### Lagrange Function

In general, solving a constrained optimization problem is difficult. One way of addressing it stems from physics with a rather simple intuition. Imagine a ball inside a box. The ball will roll to the place that is lowest and the forces of gravity will be balanced out with the forces that the sides of the box can impose on the ball. In short, the gradient of the objective function (i.e., gravity) will be offset by the gradient of the constraint function (need to remain inside the box by virtue of the walls "pushing back"). Note that any constraint that is not active (i.e., the ball does not touch the wall) will not be able to exert any force on the ball.

Skipping over the derivation of the Lagrange function $L$ (see e.g., the book by Boyd and Vandenberghe for details :cite:`Boyd.Vandenberghe.2004`) the above reasoning can be expressed via the following saddlepoint optimization problem:

$$L(\mathbf{x},\alpha) = f(\mathbf{x}) + \sum_i \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$

Here the variables $\alpha_i$ are the so-called *Lagrange Multipliers* that ensure that a constraint is properly enforced. They are chosen just large enough to ensure that $c_i(\mathbf{x}) \leq 0$ for all $i$. For instance, for any $\mathbf{x}$ for which $c_i(\mathbf{x}) < 0$ naturally, we'd end up picking $\alpha_i = 0$. Moreover, this is a *saddlepoint* optimization problem where one wants to *maximize* $L$ with respect to $\alpha$ and simultaneously *minimize* it with respect to $\mathbf{x}$. There is a rich body of literature explaining how to arrive at the function $L(\mathbf{x}, \alpha)$. For our purposes it is sufficient to know that the saddlepoint of $L$ is where the original constrained optimization problem is solved optimally.

### Penalties

One way of satisfying constrained optimization problems at least approximately is to adapt the Lagrange function $L$. Rather than satisfying $c_i(\mathbf{x}) \leq 0$ we simply add $\alpha_i c_i(\mathbf{x})$ to the objective function $f(x)$. This ensures that the constraints will not be violated too badly.

In fact, we have been using this trick all along. Consider weight decay in :numref:`sec_weight_decay`. In it we add $\frac{\lambda}{2} \|\mathbf{w}\|^2$ to the objective function to ensure that $\mathbf{w}$ does not grow too large. Using the constrained optimization point of view we can see that this will ensure that $\|\mathbf{w}\|^2 - r^2 \leq 0$ for some radius $r$. Adjusting the value of $\lambda$ allows us to vary the size of $\mathbf{w}$.

In general, adding penalties is a good way of ensuring approximate constraint satisfaction. In practice this turns out to be much more robust than exact satisfaction. Furthermore, for nonconvex problems many of the properties that make the exact approach so appealing in the convex case (e.g., optimality) no longer hold.

### Projections

An alternative strategy for satisfying constraints are projections. Again, we encountered them before, e.g., when dealing with gradient clipping in :numref:`sec_rnn_scratch`. There we ensured that a gradient has length bounded by $c$ via

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, c/\|\mathbf{g}\|).$$

This turns out to be a *projection* of $g$ onto the ball of radius $c$. More generally, a projection on a (convex) set $\mathcal{X}$ is defined as

$$\mathrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|_2.$$

It is thus the closest point in $\mathcal{X}$ to $\mathbf{x}$. This sounds a bit abstract. :numref:`fig_projections` explains it somewhat more clearly. In it we have two convex sets, a circle and a diamond. Points inside the set (yellow) remain unchanged. Points outside the set (black) are mapped to the closest point inside the set (red). While for $L_2$ balls this leaves the direction unchanged, this need not be the case in general, as can be seen in the case of the diamond.

![Convex Projections.](../img/projections.svg)
:label:`fig_projections`

One of the uses for convex projections is to compute sparse weight vectors. In this case we project $\mathbf{w}$ onto an $L_1$ ball (the latter is a generalized version of the diamond in the picture above).

## Summary

In the context of deep learning the main purpose of convex functions is to motivate optimization algorithms and help us understand them in detail. In the following we will see how gradient descent and stochastic gradient descent can be derived accordingly.

* Intersections of convex sets are convex. Unions are not.
* The expectation of a convex function is larger than the convex function of an expectation (Jensen's inequality).
* A twice-differentiable function is convex if and only if its second derivative has only nonnegative eigenvalues throughout.
* Convex constraints can be added via the Lagrange function. In practice simply add them with a penalty to the objective function.
* Projections map to points in the (convex) set closest to the original point.

## Exercises

1. Assume that we want to verify convexity of a set by drawing all lines between points within the set and checking whether the lines are contained.
    1. Prove that it is sufficient to check only the points on the boundary.
    1. Prove that it is sufficient to check only the vertices of the set.
1. Denote by $\mathcal{B}_p[r] := \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$ the ball of radius $r$ using the $p$-norm. Prove that $\mathcal{B}_p[r]$ is convex for all $p \geq 1$.
1. Given convex functions $f$ and $g$ show that $\mathrm{max}(f, g)$ is convex, too. Prove that $\mathrm{min}(f, g)$ is not convex.
1. Prove that the normalization of the softmax function is convex. More specifically prove the convexity of
    $f(x) = \log \sum_i \exp(x_i)$.
1. Prove that linear subspaces are convex sets, i.e., $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$.
1. Prove that in the case of linear subspaces with $\mathbf{b} = 0$ the projection $\mathrm{Proj}_\mathcal{X}$ can be written as $\mathbf{M} \mathbf{x}$ for some matrix $\mathbf{M}$.
1. Show that for convex twice differentiable functions $f$ we can write $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ for some $\xi \in [0, \epsilon]$.
1. Given a vector $\mathbf{w} \in \mathbb{R}^d$ with $\|\mathbf{w}\|_1 > 1$ compute the projection on the $\ell_1$ unit ball.
    1. As intermediate step write out the penalized objective $\|\mathbf{w} - \mathbf{w}'\|_2^2 + \lambda \|\mathbf{w}'\|_1$ and compute the solution for a given $\lambda > 0$.
    1. Can you find the 'right' value of $\lambda$ without a lot of trial and error?
1. Given a convex set $\mathcal{X}$ and two vectors $\mathbf{x}$ and $\mathbf{y}$ prove that projections never increase distances, i.e., $\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_\mathcal{X}(\mathbf{x}) - \mathrm{Proj}_\mathcal{X}(\mathbf{y})\|$.


[Discussions](https://discuss.d2l.ai/t/350)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk0MjE3MjIyNCwtMTc4OTA5Mjk0OSwtMj
E0NjMwMjM5XX0=
-->
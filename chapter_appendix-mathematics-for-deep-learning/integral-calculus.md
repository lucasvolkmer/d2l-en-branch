# Cálculo Integral
:label:`sec_integral_calculus`


A diferenciação representa apenas metade do conteúdo de uma educação tradicional de cálculo. O outro pilar, integração, começa parecendo uma pergunta um tanto desconexa: "Qual é a área abaixo desta curva?" Embora aparentemente não relacionada, a integração está intimamente ligada à diferenciação por meio do que é conhecido como *teorema fundamental do cálculo*.

No nível de *machine learning* que discutimos neste livro, não precisaremos ter um conhecimento profundo de integração. No entanto, forneceremos uma breve introdução para estabelecer as bases para quaisquer outras aplicações que encontraremos mais tarde.

## Interpretação Geométrica
Suponha que tenhamos uma função $f(x)$. Para simplificar, vamos supor que $f(x)$ não seja negativa (nunca assume um valor menor que zero). O que queremos tentar entender é: qual é a área contida entre $f(x)$ e o eixo $x$?

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()

x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch

x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf

x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy(), f.numpy())
d2l.plt.show()
```

Na maioria dos casos, esta área será infinita ou indefinida (considere a área sob $f(x) = x^{2}$), então as pessoas frequentemente falarão sobre a área entre um par de pontas, digamos $a$ e $b$.

```{.python .input}
x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy()[50:250], f.numpy()[50:250])
d2l.plt.show()
```

Iremos denotar esta área pelo símbolo integral abaixo:

$$
\mathrm{Area}(\mathcal{A}) = \int_a^b f(x) \;dx.
$$

A variável interna é uma variável fictícia, muito parecida com o índice de uma soma em $\sum$ e, portanto, pode ser escrita de forma equivalente com qualquer valor interno que desejarmos:

$$
\int_a^b f(x) \;dx = \int_a^b f(z) \;dz.
$$

Há uma maneira tradicional de tentar entender como podemos tentar aproximar essas integrais: podemos imaginar pegar a região entre $a$ e $b$ e dividi-la em fatias verticais de $N$. Se $N$ for grande, podemos aproximar a área de cada fatia por um retângulo e, em seguida, somar as áreas para obter a área total sob a curva. Vamos dar uma olhada em um exemplo fazendo isso no código. Veremos como obter o valor verdadeiro em uma seção posterior.

```{.python .input}
epsilon = 0.05
a = 0
b = 2

x = np.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = np.sum(epsilon*f)
true = np.log(2) / 2

d2l.set_figsize()
d2l.plt.bar(x.asnumpy(), f.asnumpy(), width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab pytorch
epsilon = 0.05
a = 0
b = 2

x = torch.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = torch.sum(epsilon*f)
true = torch.log(torch.tensor([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab tensorflow
epsilon = 0.05
a = 0
b = 2

x = tf.range(a, b, epsilon)
f = x / (1 + x**2)

approx = tf.reduce_sum(epsilon*f)
true = tf.math.log(tf.constant([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

O problema é que, embora possa ser feito numericamente, podemos fazer essa abordagem analiticamente apenas para as funções mais simples, como

$$
\int_a^b x \;dx.
$$

Qualquer coisa um pouco mais complexa como nosso exemplo do código acima

$$
\int_a^b \frac{x}{1+x^{2}} \;dx.
$$


está além do que podemos resolver com um método tão direto.

Em vez disso, faremos uma abordagem diferente. Trabalharemos intuitivamente com a noção da área, e aprenderemos a principal ferramenta computacional usada para encontrar integrais: o *teorema fundamental do cálculo*. Esta será a base para nosso estudo de integração.

## O Teorema Fundamental do Cálculo

Para mergulhar mais fundo na teoria da integração, vamos apresentar uma função

$$
F(x) = \int_0^x f(y) dy.
$$

Esta função mede a área entre $0$ e $x$ dependendo de como alteramos $x$. Observe que isso é tudo de que precisamos, já que

$$
\int_a^b f(x) \;dx = F(b) - F(a).
$$

Esta é uma codificação matemática do fato de que podemos medir a área até o ponto final distante e então subtrair a área até o ponto final próximo, conforme indicado em :numref:`fig_area-subtract`.

![Visualizando porque podemos reduzir o problema de calcular a área sob uma curva entre dois pontos para calcular a área à esquerda de um ponto.](../img/sub-area.svg)
:label:`fig_area-subtract`


Assim, podemos descobrir qual é a integral em qualquer intervalo, descobrindo o que é $F(x)$.

Para fazer isso, consideremos um experimento. Como costumamos fazer em cálculo, vamos imaginar o que acontece quando mudamos o valor um pouquinho. Pelo comentário acima, sabemos que

$$
F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y) \; dy.
$$


Isso nos diz que a função muda de acordo com a área sob uma pequena porção de uma função.

Este é o ponto em que fazemos uma aproximação. Se olharmos para uma pequena porção de área como esta, parece que esta área está próxima da área retangular com a altura o valor de $f(x)$ e a largura da base $\epsilon$ De fato, pode-se mostrar que à medida que $\epsilon \rightarrow 0$ essa aproximação se torna cada vez melhor. Assim podemos concluir:

$$
F(x+\epsilon) - F(x) \approx \epsilon f(x).
$$

Porém, agora podemos notar: este é exatamente o padrão que esperamos se estivéssemos calculando a derivada de $F$! Assim, vemos o seguinte fato bastante surpreendente:

$$
\frac{dF}{dx}(x) = f(x).
$$

Este é o *teorema fundamental do cálculo*. Podemos escrever em forma expandida como
$$\frac{d}{dx}\int_{-\infty}^x f(y) \; dy = f(x).$$
:eqlabel:`eq_ftc`

Ele pega o conceito de localização de áreas (*a priori* bastante difícil) e o reduz a derivadas de uma instrução (algo muito mais completamente compreendido). Um último comentário que devemos fazer é que isso não nos diz exatamente o que $F(x)$ é. Na verdade, $F(x) + C$ para qualquer $C$ tem a mesma derivada. Este é um fato da vida na teoria da integração. Felizmente, observe que, ao trabalhar com integrais definidas, as constantes desaparecem e, portanto, são irrelevantes para o resultado.

$$
\int_a^b f(x) \; dx = (F(b) + C) - (F(a) + C) = F(b) - F(a).
$$

Isso pode parecer sem sentido abstrato, mas vamos parar um momento para apreciar que isso nos deu uma perspectiva totalmente nova sobre as integrais computacionais. Nosso objetivo não é mais fazer algum tipo de processo de corte e soma para tentar recuperar a área, ao invés disso, precisamos apenas encontrar uma função cuja derivada é a função que temos! Isso é incrível, pois agora podemos listar muitas integrais bastante difíceis apenas revertendo a tabela de :numref:`sec_derivative_table`. Por exemplo, sabemos que a derivada de $x^{n}$ is $nx^{n-1}$. Assim, podemos dizer usando o teorema fundamental :eqref:`eq_ftc` que

$$
\int_0^{x} ny^{n-1} \; dy = x^n - 0^n = x^n.
$$

Da mesma forma, sabemos que a derivada de $e^{x}$ é ela mesma, o que significa

$$
\int_0^{x} e^{x} \; dx = e^{x} - e^{0} = e^x - 1.
$$

Desta forma, podemos desenvolver toda a teoria da integração aproveitando as idéias do cálculo diferencial livremente. Toda regra de integração deriva desse único fato.

## Mudança de Variável
:label:`integral_example`


Assim como com a diferenciação, há várias regras que tornam o cálculo de integrais mais tratáveis. Na verdade, cada regra de cálculo diferencial (como a regra do produto, regra da soma e regra da cadeia) tem uma regra correspondente para o cálculo integral (integração por partes, linearidade de integração e fórmula de mudança de variáveis, respectivamente). Nesta seção, vamos mergulhar no que é indiscutivelmente o mais importante da lista: a fórmula de mudança de variáveis.

Primeiro, suponha que temos uma função que é ela mesma uma integral:

$$
F(x) = \int_0^x f(y) \; dy.
$$

Vamos supor que queremos saber como essa função se parece quando a compomos com outra para obter $F(u(x))$. Pela regra da cadeia, sabemos

$$
\frac{d}{dx}F(u(x)) = \frac{dF}{du}(u(x))\cdot \frac{du}{dx}.
$$

Podemos transformar isso em uma declaração sobre integração usando o teorema fundamental :eqref:`eq_ftc` como acima. Isto dá

$$
F(u(x)) - F(u(0)) = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

Lembrando que $F$ é em si uma integral dá que o lado esquerdo pode ser reescrito para ser

$$
\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

Da mesma forma, lembrar que $F$ é uma integral nos permite reconhecer que $\frac{dF}{dx} = f$ usando o teorema fundamental :eqref:`eq_ftc`, e assim podemos concluir
$$\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x f(u(y))\cdot \frac{du}{dy} \;dy.$$
:eqlabel:`eq_change_var`

Esta é a fórmula de *mudança de variáveis*.

Para uma derivação mais intuitiva, considere o que acontece quando tomamos uma integral de $f(u(x))$ entre $x$ e $x+\epsilon$. Para um pequeno $\epsilon$, esta integral é aproximadamente $\epsilon f(u(x))$, a área do retângulo associado. Agora, vamos comparar isso com a integral de $f(y)$ de $u(x)$ a $u(x+\epsilon)$. Sabemos que $u(x+\epsilon) \approx u(x) + \epsilon \frac{du}{dx}(x)$, então a área deste retângulo é aproximadamente $\epsilon \frac{du}{dx}(x)f(u(x))$. Assim, para fazer a área desses dois retângulos serem iguais, precisamos multiplicar o primeiro por $\frac{du}{dx}(x)$ como está ilustrado em :numref:`fig_rect-transform`.

![Visualizando a transformação de um único retângulo fino sob a mudança de variáveis.](../img/rect-trans.svg)
:label:`fig_rect-transform`

Isso nos diz que

$$
\int_x^{x+\epsilon} f(u(y))\frac{du}{dy}(y)\;dy = \int_{u(x)}^{u(x+\epsilon)} f(y) \; dy.
$$


Esta é a fórmula de mudança de variáveis expressa para um único retângulo pequeno.

Se $u(x)$ e $f(x)$ forem escolhidos corretamente, isso pode permitir o cálculo de integrais incrivelmente complexas. Por exemplo, se escolhermos $f(y) = 1$ e $u(x) = e^{-x^{2}}$ (o que significa $\frac{du}{dx}(x) = -2xe^{-x^{2}}$), isso pode mostrar, por exemplo, que

$$
e^{-1} - 1 = \int_{e^{-0}}^{e^{-1}} 1 \; dy = -2\int_0^{1} ye^{-y^2}\;dy,
$$

e, assim, reorganizando isso

$$
\int_0^{1} ye^{-y^2}\; dy = \frac{1-e^{-1}}{2}.
$$

## A Comment on Sign Conventions

Keen-eyed readers will observe something strange about the computations above.  Namely, computations like

$$
\int_{e^{-0}}^{e^{-1}} 1 \; dy = e^{-1} -1 < 0,
$$

can produce negative numbers.  When thinking about areas, it can be strange to see a negative value, and so it is worth digging into what the convention is.

Mathematicians take the notion of signed areas.  This manifests itself in two ways.  First, if we consider a function $f(x)$ which is sometimes less than zero, then the area will also be negative.  So for instance

$$
\int_0^{1} (-1)\;dx = -1.
$$

Similarly, integrals which progress from right to left, rather than left to right are also taken to be negative areas

$$
\int_0^{-1} 1\; dx = -1.
$$

The standard area (from left to right of a positive function) is always positive.  Anything obtained by flipping it (say flipping over the $x$-axis to get the integral of a negative number, or flipping over the $y$-axis to get an integral in the wrong order) will produce a negative area.  And indeed, flipping twice will give a pair of negative signs that cancel out to have positive area

$$
\int_0^{-1} (-1)\;dx =  1.
$$

If this discussion sounds familiar, it is!  In :numref:`sec_geometry-linear-algebraic-ops` we discussed how the determinant represented the signed area in much the same way.

## Multiple Integrals
In some cases, we will need to work in higher dimensions.  For instance, suppose that we have a function of two variables, like $f(x, y)$ and we want to know the volume under $f$ when $x$ ranges over $[a, b]$ and $y$ ranges over $[c, d]$.

```{.python .input}
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101),
                   indexing='ij')
z = np.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101), torch.linspace(-2, 2, 101))
z = torch.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101), tf.linspace(-2., 2., 101))
z = tf.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

We write this as

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

Suppose that we wish to compute this integral.  My claim is that we can do this by iteratively computing first the integral in $x$ and then shifting to the integral in $y$, that is to say

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int_c^{d} \left(\int_a^{b} f(x, y) \;dx\right) \; dy.
$$

Let us see why this is.

Consider the figure above where we have split the function into $\epsilon \times \epsilon$ squares which we will index with integer coordinates $i, j$.  In this case, our integral is approximately

$$
\sum_{i, j} \epsilon^{2} f(\epsilon i, \epsilon j).
$$

Once we discretize the problem, we may add up the values on these squares in whatever order we like, and not worry about changing the values.  This is illustrated in :numref:`fig_sum-order`.  In particular, we can say that

$$
 \sum _ {j} \epsilon \left(\sum_{i} \epsilon f(\epsilon i, \epsilon j)\right).
$$

![Illustrating how to decompose a sum over many squares as a sum over first the columns (1), then adding the column sums together (2).](../img/sum-order.svg)
:label:`fig_sum-order`

The sum on the inside is precisely the discretization of the integral

$$
G(\epsilon j) = \int _a^{b} f(x, \epsilon j) \; dx.
$$

Finally, notice that if we combine these two expressions we get

$$
\sum _ {j} \epsilon G(\epsilon j) \approx \int _ {c}^{d} G(y) \; dy = \int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

Thus putting it all together, we have that

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ c^{d} \left(\int _ a^{b} f(x, y) \;dx\right) \; dy.
$$

Notice that, once discretized, all we did was rearrange the order in which we added a list of numbers.  This may make it seem like it is nothing, however this result (called *Fubini's Theorem*) is not always true!  For the type of mathematics encountered when doing machine learning (continuous functions), there is no concern, however it is possible to create examples where it fails (for example the function $f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3$ over the rectangle $[0,2]\times[0,1]$).

Note that the choice to do the integral in $x$ first, and then the integral in $y$ was arbitrary.  We could have equally well chosen to do $y$ first and then $x$ to see

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ a^{b} \left(\int _ c^{d} f(x, y) \;dy\right) \; dx.
$$

Often times, we will condense down to vector notation, and say that for $U = [a, b]\times [c, d]$ this is

$$
\int _ U f(\mathbf{x})\;d\mathbf{x}.
$$

## Change of Variables in Multiple Integrals
As with single variables in :eqref:`eq_change_var`, the ability to change variables inside a higher dimensional integral is a key tool.  Let us summarize the result without derivation.

We need a function that reparameterizes our domain of integration.  We can take this to be $\phi : \mathbb{R}^n \rightarrow \mathbb{R}^n$, that is any function which takes in $n$ real variables and returns another $n$.  To keep the expressions clean, we will assume that $\phi$ is *injective* which is to say it never folds over itself ($\phi(\mathbf{x}) = \phi(\mathbf{y}) \implies \mathbf{x} = \mathbf{y}$).

In this case, we can say that

$$
\int _ {\phi(U)} f(\mathbf{x})\;d\mathbf{x} = \int _ {U} f(\phi(\mathbf{x})) \left|\det(D\phi(\mathbf{x}))\right|\;d\mathbf{x}.
$$

where $D\phi$ is the *Jacobian* of $\phi$, which is the matrix of partial derivatives of $\boldsymbol{\phi} = (\phi_1(x_1, \ldots, x_n), \ldots, \phi_n(x_1, \ldots, x_n))$,

$$
D\boldsymbol{\phi} = \begin{bmatrix}
\frac{\partial \phi _ 1}{\partial x _ 1} & \cdots & \frac{\partial \phi _ 1}{\partial x _ n} \\
\vdots & \ddots & \vdots \\
\frac{\partial \phi _ n}{\partial x _ 1} & \cdots & \frac{\partial \phi _ n}{\partial x _ n}
\end{bmatrix}.
$$

Looking closely, we see that this is similar to the single variable chain rule :eqref:`eq_change_var`, except we have replaced the term $\frac{du}{dx}(x)$ with $\left|\det(D\phi(\mathbf{x}))\right|$.  Let us see how we can to interpret this term.  Recall that the $\frac{du}{dx}(x)$ term existed to say how much we stretched our $x$-axis by applying $u$.  The same process in higher dimensions is to determine how much we stretch the area (or volume, or hyper-volume) of a little square (or little *hyper-cube*) by applying $\boldsymbol{\phi}$.  If $\boldsymbol{\phi}$ was the multiplication by a matrix, then we know how the determinant already gives the answer.

With some work, one can show that the *Jacobian* provides the best approximation to a multivariable function $\boldsymbol{\phi}$ at a point by a matrix in the same way we could approximate by lines or planes with derivatives and gradients. Thus the determinant of the Jacobian exactly mirrors the scaling factor we identified in one dimension.

It takes some work to fill in the details to this, so do not worry if they are not clear now.  Let us see at least one example we will make use of later on.  Consider the integral

$$
\int _ {-\infty}^{\infty} \int _ {-\infty}^{\infty} e^{-x^{2}-y^{2}} \;dx\;dy.
$$

Playing with this integral directly will get us no-where, but if we change variables, we can make significant progress.  If we let $\boldsymbol{\phi}(r, \theta) = (r \cos(\theta),  r\sin(\theta))$ (which is to say that $x = r \cos(\theta)$, $y = r \sin(\theta)$), then we can apply the change of variable formula to see that this is the same thing as

$$
\int _ 0^\infty \int_0 ^ {2\pi} e^{-r^{2}} \left|\det(D\mathbf{\phi}(\mathbf{x}))\right|\;d\theta\;dr,
$$

where

$$
\left|\det(D\mathbf{\phi}(\mathbf{x}))\right| = \left|\det\begin{bmatrix}
\cos(\theta) & -r\sin(\theta) \\
\sin(\theta) & r\cos(\theta)
\end{bmatrix}\right| = r(\cos^{2}(\theta) + \sin^{2}(\theta)) = r.
$$

Thus, the integral is

$$
\int _ 0^\infty \int _ 0 ^ {2\pi} re^{-r^{2}} \;d\theta\;dr = 2\pi\int _ 0^\infty re^{-r^{2}} \;dr = \pi,
$$

where the final equality follows by the same computation that we used in section :numref:`integral_example`.

We will meet this integral again when we study continuous random variables in :numref:`sec_random_variables`.

## Summary

* The theory of integration allows us to answer questions about areas or volumes.
* The fundamental theorem of calculus allows us to leverage knowledge about derivatives to compute areas via the observation that the derivative of the area up to some point is given by the value of the function being integrated.
* Integrals in higher dimensions can be computed by iterating single variable integrals.

## Exercises
1. What is $\int_1^2 \frac{1}{x} \;dx$?
2. Use the change of variables formula to integrate $\int_0^{\sqrt{\pi}}x\sin(x^2)\;dx$.
3. What is $\int_{[0,1]^2} xy \;dx\;dy$?
4. Use the change of variables formula to compute $\int_0^2\int_0^1xy(x^2-y^2)/(x^2+y^2)^3\;dy\;dx$ and $\int_0^1\int_0^2f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3\;dx\;dy$ to see they are different.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/414)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1092)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1093)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIxMzUyMjc0NTIsMTQ1MTk2NDQ2MSwtNj
Y2MDA5NDg4LC0zMzA0NTQzODAsLTE0OTY4MDc4MzJdfQ==
-->
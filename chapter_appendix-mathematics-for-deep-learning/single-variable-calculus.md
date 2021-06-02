# Cálculo de Variável Única
:label:`sec_single_variable_calculus`

Em :numref:`sec_calculus`, vimos os elementos básicos do cálculo diferencial. Esta seção dá um mergulho mais profundo nos fundamentos do cálculo e como podemos entendê-lo e aplicá-lo no contexto do aprendizado de máquina.

## Cálculo diferencial

O cálculo diferencial é fundamentalmente o estudo de como as funções se comportam sob pequenas mudanças. Para ver por que isso é tão importante para o*deep learning*, vamos considerar um exemplo.

Suponha que temos uma rede neural profunda onde os pesos são, por conveniência, concatenados em um único vetor $\mathbf{w} = (w_1, \ldots, w_n)$. Dado um conjunto de dados de treinamento, consideramos a perda de nossa rede neural neste conjunto de dados, que escreveremos como $\mathcal{L}(\mathbf{w})$.  

Esta função é extraordinariamente complexa, codificando o desempenho de todos os modelos possíveis da arquitetura dada neste conjunto de dados, então é quase impossível dizer qual conjunto de pesos $\mathbf{w}$ irá minimizar a perda. Assim, na prática, geralmente começamos inicializando nossos pesos *aleatoriamente* e, em seguida, damos passos pequenos iterativamente na direção que faz com que a perda diminua o mais rápido possível.


A questão então se torna algo que superficialmente não é mais fácil: como encontramos a direção que faz com que os pesos diminuam o mais rápido possível? Para nos aprofundarmos nisso, vamos primeiro examinar o caso com apenas um único peso: $L(\mathbf{w}) = L(x)$ para um único valor real $x$.

Vamos pegar $x$ e tentar entender o que acontece quando o alteramos por uma pequena quantia para $x + \epsilon$. Se você deseja ser concreto, pense em um número como $\epsilon = 0.0000001$. Para nos ajudar a visualizar o que acontece, vamos representar graficamente uma função de exemplo, $f(x) = \sin(x^x)$, sobre $[0, 3]$.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot a function in a normal range
x_big = np.arange(0.01, 3.01, 0.01)
ys = np.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot a function in a normal range
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot a function in a normal range
x_big = tf.range(0.01, 3.01, 0.01)
ys = tf.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

Em grande escala, o comportamento da função não é simples. No entanto, se reduzirmos nosso intervalo para algo menor como $[1.75,2.25]$, vemos que o gráfico se torna muito mais simples.

```{.python .input}
# Plot a the same function in a tiny range
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_med = tf.range(1.75, 2.25, 0.001)
ys = tf.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

Levando isso ao extremo, se ampliarmos em um segmento minúsculo, o comportamento se torna muito mais simples: é apenas uma linha reta.

```{.python .input}
# Plot a the same function in a tiny range
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_small = tf.range(2.0, 2.01, 0.0001)
ys = tf.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```


Esta é a observação chave do cálculo de variável única: o comportamento de funções familiares pode ser modelado por uma linha em um intervalo pequeno o suficiente. Isso significa que, para a maioria das funções, é razoável esperar que, à medida que deslocamos um pouco o valor $x$ da função, a saída $f(x)$ também seja deslocada um pouco. A única pergunta que precisamos responder é: "Qual é o tamanho da mudança na produção em comparação com a mudança na entrada? É a metade? Duas vezes maior?"

Assim, podemos considerar a razão da mudança na saída de uma função para uma pequena mudança na entrada da função. Podemos escrever isso formalmente como

$$
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}.
$$

Isso já é o suficiente para começar a brincar com o código. Por exemplo, suponha que saibamos que $L(x) = x^{2} + 1701(x-4)^3$, então podemos ver o quão grande é esse valor no ponto $x = 4$ como segue.

```{.python .input}
#@tab all
# Define our function
def L(x):
    return x**2 + 1701*(x-4)**3

# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```

Agora, se formos observadores, notaremos que a saída desse número é suspeitamente próxima de $8$. De fato, se diminuirmos $\epsilon$, veremos que o valor se torna progressivamente mais próximo de $8$. Assim, podemos concluir, corretamente, que o valor que buscamos (o grau em que uma mudança na entrada muda a saída) deve ser $8$ no ponto $x=4$. A forma como um matemático codifica este fato é

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$


Como uma pequena digressão histórica: nas primeiras décadas de pesquisa de redes neurais, os cientistas usaram este algoritmo (o *método das diferenças finitas*) para avaliar como uma função de perda mudou sob pequenas perturbações: basta alterar os pesos e ver como o perda mudou. Isso é computacionalmente ineficiente, exigindo duas avaliações da função de perda para ver como uma única mudança de uma variável influenciou a perda. Se tentássemos fazer isso mesmo com alguns poucos milhares de parâmetros, seriam necessários vários milhares de avaliações da rede em todo o conjunto de dados! Não foi resolvido até 1986 que o *algoritmo de retropropagação* introduzido em :cite:`Rumelhart.Hinton.Williams.ea.1988` forneceu uma maneira de calcular como *qualquer* alteração dos pesos juntos mudaria a perda no mesmo cálculo tempo como uma única previsão da rede no conjunto de dados.

De volta ao nosso exemplo, este valor $8$ é diferente para diferentes valores de $x$, então faz sentido defini-lo como uma função de $x$. Mais formalmente, esta taxa de variação dependente do valor é referida como a *derivada* que é escrita como

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$
:eqlabel:`eq_der_def`

Textos diferentes usarão notações diferentes para a derivada. Por exemplo, todas as notações abaixo indicam a mesma coisa:
$$
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x.
$$

A maioria dos autores escolherá uma única notação e a manterá, porém nem isso é garantido. É melhor estar familiarizado com tudo isso. Usaremos a notação $\frac{df}{dx}$ ao longo deste texto, a menos que queiramos tirar a derivada de uma expressão complexa, caso em que usaremos $\frac{d}{dx}f$ para escrever expressões como
$$
\frac{d}{dx}\left[x^4+\cos\left(\frac{x^2+1}{2x-1}\right)\right].
$$

Muitas vezes, é intuitivamente útil desvendar a definição de derivada :eqref:`eq_der_def` novamente para ver como uma função muda quando fazemos uma pequena mudança de $x$:

$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \\ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \\ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). \end{aligned}$$
:eqlabel:`eq_small_change`


Vale a pena mencionar explicitamente a última equação. Isso nos diz que se você pegar qualquer função e alterar a entrada em um pequeno valor, a saída mudará nesse pequeno valor escalonado pela derivada.

Desta forma, podemos entender a derivada como o fator de escala que nos diz quão grande é a mudança que obtemos na saída de uma mudança na entrada.

## Regras de Cálculo
:label:`sec_derivative_table`

Agora nos voltamos para a tarefa de entender como calcular a derivada de uma função explícita. Um tratamento formal completo do cálculo derivaria tudo dos primeiros princípios. Não vamos ceder a esta tentação aqui, mas sim fornecer uma compreensão das regras comuns encontradas.

### Derivadas Comuns
Como foi visto em :numref:`sec_calculus`, ao calcular derivadas, muitas vezes pode-se usar uma série de regras para reduzir o cálculo a algumas funções básicas. Nós os repetimos aqui para facilitar a referência.

* **Derivada de constantes.** $\frac{d}{dx}c = 0$.
* **Derivada de funções lineares.** $\frac{d}{dx}(ax) = a$.
* **Regra de potência.** $\frac{d}{dx}x^n = nx^{n-1}$.
* **Derivada de exponenciais.** $\frac{d}{dx}e^x = e^x$.
* **Derivada do logaritmo.** $\frac{d}{dx}\log(x) = \frac{1}{x}$.

### Regras de Derivadas
Se cada derivada precisasse ser calculada separadamente e armazenada em uma tabela, o cálculo diferencial seria quase impossível. É um presente da matemática que podemos generalizar as derivadas acima e calcular derivadas mais complexas, como encontrar a derivada de $f(x) = \log\left(1+(x-1)^{10}\right)$. Como foi mencionado em :numref:`sec_calculus`, a chave para fazer isso é codificar o que acontece quando pegamos funções e as combinamos de várias maneiras, o mais importante: somas, produtos e composições.

* **Regra da soma.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$.
* **Regra do produto.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* **Regra da cadeia.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.

Vamos ver como podemos usar :eqref:`eq_small_change` para entender essas regras. Para a regra da soma, considere a seguinte cadeia de raciocínio:

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\
& \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\
& = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\
& = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right).
\end{aligned}
$$


Comparando este resultado com o fato de que  $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$ vemos que $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$ conforme desejado. A intuição aqui é: quando mudamos a entrada $x$, $g$ e $h$ contribuem conjuntamente para a mudança da saída $\frac{dg}{dx}(x)$ e $\frac{dh}{dx}(x)$.


O produto é mais sutil e exigirá uma nova observação sobre como trabalhar com essas expressões. Começaremos como antes usando :eqref:`eq_small_change`:

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon)\cdot h(x+\epsilon) \\
& \approx \left(g(x) + \epsilon \frac{dg}{dx}(x)\right)\cdot\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& = g(x)\cdot h(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x) \\
& = f(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x). \\
\end{aligned}
$$


Isso se assemelha ao cálculo feito acima, e de fato vemos nossa resposta ($\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$) sentado ao lado de $\epsilon$, mas há a questão desse termo de tamanho $\epsilon^{2}$. Iremos nos referir a isso como um *termo de ordem superior*, uma vez que a potência de $\epsilon^2$ é maior do que a potência de $\epsilon^1$. Veremos em uma seção posterior que às vezes desejaremos mantê-los sob controle; no entanto, por enquanto, observe que se $\epsilon = 0.0000001$, então $\epsilon^{2}= 0.0000000000001$, que é muito menor. Conforme enviamos $\epsilon \rightarrow 0$, podemos ignorar com segurança os termos de pedido superior. Como uma convenção geral neste apêndice, usaremos "$\approx$" para denotar que os dois termos são iguais até os termos de ordem superior. No entanto, se quisermos ser mais formais, podemos examinar o quociente de diferença
$$
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x),
$$


e veja que conforme enviamos $\epsilon \rightarrow 0$, o termo do lado direito vai para zero também.

Finalmente, com a regra da cadeia, podemos novamente progredir como antes usando :eqref:`eq_small_change` e ver que

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x),
\end{aligned}
$$


onde na segunda linha vemos a função $g$ como tendo sua entrada ($h(x)$) deslocada pela pequena quantidade $\epsilon \frac{dh}{dx}(x)$.

Essa regra nos fornece um conjunto flexível de ferramentas para calcular essencialmente qualquer expressão desejada. Por exemplo,

$$
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\
& = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\
& = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\
& = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\
& = \frac{10(x-1)^9}{1+(x-1)^{10}}.
\end{aligned}
$$


Onde cada linha usou as seguintes regras:

1. A regra da cadeia e derivada do logaritmo.
2. A regra da soma.
3. A derivada de constantes, regra da cadeia e regra de potência.
4. A regra da soma, derivada de funções lineares, derivada de constantes.

Duas coisas devem ficar claras depois de fazer este exemplo:


1. Qualquer função que possamos escrever usando somas, produtos, constantes, potências, exponenciais e logaritmos pode ter sua derivada calculada mecanicamente seguindo essas regras.
2. Fazer com que um humano siga essas regras pode ser entediante e sujeito a erros!

Felizmente, esses dois fatos juntos apontam para um caminho a seguir: este é um candidato perfeito para a mecanização! Na verdade, a retropropagação, que revisitaremos mais tarde nesta seção, é exatamente isso.

### Aproximação Linear
Ao trabalhar com derivadas, geralmente é útil interpretar geometricamente a aproximação usada acima. Em particular, observe que a equação
$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x),
$$

approximates the value of $f$ by a line which passes through the point $(x, f(x))$ and has slope $\frac{df}{dx}(x)$.  In this way we say that the derivative gives a linear approximation to the function $f$, as illustrated below:

```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some linear approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) * 
                 tf.cos(tf.constant(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

### Higher Order Derivatives

Let us now do something that may on the surface seem strange.  Take a function $f$ and compute the derivative $\frac{df}{dx}$.  This gives us the rate of change of $f$ at any point.

However, the derivative, $\frac{df}{dx}$, can be viewed as a function itself, so nothing stops us from computing the derivative of $\frac{df}{dx}$ to get $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$.  We will call this the second derivative of $f$.  This function is the rate of change of the rate of change of $f$, or in other words, how the rate of change is changing. We may apply the derivative any number of times to obtain what is called the $n$-th derivative. To keep the notation clean, we will denote the $n$-th derivative as 

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

Let us try to understand *why* this is a useful notion.  Below, we visualize $f^{(2)}(x)$, $f^{(1)}(x)$, and $f(x)$.  

First, consider the case that the second derivative $f^{(2)}(x)$ is a positive constant.  This means that the slope of the first derivative is positive.  As a result, the first derivative $f^{(1)}(x)$ may start out negative, becomes zero at a point, and then becomes positive in the end. This tells us the slope of our original function $f$ and therefore, the function $f$ itself decreases, flattens out, then increases.  In other words, the function $f$ curves up, and has a single minimum as is shown in :numref:`fig_positive-second`.

![If we assume the second derivative is a positive constant, then the fist derivative in increasing, which implies the function itself has a minimum.](../img/posSecDer.svg)
:label:`fig_positive-second`


Second, if the second derivative is a negative constant, that means that the first derivative is decreasing.  This implies the first derivative may start out positive, becomes zero at a point, and then becomes negative. Hence, the function $f$ itself increases, flattens out, then decreases.  In other words, the function $f$ curves down, and has a single maximum as is shown in :numref:`fig_negative-second`.

![If we assume the second derivative is a negative constant, then the fist derivative in decreasing, which implies the function itself has a maximum.](../img/negSecDer.svg)
:label:`fig_negative-second`


Third, if the second derivative is a always zero, then the first derivative will never change---it is constant!  This means that $f$ increases (or decreases) at a fixed rate, and $f$ is itself a straight line  as is shown in :numref:`fig_zero-second`.

![If we assume the second derivative is zero, then the fist derivative is constant, which implies the function itself is a straight line.](../img/zeroSecDer.svg)
:label:`fig_zero-second`

To summarize, the second derivative can be interpreted as describing the way that the function $f$ curves.  A positive second derivative leads to a upwards curve, while a negative second derivative means that $f$ curves downwards, and a zero second derivative means that $f$ does not curve at all.

Let us take this one step further. Consider the function $g(x) = ax^{2}+ bx + c$.  We can then compute that

$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$

If we have some original function $f(x)$ in mind, we may compute the first two derivatives and find the values for $a, b$, and $c$ that make them match this computation.  Similarly to the previous section where we saw that the first derivative gave the best approximation with a straight line, this construction provides the best approximation by a quadratic.  Let us visualize this for $f(x) = \sin(x)$.

```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)) - (xs - x0)**2 *
                 torch.sin(torch.tensor(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) * 
                 tf.cos(tf.constant(x0)) - (xs - x0)**2 *
                 tf.sin(tf.constant(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

We will extend this idea to the idea of a *Taylor series* in the next section. 

### Taylor Series


The *Taylor series* provides a method to approximate the function $f(x)$ if we are given values for the first $n$ derivatives at a point $x_0$, i.e., $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$. The idea will be to find a degree $n$ polynomial that matches all the given derivatives at $x_0$.

We saw the case of $n=2$ in the previous section and a little algebra shows this is

$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

As we can see above, the denominator of $2$ is there to cancel out the $2$ we get when we take two derivatives of $x^2$, while the other terms are all zero.  Same logic applies for the first derivative and the value itself.

If we push the logic further to $n=3$, we will conclude that

$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

where the $6 = 3 \times 2 = 3!$ comes from the constant we get in front if we take three derivatives of $x^3$.


Furthermore, we can get a degree $n$ polynomial by 

$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

where the notation 

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$


Indeed, $P_n(x)$ can be viewed as the best $n$-th degree polynomial approximation to our function $f(x)$.

While we are not going to dive all the way into the error of the above approximations, it is worth mentioning the infinite limit. In this case, for well behaved functions (known as real analytic functions) like $\cos(x)$ or $e^{x}$, we can write out the infinite number of terms and approximate the exactly same function

$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$

Take $f(x) = e^{x}$ as am example. Since $e^{x}$ is its own derivative, we know that $f^{(n)}(x) = e^{x}$. Therefore, $e^{x}$ can be reconstructed by taking the Taylor series at $x_0 = 0$, i.e.,

$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$

Let us see how this works in code and observe how increasing the degree of the Taylor approximation brings us closer to the desired function $e^x$.

```{.python .input}
# Compute the exponential function
xs = np.arange(0, 3, 0.01)
ys = np.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab pytorch
# Compute the exponential function
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab tensorflow
# Compute the exponential function
xs = tf.range(0, 3, 0.01)
ys = tf.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

Taylor series have two primary applications:

1. *Theoretical applications*: Often when we try to understand a too complex function, using Taylor series enables us to turn it into a polynomial that we can work with directly.

2. *Numerical applications*: Some functions like $e^{x}$ or $\cos(x)$ are  difficult for machines to compute.  They can store tables of values at a fixed precision (and this is often done), but it still leaves open questions like "What is the 1000-th digit of $\cos(1)$?"  Taylor series are often helpful to answer such questions.  


## Summary

* Derivatives can be used to express how functions change when we change the input by a small amount.
* Elementary derivatives can be combined using derivative rules to create arbitrarily complex derivatives.
* Derivatives can be iterated to get second or higher order derivatives.  Each increase in order provides more fine grained information on the behavior of the function.
* Using information in the derivatives of a single data example, we can approximate well behaved functions by polynomials obtained from the Taylor series.


## Exercises

1. What is the derivative of $x^3-4x+1$?
2. What is the derivative of $\log(\frac{1}{x})$?
3. True or False: If $f'(x) = 0$ then $f$ has a maximum or minimum at $x$?
4. Where is the minimum of $f(x) = x\log(x)$ for $x\ge0$ (where we assume that $f$ takes the limiting value of $0$ at $f(0)$)?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/412)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1088)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1089)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjIwMTM2MTQ5LC00OTM5MTk4MDYsLTUyNj
E0MzY2MywyMTE2NTYyMjUyLC0xMDI5MjUxNjQ2LC0xMjkwNjIy
OTcyLC0xMTMzOTI4ODMyXX0=
-->
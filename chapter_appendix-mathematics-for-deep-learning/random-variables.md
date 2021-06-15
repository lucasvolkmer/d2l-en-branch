# Variáveis Aleatórias
:label:`sec_random_variables`

Em :numref:`sec_prob` vimos o básico de como trabalhar com variáveis aleatórias discretas, que em nosso caso se referem àquelas variáveis aleatórias que tomam um conjunto finito de valores possíveis, ou os inteiros. Nesta seção, desenvolvemos a teoria das *variáveis aleatórias contínuas*, que são variáveis aleatórias que podem assumir qualquer valor real.

## Variáveis Aleatórias Contínuas

Variáveis aleatórias contínuas são um tópico significativamente mais sutil do que variáveis aleatórias discretas. Uma boa analogia a ser feita é que o salto técnico é comparável ao salto entre adicionar listas de números e integrar funções. Como tal, precisaremos levar algum tempo para desenvolver a teoria.

### Do Discreto ao Contínuo

Para entender os desafios técnicos adicionais encontrados ao trabalhar com variáveis aleatórias contínuas, vamos realizar um experimento de pensamento. Suponha que estejamos jogando um dardo no alvo e queremos saber a probabilidade de que ele atinja exatamente $2 \text{cm}$  do centro do alvo.


Para começar, imaginamos medir um único dígito de precisão, ou seja, com valores para $0 \text{cm}$, $1 \text{cm}$, $2 \text{cm}$ e assim por diante. Jogamos, digamos, $100$ dardos no alvo de dardos, e se $20$ deles caem no valor de $2\text{cm}$, concluímos que $20\%$ dos dardos que jogamos atingem o tabuleiro $2 \text{cm}$ longe do centro.

No entanto, quando olhamos mais de perto, isso não corresponde à nossa pergunta! Queríamos igualdade exata, ao passo que essas caixas mantêm tudo o que está entre, digamos, $1.5\text{cm}$ e $2.5\text{cm}$.


Sem desanimar, continuamos adiante. Medimos ainda mais precisamente, digamos $1.9\text{cm}$, $2.0\text{cm}$, $2.1\text{cm}$, e agora vemos que talvez $3$ dos $100$ dardos atingiram o tabuleiro no $2.0\text{cm}$ balde. Assim, concluímos que a probabilidade é $3\%$.

Porém, isso não resolve nada! Apenas resolvemos o problema um dígito mais. Vamos abstrair um pouco. Imagine que sabemos a probabilidade de que os primeiros $k$ dígitos correspondam a $2.00000\ldots$ e queremos saber a probabilidade de que correspondam aos primeiros $k+1$ dígitos. É bastante razoável supor que o dígito ${k+1}^{\mathrm{th}}$ é essencialmente uma escolha aleatória do conjunto $\{0, 1, 2, \ldots, 9\}$. Pelo menos, não podemos conceber um processo fisicamente significativo que forçaria o número de micrômetros a se afastar do centro a preferir terminar em $7$ vs $3$.

O que isso significa é que, em essência, cada dígito adicional de precisão que exigimos deve diminuir a probabilidade de correspondência por um fator de $10$. Ou dito de outra forma, poderíamos esperar que

$$
P(\text{distance is}\; 2.00\ldots, \;\text{to}\; k \;\text{digits} ) \approx p\cdot10^{-k}.
$$


O valor $p$ essencialmente codifica o que acontece com os primeiros dígitos, e $10^{-k}$ trata do resto.

Observe que, se conhecermos a posição com precisão de $k=4$ dígitos após o decimal. isso significa que sabemos que o valor está dentro do intervalo, digamos $[(1.99995,2.00005]$, que é um intervalo de comprimento $2.00005-1.99995 = 10^{-4}$. Portanto, se chamarmos o comprimento deste intervalo de $\epsilon$, podemos dizer

$$
P(\text{distance is in an}\; \epsilon\text{-sized interval around}\; 2 ) \approx \epsilon \cdot p.
$$

Vamos dar mais um passo final. Temos pensado nesse ponto $2$ o tempo todo, mas nunca pensando em outros pontos. Nada é fundamentalmente diferente aqui, mas é o caso que o valor $p$ provavelmente será diferente. Esperaríamos pelo menos que um arremessador de dardo tivesse mais probabilidade de acertar um ponto próximo ao centro, como $2\text{cm}$. em vez de $20\text{cm}$. Assim, o valor $p$ não é fixo, mas deve depender do ponto $x$. Isso nos diz que devemos esperar

$$P(\text{distance is in an}\; \epsilon \text{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_deriv`

Na verdade, :eqref:`eq_pdf_deriv` define com precisão a *função de densidade de probabilidade*. É uma função $p(x)$ que codifica a probabilidade relativa de acertar perto de um ponto em relação a outro. Vamos visualizar a aparência de tal função.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot the probability density function for some random variable
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2)/np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2)/np.sqrt(2 * np.pi)

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot the probability density function for some random variable
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi)) + \
    0.8*torch.exp(-(x + 1)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi))

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot the probability density function for some random variable
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi)) + \
    0.8*tf.exp(-(x + 1)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi))

d2l.plot(x, p, 'x', 'Density')
```

Os locais onde o valor da função é grande indicam regiões onde é mais provável encontrar o valor aleatório. As porções baixas são áreas onde é improvável que encontremos o valor aleatório.

### Funções de Densidade de Probabilidade

Vamos agora investigar isso mais detalhadamente. Já vimos o que é uma função de densidade de probabilidade intuitivamente para uma variável aleatória $X$, ou seja, a função de densidade é uma função $p(x)$ de modo que

$$P(X \; \text{is in an}\; \epsilon \text{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_def`


Mas o que isso implica para as propriedades de $p(x)$?

Primeiro, as probabilidades nunca são negativas, portanto, devemos esperar que $p(x) \ge 0$  também.

Em segundo lugar, vamos imaginar que dividimos $\mathbb{R}$ em um número infinito de fatias de $\epsilon$ de largura, digamos com fatias $(\epsilon\cdot i, \epsilon \cdot (i+1)]$. Para cada um deles, sabemos de :eqref:`eq_pdf_def` que a probabilidade é de aproximadamente

$$
P(X \; \text{is in an}\; \epsilon\text{-sized interval around}\; x ) \approx \epsilon \cdot p(\epsilon \cdot i),
$$

então, resumido sobre todos eles, deveria ser

$$
P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i).
$$

Isso nada mais é do que a aproximação de uma integral discutida em :numref:`sec_integral_calculus`, portanto, podemos dizer que

$$
P(X\in\mathbb{R}) = \int_{-\infty}^{\infty} p(x) \; dx.
$$

Sabemos que $P(X\in\mathbb{R}) = 1$, uma vez que a variável aleatória deve assumir *algum* número, podemos concluir que para qualquer densidade

$$
\int_{-\infty}^{\infty} p(x) \; dx = 1.
$$

Na verdade, aprofundarmos mais nisso mostra que para qualquer $a$ e $b$, vemos que

$$
P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.
$$

Podemos aproximar isso no código usando os mesmos métodos de aproximação discretos de antes. Nesse caso, podemos aproximar a probabilidade de cair na região azul.

```{.python .input}
# Approximate probability using numerical integration
epsilon = 0.01
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {np.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab pytorch
# Approximate probability using numerical integration
epsilon = 0.01
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi)) +\
    0.8*torch.exp(-(x + 1)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {torch.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab tensorflow
# Approximate probability using numerical integration
epsilon = 0.01
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi)) +\
    0.8*tf.exp(-(x + 1)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.numpy().tolist()[300:800], p.numpy().tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {tf.reduce_sum(epsilon*p[300:800])}'
```

Acontece que essas duas propriedades descrevem exatamente o espaço das funções de densidade de probabilidade possíveis (ou *f.d.f.* 'S para a abreviação comumente encontrada). Eles são funções não negativas $p(x) \ge 0$ tais que

$$\int_{-\infty}^{\infty} p(x) \; dx = 1.$$
:eqlabel:`eq_pdf_int_one`

Interpretamos essa função usando integração para obter a probabilidade de nossa variável aleatória estar em um intervalo específico:

$$P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.$$
:eqlabel:`eq_pdf_int_int`

Em :numref:`sec_distributions` veremos uma série de distribuições comuns, mas vamos continuar trabalhando de forma abstrata.

### Funções de Distribuição Cumulativas


Na seção anterior, vimos a noção de f.d.p. Na prática, esse é um método comumente encontrado para discutir variáveis aleatórias contínuas, mas tem uma armadilha significativa: os valores de f.d.p. não são em si probabilidades, mas sim uma função que devemos integrar para produzir probabilidades. Não há nada de errado em uma densidade ser maior que $10$, desde que não seja maior que $10$ por mais de um intervalo de comprimento $1/10$. Isso pode ser contra-intuitivo, então as pessoas muitas vezes também pensam em termos da *função de distribuição cumulativa*, ou f.d.c., que *é* uma probabilidade.

Em particular, usando :eqref:`eq_pdf_int_int`, definimos o f.d.c. para uma variável aleatória $X$ com densidade $p(x)$ por

$$
F(x) = \int _ {-\infty}^{x} p(x) \; dx = P(X \le x).
$$

Vamos observar algumas propriedades.

* $F(x) \rightarrow 0$ as $x\rightarrow -\infty$.
* $F(x) \rightarrow 1$ as $x\rightarrow \infty$.
* $F(x)$ is non-decreasing ($y > x \implies F(y) \ge F(x)$).
* $F(x)$ is continuous (has no jumps) if $X$ is a continuous random variable.

Com o quarto marcador, observe que isso não seria verdade se $X$ fosse discreto, digamos, tomando os valores $0$ e $1$ ambos com probabilidade $1/2$. Nesse caso

$$
F(x) = \begin{cases}
0 & x < 0, \\
\frac{1}{2} & x < 1, \\
1 & x \ge 1.
\end{cases}
$$

Neste exemplo, vemos um dos benefícios de trabalhar com o cdf, a capacidade de lidar com variáveis aleatórias contínuas ou discretas na mesma estrutura, ou mesmo misturas das duas (lance uma moeda: se cara retornar o lançamento de um dado , se a cauda retornar a distância de um lançamento de dardo do centro de um alvo de dardo).

### Médias


Suponha que estamos lidando com variáveis aleatórias $X$. A distribuição em si pode ser difícil de interpretar. Muitas vezes é útil ser capaz de resumir o comportamento de uma variável aleatória de forma concisa. Os números que nos ajudam a capturar o comportamento de uma variável aleatória são chamados de *estatísticas resumidas*. Os mais comumente encontrados são a *média*, a *variância* e o *desvio padrão*.

A *média* codifica o valor médio de uma variável aleatória. Se tivermos uma variável aleatória discreta $X$, que assume os valores $x_i$ com probabilidades $p_i$, então a média é dada pela média ponderada: some os valores vezes a probabilidade de que a variável aleatória assuma esse valor:

$$\mu_X = E[X] = \sum_i x_i p_i.$$
:eqlabel:`eq_exp_def`


A maneira como devemos interpretar a média (embora com cautela) é que ela nos diz essencialmente onde a variável aleatória tende a estar localizada.

Como um exemplo minimalista que examinaremos ao longo desta seção, tomemos $X$ como a variável aleatória que assume o valor $a-2$ com probabilidade $p$, $a+2$ com probabilidade $p$ e $a$ com probabilidade $1-2p$. Podemos calcular usando :eqref:`eq_exp_def` que, para qualquer escolha possível de $a$ e $p$, a média é

$$
\mu_X = E[X] = \sum_i x_i p_i = (a-2)p + a(1-2p) + (a+2)p = a.
$$


Assim, vemos que a média é $a$. Isso corresponde à intuição, já que $a$ é o local em torno do qual centralizamos nossa variável aleatória.

Por serem úteis, vamos resumir algumas propriedades.

* Para qualquer variável aleatória $X$ e números $a$ e $b$, temos que $\mu_{aX+b} = a\mu_X + b$.
* Se temos duas variáveis aleatórias $X$ e $Y$, temos $\mu_{X+Y} = \mu_X+\mu_Y$.

As médias são úteis para entender o comportamento médio de uma variável aleatória, porém a média não é suficiente para ter um entendimento intuitivo completo. Ter um lucro de $\$10 \pm \$1$ por venda é muito diferente de ganhar $\$10 \pm \$15$ por venda, apesar de ter o mesmo valor médio. O segundo tem um grau de flutuação muito maior e, portanto, representa um risco muito maior. Assim, para entender o comportamento de uma variável aleatória, precisaremos, no mínimo, de mais uma medida: alguma medida de quão amplamente uma variável aleatória flutua.

### Variâncias


Isso nos leva a considerar a *variância* de uma variável aleatória. Esta é uma medida quantitativa de quão longe uma variável aleatória se desvia da média. Considere a expressão $X - \mu_X$. Este é o desvio da variável aleatória de sua média. Esse valor pode ser positivo ou negativo, portanto, precisamos fazer algo para torná-lo positivo para que possamos medir a magnitude do desvio.

Uma coisa razoável a tentar é olhar para $\left|X-\mu_X\right|$, e de fato isso leva a uma quantidade útil chamada *desvio médio absoluto*, no entanto, devido a conexões com outras áreas da matemática e estatística, as pessoas costumam usar uma solução diferente.

Em particular, eles olham para $(X-\mu_X)^2.$. Se olharmos para o tamanho típico desta quantidade tomando a média, chegamos à variância

$$\sigma_X^2 = \mathrm{Var}(X) = E\left[(X-\mu_X)^2\right] = E[X^2] - \mu_X^2.$$
:eqlabel:`eq_var_def`


A última igualdade em :eqref:`eq_var_def` se mantém expandindo a definição no meio e aplicando as propriedades de expectativa.

Vejamos nosso exemplo onde $X$ é a variável aleatória que assume o valor $a-2$ com probabilidade $p$, $a+2$ com probabilidade $p$ e $a$ com probabilidade $1-2p$. Neste caso $\mu_X = a$, então tudo que precisamos calcular é $E\left[X^2\right]$. Isso pode ser feito prontamente:

$$
E\left[X^2\right] = (a-2)^2p + a^2(1-2p) + (a+2)^2p = a^2 + 8p.
$$

Assim, vemos que por :eqref:`eq_var_def` nossa variância é
$$
\sigma_X^2 = \mathrm{Var}(X) = E[X^2] - \mu_X^2 = a^2 + 8p - a^2 = 8p.
$$


Este resultado novamente faz sentido. O maior $p$ pode ser $1/2$ que corresponde a escolher $a-2$ ou $a+2$ com um cara ou coroa. A variação de ser $4$ corresponde ao fato de que $a-2$ e $a+2$ estão $2$ unidades de distância da média e $2^2 = 4$. Na outra extremidade do espectro, se $p=0$, essa variável aleatória sempre assume o valor $0$ e, portanto, não tem nenhuma variação.

Listaremos algumas propriedades de variação abaixo:

* Para qualquer variável aleatória $X$, $\mathrm{Var}(X) \ge 0$, com $\mathrm{Var}(X) = 0$ se e somente se $X$ for uma constante.
* Para qualquer variável aleatória $X$ e números $a$ e $b$, temos que $\mathrm{Var}(aX+b) = a^2\mathrm{Var}(X)$.
* Se temos duas variáveis aleatórias *independentes* $X$ e $Y$, temos $\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$.

Ao interpretar esses valores, pode haver um pequeno soluço. Em particular, vamos tentar imaginar o que acontece se rastrearmos as unidades por meio desse cálculo. Suponha que estejamos trabalhando com a classificação por estrelas atribuída a um produto na página da web. Então $a$, $a-2$ e $a+2$ são medidos em unidades de estrelas. Da mesma forma, a média $\mu_X$ também é medida em estrelas (sendo uma média ponderada). No entanto, se chegarmos à variância, imediatamente encontramos um problema, que é queremos olhar para $(X-\mu_X)^2$, que está em unidades de *estrelas ao quadrado*. Isso significa que a própria variação não é comparável às medições originais. Para torná-lo interpretável, precisaremos retornar às nossas unidades originais.

### Desvio Padrão

Essas estatísticas resumidas sempre podem ser deduzidas da variância calculando a raiz quadrada! Assim, definimos o *desvio padrão* como sendo

$$
\sigma_X = \sqrt{\mathrm{Var}(X)}.
$$


Em nosso exemplo, isso significa que agora temos o desvio padrão $\sigma_X = 2\sqrt{2p}$. Se estamos lidando com unidades de estrelas para nosso exemplo de revisão, $\sigma_X$ está novamente em unidades de estrelas.

As propriedades que tínhamos para a variância podem ser reapresentadas para o desvio padrão.

* Para qualquer variável aleatória $X$, $\sigma_{X} \ge 0$.
* Para qualquer variável aleatória $X$ e números $a$ e $b$, temos que $\sigma_{aX+b} = |a|\sigma_{X}$
* Se temos duas variáveis aleatórias *independentes* $X$ e $Y$, temos $\sigma_{X+Y} = \sqrt{\sigma_{X}^2 + \sigma_{Y}^2}$.

É natural, neste momento, perguntar: "Se o desvio padrão está nas unidades de nossa variável aleatória original, ele representa algo que podemos desenhar em relação a essa variável aleatória?" A resposta é um sim retumbante! Na verdade, muito parecido com a média informada sobre a localização típica de nossa variável aleatória, o desvio padrão fornece a faixa típica de variação dessa variável aleatória. Podemos tornar isso rigoroso com o que é conhecido como desigualdade de Chebyshev:

$$P\left(X \not\in [\mu_X - \alpha\sigma_X, \mu_X + \alpha\sigma_X]\right) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_chebyshev`


Ou, para declará-lo verbalmente no caso de $\alpha=10$, $99\%$ das amostras de qualquer variável aleatória se enquadram nos desvios padrão de $10$ da média. Isso dá uma interpretação imediata de nossas estatísticas de resumo padrão.

Para ver como essa afirmação é bastante sutil, vamos dar uma olhada em nosso exemplo em execução novamente, onde $X$ é a variável aleatória que assume o valor $a-2$ com probabilidade $p$, $a+2$ com probabilidade $p$ e $a$ com probabilidade $1-2p$. Vimos que a média era $a$ e o desvio padrão era $2\sqrt{2p}$. Isso significa que, se tomarmos a desigualdade de Chebyshev :eqref:`eq_chebyshev` com $\alpha = 2$, vemos que a expressão é

$$
P\left(X \not\in [a - 4\sqrt{2p}, a + 4\sqrt{2p}]\right) \le \frac{1}{4}.
$$

Isso significa que $75\%$ do tempo, essa variável aleatória ficará dentro desse intervalo para qualquer valor de $p$. Agora, observe que como $p \rightarrow 0$, este intervalo também converge para o único ponto $a$. Mas sabemos que nossa variável aleatória assume os valores $a-2, a$ e $a+2$ apenas então, eventualmente, podemos ter certeza de que $a-2$ e $a+2$ ficarão fora do intervalo! A questão é: em que $p$ isso acontece. Então, queremos resolver: para que $p$ $a+4\sqrt{2p} = a+2$, que é resolvido quando $p=1/8$, que é *exatamente* o primeiro $p$ onde isso poderia acontecer sem violar nossa alegação de que não mais do que $1/4$  de amostras da distribuição ficariam fora do intervalo ($1/8$ à esquerda e $1/8$ à direita).

Vamos visualizar isso. Mostraremos a probabilidade de obter os três valores como três barras verticais com altura proporcional à probabilidade. O intervalo será desenhado como uma linha horizontal no meio. O primeiro gráfico mostra o que acontece para $p>1/8$ onde o intervalo contém com segurança todos os pontos.

```{.python .input}
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * np.sqrt(2 * p),
                   a + 4 * np.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, 0.2)
```

```{.python .input}
#@tab pytorch
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * torch.sqrt(2 * p),
                   a + 4 * torch.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, torch.tensor(0.2))
```

```{.python .input}
#@tab tensorflow
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * tf.sqrt(2 * p),
                   a + 4 * tf.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, tf.constant(0.2))
```

O segundo mostra que em $p = 1/8$, o intervalo toca exatamente os dois pontos. Isso mostra que a desigualdade é *nítida*, uma vez que nenhum intervalo menor pode ser obtido mantendo a desigualdade verdadeira.

```{.python .input}
# Plot interval when p = 1/8
plot_chebyshev(0.0, 0.125)
```

```{.python .input}
#@tab pytorch
# Plot interval when p = 1/8
plot_chebyshev(0.0, torch.tensor(0.125))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p = 1/8
plot_chebyshev(0.0, tf.constant(0.125))
```

The third shows that for $p < 1/8$ the interval only contains the center.  This does not invalidate the inequality since we only needed to ensure that no more than $1/4$ of the probability falls outside the interval, which means that once $p < 1/8$, the two points at $a-2$ and $a+2$ can be discarded.

```{.python .input}
# Plot interval when p < 1/8
plot_chebyshev(0.0, 0.05)
```

```{.python .input}
#@tab pytorch
# Plot interval when p < 1/8
plot_chebyshev(0.0, torch.tensor(0.05))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p < 1/8
plot_chebyshev(0.0, tf.constant(0.05))
```

### Means and Variances in the Continuum

This has all been in terms of discrete random variables, but the case of continuous random variables is similar.  To intuitively understand how this works, imagine that we split the real number line into intervals of length $\epsilon$ given by $(\epsilon i, \epsilon (i+1)]$.  Once we do this, our continuous random variable has been made discrete and we can use :eqref:`eq_exp_def` say that

$$
\begin{aligned}
\mu_X & \approx \sum_{i} (\epsilon i)P(X \in (\epsilon i, \epsilon (i+1)]) \\
& \approx \sum_{i} (\epsilon i)p_X(\epsilon i)\epsilon, \\
\end{aligned}
$$

where $p_X$ is the density of $X$.  This is an approximation to the integral of $xp_X(x)$, so we can conclude that

$$
\mu_X = \int_{-\infty}^\infty xp_X(x) \; dx.
$$

Similarly, using :eqref:`eq_var_def` the variance can be written as

$$
\sigma^2_X = E[X^2] - \mu_X^2 = \int_{-\infty}^\infty x^2p_X(x) \; dx - \left(\int_{-\infty}^\infty xp_X(x) \; dx\right)^2.
$$

Everything stated above about the mean, the variance, and the standard deviation still applies in this case.  For instance, if we consider the random variable with density

$$
p(x) = \begin{cases}
1 & x \in [0,1], \\
0 & \text{otherwise}.
\end{cases}
$$

we can compute

$$
\mu_X = \int_{-\infty}^\infty xp(x) \; dx = \int_0^1 x \; dx = \frac{1}{2}.
$$

and

$$
\sigma_X^2 = \int_{-\infty}^\infty x^2p(x) \; dx - \left(\frac{1}{2}\right)^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}.
$$

As a warning, let us examine one more example, known as the *Cauchy distribution*.  This is the distribution with p.d.f. given by

$$
p(x) = \frac{1}{1+x^2}.
$$

```{.python .input}
# Plot the Cauchy distribution p.d.f.
x = np.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
# Plot the Cauchy distribution p.d.f.
x = torch.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
# Plot the Cauchy distribution p.d.f.
x = tf.range(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

This function looks innocent, and indeed consulting a table of integrals will show it has area one under it, and thus it defines a continuous random variable.

To see what goes astray, let us try to compute the variance of this.  This would involve using :eqref:`eq_var_def` computing

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx.
$$

The function on the inside looks like this:

```{.python .input}
# Plot the integrand needed to compute the variance
x = np.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab pytorch
# Plot the integrand needed to compute the variance
x = torch.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab tensorflow
# Plot the integrand needed to compute the variance
x = tf.range(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

This function clearly has infinite area under it since it is essentially the constant one with a small dip near zero, and indeed we could show that

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx = \infty.
$$

This means it does not have a well-defined finite variance.

However, looking deeper shows an even more disturbing result.  Let us try to compute the mean using :eqref:`eq_exp_def`.  Using the change of variables formula, we see

$$
\mu_X = \int_{-\infty}^{\infty} \frac{x}{1+x^2} \; dx = \frac{1}{2}\int_1^\infty \frac{1}{u} \; du.
$$

The integral inside is the definition of the logarithm, so this is in essence $\log(\infty) = \infty$, so there is no well-defined average value either!

Machine learning scientists define their models so that we most often do not need to deal with these issues, and will in the vast majority of cases deal with random variables with well-defined means and variances.  However, every so often random variables with *heavy tails* (that is those random variables where the probabilities of getting large values are large enough to make things like the mean or variance undefined) are helpful in modeling physical systems, thus it is worth knowing that they exist.

### Joint Density Functions

The above work all assumes we are working with a single real valued random variable.  But what if we are dealing with two or more potentially highly correlated random variables?  This circumstance is the norm in machine learning: imagine random variables like $R_{i, j}$ which encode the red value of the pixel at the $(i, j)$ coordinate in an image, or $P_t$ which is a random variable given by a stock price at time $t$.  Nearby pixels tend to have similar color, and nearby times tend to have similar prices.  We cannot treat them as separate random variables, and expect to create a successful model (we will see in :numref:`sec_naive_bayes` a model that under-performs due to such an assumption).  We need to develop the mathematical language to handle these correlated continuous random variables.

Thankfully, with the multiple integrals in :numref:`sec_integral_calculus` we can develop such a language.  Suppose that we have, for simplicity, two random variables $X, Y$ which can be correlated.  Then, similar to the case of a single variable, we can ask the question:

$$
P(X \;\text{is in an}\; \epsilon \text{-sized interval around}\; x \; \text{and} \;Y \;\text{is in an}\; \epsilon \text{-sized interval around}\; y ).
$$

Similar reasoning to the single variable case shows that this should be approximately

$$
P(X \;\text{is in an}\; \epsilon \text{-sized interval around}\; x \; \text{and} \;Y \;\text{is in an}\; \epsilon \text{-sized interval around}\; y ) \approx \epsilon^{2}p(x, y),
$$

for some function $p(x, y)$.  This is referred to as the joint density of $X$ and $Y$.  Similar properties are true for this as we saw in the single variable case. Namely:

* $p(x, y) \ge 0$;
* $\int _ {\mathbb{R}^2} p(x, y) \;dx \;dy = 1$;
* $P((X, Y) \in \mathcal{D}) = \int _ {\mathcal{D}} p(x, y) \;dx \;dy$.

In this way, we can deal with multiple, potentially correlated random variables.  If we wish to work with more than two random variables, we can extend the multivariate density to as many coordinates as desired by considering $p(\mathbf{x}) = p(x_1, \ldots, x_n)$.  The same properties of being non-negative, and having total integral of one still hold.

### Marginal Distributions
When dealing with multiple variables, we oftentimes want to be able to ignore the relationships and ask, "how is this one variable distributed?"  Such a distribution is called a *marginal distribution*.

To be concrete, let us suppose that we have two random variables $X, Y$ with joint density given by $p _ {X, Y}(x, y)$.  We will be using the subscript to indicate what random variables the density is for.  The question of finding the marginal distribution is taking this function, and using it to find $p _ X(x)$.

As with most things, it is best to return to the intuitive picture to figure out what should be true.  Recall that the density is the function $p _ X$ so that

$$
P(X \in [x, x+\epsilon]) \approx \epsilon \cdot p _ X(x).
$$

There is no mention of $Y$, but if all we are given is $p _{X, Y}$, we need to include $Y$ somehow. We can first observe that this is the same as

$$
P(X \in [x, x+\epsilon] \text{, and } Y \in \mathbb{R}) \approx \epsilon \cdot p _ X(x).
$$

Our density does not directly tell us about what happens in this case, we need to split into small intervals in $y$ as well, so we can write this as

$$
\begin{aligned}
\epsilon \cdot p _ X(x) & \approx \sum _ {i} P(X \in [x, x+\epsilon] \text{, and } Y \in [\epsilon \cdot i, \epsilon \cdot (i+1)]) \\
& \approx \sum _ {i} \epsilon^{2} p _ {X, Y}(x, \epsilon\cdot i).
\end{aligned}
$$

![By summing along the columns of our array of probabilities, we are able to obtain the marginal distribution for just the random variable represented along the $x$-axis.](../img/marginal.svg)
:label:`fig_marginal`

This tells us to add up the value of the density along a series of squares in a line as is shown in :numref:`fig_marginal`.  Indeed, after canceling one factor of epsilon from both sides, and recognizing the sum on the right is the integral over $y$, we can conclude that

$$
\begin{aligned}
 p _ X(x) &  \approx \sum _ {i} \epsilon p _ {X, Y}(x, \epsilon\cdot i) \\
 & \approx \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
\end{aligned}
$$

Thus we see

$$
p _ X(x) = \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
$$

This tells us that to get a marginal distribution, we integrate over the variables we do not care about.  This process is often referred to as *integrating out* or *marginalized out* the unneeded variables.

### Covariance

When dealing with multiple random variables, there is one additional summary statistic which is helpful to know: the *covariance*.  This measures the degree that two random variable fluctuate together.

Suppose that we have two random variables $X$ and $Y$, to begin with, let us suppose they are discrete, taking on values $(x_i, y_j)$ with probability $p_{ij}$.  In this case, the covariance is defined as

$$\sigma_{XY} = \mathrm{Cov}(X, Y) = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij}. = E[XY] - E[X]E[Y].$$
:eqlabel:`eq_cov_def`

To think about this intuitively: consider the following pair of random variables.  Suppose that $X$ takes the values $1$ and $3$, and $Y$ takes the values $-1$ and $3$.  Suppose that we have the following probabilities

$$
\begin{aligned}
P(X = 1 \; \text{and} \; Y = -1) & = \frac{p}{2}, \\
P(X = 1 \; \text{and} \; Y = 3) & = \frac{1-p}{2}, \\
P(X = 3 \; \text{and} \; Y = -1) & = \frac{1-p}{2}, \\
P(X = 3 \; \text{and} \; Y = 3) & = \frac{p}{2},
\end{aligned}
$$

where $p$ is a parameter in $[0,1]$ we get to pick.  Notice that if $p=1$ then they are both always their minimum or maximum values simultaneously, and if $p=0$ they are guaranteed to take their flipped values simultaneously (one is large when the other is small and vice versa).  If $p=1/2$, then the four possibilities are all equally likely, and neither should be related.  Let us compute the covariance.  First, note $\mu_X = 2$ and $\mu_Y = 1$, so we may compute using :eqref:`eq_cov_def`:

$$
\begin{aligned}
\mathrm{Cov}(X, Y) & = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij} \\
& = (1-2)(-1-1)\frac{p}{2} + (1-2)(3-1)\frac{1-p}{2} + (3-2)(-1-1)\frac{1-p}{2} + (3-2)(3-1)\frac{p}{2} \\
& = 4p-2.
\end{aligned}
$$

When $p=1$ (the case where they are both maximally positive or negative at the same time) has a covariance of $2$. When $p=0$ (the case where they are flipped) the covariance is $-2$.  Finally, when $p=1/2$ (the case where they are unrelated), the covariance is $0$.  Thus we see that the covariance measures how these two random variables are related.

A quick note on the covariance is that it only measures these linear relationships.  More complex relationships like $X = Y^2$ where $Y$ is randomly chosen from $\{-2, -1, 0, 1, 2\}$ with equal probability can be missed.  Indeed a quick computation shows that these random variables have covariance zero, despite one being a deterministic function of the other.

For continuous random variables, much the same story holds.  At this point, we are pretty comfortable with doing the transition between discrete and continuous, so we will provide the continuous analogue of :eqref:`eq_cov_def` without any derivation.

$$
\sigma_{XY} = \int_{\mathbb{R}^2} (x-\mu_X)(y-\mu_Y)p(x, y) \;dx \;dy.
$$

For visualization, let us take a look at a collection of random variables with tunable covariance.

```{.python .input}
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = covs[i]*X + np.random.normal(0, 1, (500))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = covs[i]*X + torch.randn(500)

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = covs[i]*X + tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

Let us see some properties of covariances:

* For any random variable $X$, $\mathrm{Cov}(X, X) = \mathrm{Var}(X)$.
* For any random variables $X, Y$ and numbers $a$ and $b$, $\mathrm{Cov}(aX+b, Y) = \mathrm{Cov}(X, aY+b) = a\mathrm{Cov}(X, Y)$.
* If $X$ and $Y$ are independent then $\mathrm{Cov}(X, Y) = 0$.

In addition, we can use the covariance to expand a relationship we saw before.  Recall that is $X$ and $Y$ are two independent random variables then

$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y).
$$

With knowledge of covariances, we can expand this relationship.  Indeed, some algebra can show that in general,

$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X, Y).
$$

This allows us to generalize the variance summation rule for correlated random variables.

### Correlation

As we did in the case of means and variances, let us now consider units.  If $X$ is measured in one unit (say inches), and $Y$ is measured in another (say dollars), the covariance is measured in the product of these two units $\text{inches} \times \text{dollars}$.  These units can be hard to interpret.  What we will often want in this case is a unit-less measurement of relatedness.  Indeed, often we do not care about exact quantitative correlation, but rather ask if the correlation is in the same direction, and how strong the relationship is.

To see what makes sense, let us perform a thought experiment.  Suppose that we convert our random variables in inches and dollars to be in inches and cents.  In this case the random variable $Y$ is multiplied by $100$.  If we work through the definition, this means that $\mathrm{Cov}(X, Y)$ will be multiplied by $100$.  Thus we see that in this case a change of units change the covariance by a factor of $100$.  Thus, to find our unit-invariant measure of correlation, we will need to divide by something else that also gets scaled by $100$.  Indeed we have a clear candidate, the standard deviation!  Indeed if we define the *correlation coefficient* to be

$$\rho(X, Y) = \frac{\mathrm{Cov}(X, Y)}{\sigma_{X}\sigma_{Y}},$$
:eqlabel:`eq_cor_def`

we see that this is a unit-less value.  A little mathematics can show that this number is between $-1$ and $1$ with $1$ meaning maximally positively correlated, whereas $-1$ means maximally negatively correlated.

Returning to our explicit discrete example above, we can see that $\sigma_X = 1$ and $\sigma_Y = 2$, so we can compute the correlation between the two random variables using :eqref:`eq_cor_def` to see that

$$
\rho(X, Y) = \frac{4p-2}{1\cdot 2} = 2p-1.
$$

This now ranges between $-1$ and $1$ with the expected behavior of $1$ meaning most correlated, and $-1$ meaning minimally correlated.

As another example, consider $X$ as any random variable, and $Y=aX+b$ as any linear deterministic function of $X$.  Then, one can compute that

$$\sigma_{Y} = \sigma_{aX+b} = |a|\sigma_{X},$$

$$\mathrm{Cov}(X, Y) = \mathrm{Cov}(X, aX+b) = a\mathrm{Cov}(X, X) = a\mathrm{Var}(X),$$

and thus by :eqref:`eq_cor_def` that

$$
\rho(X, Y) = \frac{a\mathrm{Var}(X)}{|a|\sigma_{X}^2} = \frac{a}{|a|} = \mathrm{sign}(a).
$$

Thus we see that the correlation is $+1$ for any $a > 0$, and $-1$ for any $a < 0$ illustrating that correlation measures the degree and directionality the two random variables are related, not the scale that the variation takes.

Let us again plot a collection of random variables with tunable correlation.

```{.python .input}
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = cors[i] * X + np.sqrt(1 - cors[i]**2) * np.random.normal(0, 1, 500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = cors[i] * X + torch.sqrt(torch.tensor(1) -
                                 cors[i]**2) * torch.randn(500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = cors[i] * X + tf.sqrt(tf.constant(1.) -
                                 cors[i]**2) * tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

Let us list a few properties of the correlation below.

* For any random variable $X$, $\rho(X, X) = 1$.
* For any random variables $X, Y$ and numbers $a$ and $b$, $\rho(aX+b, Y) = \rho(X, aY+b) = \rho(X, Y)$.
* If $X$ and $Y$ are independent with non-zero variance then $\rho(X, Y) = 0$.

As a final note, you may feel like some of these formulae are familiar.  Indeed, if we expand everything out assuming that $\mu_X = \mu_Y = 0$, we see that this is

$$
\rho(X, Y) = \frac{\sum_{i, j} x_iy_ip_{ij}}{\sqrt{\sum_{i, j}x_i^2 p_{ij}}\sqrt{\sum_{i, j}y_j^2 p_{ij}}}.
$$

This looks like a sum of a product of terms divided by the square root of sums of terms.  This is exactly the formula for the cosine of the angle between two vectors $\mathbf{v}, \mathbf{w}$ with the different coordinates weighted by $p_{ij}$:

$$
\cos(\theta) = \frac{\mathbf{v}\cdot \mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|} = \frac{\sum_{i} v_iw_i}{\sqrt{\sum_{i}v_i^2}\sqrt{\sum_{i}w_i^2}}.
$$

Indeed if we think of norms as being related to standard deviations, and correlations as being cosines of angles, much of the intuition we have from geometry can be applied to thinking about random variables.

## Summary
* Continuous random variables are random variables that can take on a continuum of values.  They have some technical difficulties that make them more challenging to work with compared to discrete random variables.
* The probability density function allows us to work with continuous random variables by giving a function where the area under the curve on some interval gives the probability of finding a sample point in that interval.
* The cumulative distribution function is the probability of observing the random variable to be less than a given threshold.  It can provide a useful alternate viewpoint which unifies discrete and continuous variables.
* The mean is the average value of a random variable.
* The variance is the expected square of the difference between the random variable and its mean.
* The standard deviation is the square root of the variance.  It can be thought of as measuring the range of values the random variable may take.
* Chebyshev's inequality allows us to make this intuition rigorous by giving an explicit interval that contains the random variable most of the time.
* Joint densities allow us to work with correlated random variables.  We may marginalize joint densities by integrating over unwanted random variables to get the distribution of the desired random variable.
* The covariance and correlation coefficient provide a way to measure any linear relationship between two correlated random variables.

## Exercises
1. Suppose that we have the random variable with density given by $p(x) = \frac{1}{x^2}$ for $x \ge 1$ and $p(x) = 0$ otherwise.  What is $P(X > 2)$?
2. The Laplace distribution is a random variable whose density is given by $p(x = \frac{1}{2}e^{-|x|}$.  What is the mean and the standard deviation of this function?  As a hint, $\int_0^\infty xe^{-x} \; dx = 1$ and $\int_0^\infty x^2e^{-x} \; dx = 2$.
3. I walk up to you on the street and say "I have a random variable with mean $1$, standard deviation $2$, and I observed $25\%$ of my samples taking a value larger than $9$."  Do you believe me?  Why or why not?
4. Suppose that you have two random variables $X, Y$, with joint density given by $p_{XY}(x, y) = 4xy$ for $x, y \in [0,1]$ and $p_{XY}(x, y) = 0$ otherwise.  What is the covariance of $X$ and $Y$?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/415)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1094)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1095)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0NjU2MzgxNTMsNTU1OTA0MDYsLTE5Mz
A0MDY2MjEsMzk0MDI0MTc0LC0xNjg3NTk1NTcxLC01MTY3OTMy
NzksMTcyOTg3ODg0LDE2NDc1MTY2MjEsNDM3ODU2NTMxLDY3MD
M2MDQwNCwyMTI0NjQ1OTU5LC00NDUwNTgyOTZdfQ==
-->
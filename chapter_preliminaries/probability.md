# Probabilitydade
:label:`sec_prob`

In some form or another, machine learning is all about making predictions.
We might want to predict the *probability* of a patient suffering a heart attack in the next year, given their clinical history. In anomaly detection, we might want to assess how *likely* a set of readings from an airplane's jet engine would be, were it operating normally. In reinforcement learning, we want an agent to act intelligently in an environment. This means we need to think about the probability of getting a high reward under each of the available actions. And when we build recommender systems we also need to think about probability. For example, say *hypothetically* that we worked for a large online bookseller. We might want to estimate the probability that a particular user would buy a particular book. For this we need to use the language of probability.
Entire courses, majors, theses, careers, and even departments, are devoted to probability. So naturally, our goal in this section is not to teach the whole subject. Instead we hope to get you off the ground, to teach you just enough that you can start building your first deep learning models, and to give you enough of a flavor for the subject that you can begin to explore it on your own if you wish.

De uma forma ou de outra, o aprendizado de máquina envolve fazer previsões.
Podemos querer prever a *probabilidade* de um paciente sofrer um ataque cardíaco no próximo ano, considerando sua história clínica. Na detecção de anomalias, podemos avaliar quão * provável * seria um conjunto de leituras do motor a jato de um avião, se ele estivesse operando normalmente. Na aprendizagem por reforço, queremos que um agente aja de forma inteligente em um ambiente. Isso significa que precisamos pensar sobre a probabilidade de obter uma alta recompensa em cada uma das ações disponíveis. E quando construímos sistemas de recomendação, também precisamos pensar sobre probabilidade. Por exemplo, diga *hipoteticamente* que trabalhamos para uma grande livraria online. Podemos querer estimar a probabilidade de um determinado usuário comprar um determinado livro. Para isso, precisamos usar a linguagem da probabilidade.
Cursos inteiros, majores, teses, carreiras e até departamentos são dedicados à probabilidade. Então, naturalmente, nosso objetivo nesta seção não é ensinar todo o assunto. Em vez disso, esperamos fazer você decolar, ensinar apenas o suficiente para que você possa começar a construir seus primeiros modelos de aprendizagem profunda e dar-lhe um sabor suficiente para o assunto que você pode começar a explorá-lo por conta própria, se desejar.

We have already invoked probabilities in previous sections without articulating what precisely they are or giving a concrete example. Let us get more serious now by considering the first case: distinguishing cats and dogs based on photographs. This might sound simple but it is actually a formidable challenge. To start with, the difficulty of the problem may depend on the resolution of the image.

Já invocamos as probabilidades nas seções anteriores, sem articular o que são precisamente ou dar um exemplo concreto. Vamos ser mais sérios agora, considerando o primeiro caso: distinguir cães e gatos com base em fotografias. Isso pode parecer simples, mas na verdade é um desafio formidável. Para começar, a dificuldade do problema pode depender da resolução da imagem.

![Images of varying resolutions ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, and $160 \times 160$ pixels).](../img/cat-dog-pixels.png)
:width:`300px`
:label:`fig_cat_dog`

As shown in :numref:`fig_cat_dog`,
while it is easy for humans to recognize cats and dogs at the resolution of $160 \times 160$ pixels,
it becomes challenging at $40 \times 40$ pixels and next to impossible at $10 \times 10$ pixels. In
other words, our ability to tell cats and dogs apart at a large distance (and thus low resolution) might approach uninformed guessing. Probability gives us a
formal way of reasoning about our level of certainty.
If we are completely sure
that the image depicts a cat, we say that the *probability* that the corresponding label $y$ is "cat", denoted $P(y=$ "cat"$)$ equals $1$.
If we had no evidence to suggest that $y =$ "cat" or that $y =$ "dog", then we might say that the two possibilities were equally
*likely* expressing this as $P(y=$ "cat"$) = P(y=$ "dog"$) = 0.5$. If we were reasonably
confident, but not sure that the image depicted a cat, we might assign a
probability $0.5  < P(y=$ "cat"$) < 1$.

Conforme mostrado em: numref: `fig_cat_dog`,
embora seja fácil para os humanos reconhecerem cães e gatos na resolução de $160 \times 160$ pixels,
torna-se um desafio em $40 \times 40$ pixels e quase impossível em $10 \times 10$ pixels. No
Em outras palavras, nossa capacidade de distinguir cães e gatos a uma grande distância (e, portanto, em baixa resolução) pode se aproximar de uma suposição desinformada. A probabilidade nos dá um
maneira formal de raciocinar sobre nosso nível de certeza.
Se tivermos certeza absoluta
que a imagem representa um gato, dizemos que a * probabilidade * de que o rótulo $y$ correspondente seja "cat", denotado $P(y=$ "cat"$)$ é igual a $1$.
Se não tivéssemos nenhuma evidência para sugerir que $y =$ "cat" ou que $y =$ "dog", então poderíamos dizer que as duas possibilidades eram igualmente
*provavelmente* expressando isso como  $P(y=$ "cat"$) = P(y=$ "dog"$) = 0.5$. Se estivéssemos razoavelmente
confiantes, mas não temos certeza de que a imagem representava um gato, podemos atribuir um
probabilidade $0,5 <P (y = $"cat"$) <1$.

Now consider the second case: given some weather monitoring data, we want to predict the probability that it will rain in Taipei tomorrow. If it is summertime, the rain might come with probability 0.5.

Agora considere o segundo caso: dados alguns dados de monitoramento do tempo, queremos prever a probabilidade de que choverá em Taipei amanhã. Se for verão, a chuva pode vir com probabilidade 0,5.

In both cases, we have some value of interest. And in both cases we are uncertain about the outcome.
But there is a key difference between the two cases. In this first case, the image is in fact either a dog or a cat, and we just do not know which. In the second case, the outcome may actually be a random event, if you believe in such things (and most physicists do). So probability is a flexible language for reasoning about our level of certainty, and it can be applied effectively in a broad set of contexts.

Em ambos os casos, temos algum valor de interesse. E em ambos os casos não temos certeza sobre o resultado.
Mas existe uma diferença fundamental entre os dois casos. Neste primeiro caso, a imagem é de fato um cachorro ou um gato, e simplesmente não sabemos qual. No segundo caso, o resultado pode realmente ser um evento aleatório, se você acredita em tais coisas (e a maioria dos físicos acredita). Portanto, probabilidade é uma linguagem flexível para raciocinar sobre nosso nível de certeza e pode ser aplicada com eficácia em um amplo conjunto de contextos.

## Basic Probability Theory

Say that we cast a die and want to know what the chance is of seeing a 1 rather than another digit. If the die is fair, all the six outcomes $\{1, \ldots, 6\}$ are equally likely to occur, and thus we would see a $1$ in one out of six cases. Formally we state that $1$ occurs with probability $\frac{1}{6}$.

Digamos que lançamos um dado e queremos saber qual é a chance de ver um 1 em vez de outro dígito. Se o dado for justo, todos os seis resultados $\{1, \ldots, 6\}$ têm a mesma probabilidade de ocorrer e, portanto, veríamos $1$ em um dos seis casos. Formalmente afirmamos que $1$ ocorre com probabilidade $\frac{1}{6}$.

For a real die that we receive from a factory, we might not know those proportions and we would need to check whether it is tainted. The only way to investigate the die is by casting it many times and recording the outcomes. For each cast of the die, we will observe a value in $\{1, \ldots, 6\}$. Given these outcomes, we want to investigate the probability of observing each outcome.

Para um dado real que recebemos de uma fábrica, podemos não saber essas proporções e precisaríamos verificar se ele está contaminado. A única maneira de investigar o dado é lançando-o várias vezes e registrando os resultados. Para cada lançamento do dado, observaremos um valor em $\{1, \ldots, 6\}$. Dados esses resultados, queremos investigar a probabilidade de observar cada resultado.

One natural approach for each value is to take the
individual count for that value and to divide it by the total number of tosses.
This gives us an *estimate* of the probability of a given *event*. The *law of
large numbers* tell us that as the number of tosses grows this estimate will draw closer and closer to the true underlying probability. Before going into the details of what is going here, let us try it out.

Uma abordagem natural para cada valor é pegar o
contagem individual para aquele valor e dividi-lo pelo número total de jogadas.
Isso nos dá uma *estimativa* da probabilidade de um determinado *evento*. A *lei de
grandes números* nos dizem que, conforme o número de lançamentos aumenta, essa estimativa se aproxima cada vez mais da verdadeira probabilidade subjacente. Antes de entrar em detalhes sobre o que está acontecendo aqui, vamos experimentar.

To start, let us import the necessary packages.

Para começar, importemos os pacotes necessários.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch.distributions import multinomial
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
```

Next, we will want to be able to cast the die. In statistics we call this process
of drawing examples from probability distributions *sampling*.
The distribution
that assigns probabilities to a number of discrete choices is called the
*multinomial distribution*. We will give a more formal definition of
*distribution* later, but at a high level, think of it as just an assignment of
probabilities to events.

Em seguida, queremos ser capazes de lançar o dado. Nas estatísticas, chamamos este processo
de exemplos de desenho de distribuições de probabilidade *amostragem*.
A distribuição
que atribui probabilidades a uma série de escolhas discretas é chamado de
*distribuição multinomial*. Daremos uma definição mais formal de
*distribuição* mais tarde, mas em um alto nível, pense nisso como apenas uma atribuição de
probabilidades para eventos.

To draw a single sample, we simply pass in a vector of probabilities.
The output is another vector of the same length:
its value at index $i$ is the number of times the sampling outcome corresponds to $i$.

Para desenhar uma única amostra, simplesmente passamos um vetor de probabilidades.
A saída é outro vetor do mesmo comprimento:
seu valor no índice $i$ é o número de vezes que o resultado da amostragem corresponde a $i$.

```{.python .input}
fair_probs = [1.0 / 6] * 6
np.random.multinomial(1, fair_probs)
```

```{.python .input}
#@tab pytorch
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
fair_probs = tf.ones(6) / 6
tfp.distributions.Multinomial(1, fair_probs).sample()
```

If you run the sampler a bunch of times, you will find that you get out random
values each time. As with estimating the fairness of a die, we often want to
generate many samples from the same distribution. It would be unbearably slow to
do this with a Python `for` loop, so the function we are using supports drawing
multiple samples at once, returning an array of independent samples in any shape
we might desire.

Se você executar o amostrador várias vezes, descobrirá que sai aleatoriamente
valores de cada vez. Tal como acontece com a estimativa da justiça de um dado, muitas vezes queremos
gerar muitas amostras da mesma distribuição. Seria insuportavelmente lento para
fazer isso com um loop Python `for`, então a função que estamos usando suporta desenho
várias amostras de uma vez, retornando uma matriz de amostras independentes em qualquer forma
podemos desejar.

```{.python .input}
np.random.multinomial(10, fair_probs)
```

```{.python .input}
#@tab pytorch
multinomial.Multinomial(10, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
tfp.distributions.Multinomial(10, fair_probs).sample()
```

Now that we know how to sample rolls of a die, we can simulate 1000 rolls. We
can then go through and count, after each of the 1000 rolls, how many times each
number was rolled.
Specifically, we calculate the relative frequency as the estimate of the true probability.

Agora que sabemos como obter amostras de rolos de um dado, podemos simular 1000 rolos. Nós
pode então passar e contar, após cada um dos 1000 lançamentos, quantas vezes cada
número foi rolado.
Especificamente, calculamos a frequência relativa como a estimativa da probabilidade verdadeira.

```{.python .input}
counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
counts / 1000
```

```{.python .input}
#@tab pytorch
# Store the results as 32-bit floats for division
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # Relative frequency as the estimate
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(1000, fair_probs).sample()
counts / 1000
```

Because we generated the data from a fair die, we know that each outcome has true probability $\frac{1}{6}$, roughly $0.167$, so the above output estimates look good.

Como geramos os dados de um dado justo, sabemos que cada resultado tem probabilidade real $\frac{1}{6}$, cerca de $0,167$, portanto, as estimativas de saída acima parecem boas.

We can also visualize how these probabilities converge over time towards the true probability.
Let us conduct 500 groups of experiments where each group draws 10 samples.

Também podemos visualizar como essas probabilidades convergem ao longo do tempo para a probabilidade verdadeira.
Vamos conduzir 500 grupos de experimentos onde cada grupo extrai 10 amostras.

```{.python .input}
counts = np.random.multinomial(10, fair_probs, size=500)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].asnumpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(10, fair_probs).sample(500)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

Each solid curve corresponds to one of the six values of the die and gives our estimated probability that the die turns up that value as assessed after each group of experiments.
The dashed black line gives the true underlying probability.
As we get more data by conducting more experiments,
the $6$ solid curves converge towards the true probability.

Cada curva sólida corresponde a um dos seis valores do dado e dá nossa probabilidade estimada de que o dado aumente esse valor conforme avaliado após cada grupo de experimentos.
A linha preta tracejada fornece a verdadeira probabilidade subjacente.
À medida que obtemos mais dados conduzindo mais experimentos,
as curvas sólidas de $6$ convergem para a probabilidade verdadeira.

### Axiomas da Teoria de Probabilidade

When dealing with the rolls of a die,
we call the set $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ the *sample space* or *outcome space*, where each element is an *outcome*.
An *event* is a set of outcomes from a given sample space.
For instance, "seeing a $5$" ($\{5\}$) and "seeing an odd number" ($\{1, 3, 5\}$) are both valid events of rolling a die.
Note that if the outcome of a random experiment is in event $\mathcal{A}$,
then event $\mathcal{A}$ has occurred.
That is to say, if $3$ dots faced up after rolling a die, since $3 \in \{1, 3, 5\}$,
we can say that the event "seeing an odd number" has occurred.

Ao lidar com as jogadas de um dado,
chamamos o conjunto $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ o *espaço de amostra* ou *espaço de resultado*, onde cada elemento é um *resultado*.
Um *evento* é um conjunto de resultados de um determinado espaço amostral.
Por exemplo, "ver $5$" ($\{5 \}$) e "ver um número ímpar" ($\{1, 3, 5 \}$) são eventos válidos de lançar um dado.
Observe que se o resultado de um experimento aleatório estiver no evento $\mathcal {A}$,
então o evento $\mathcal {A}$ ocorreu.
Ou seja, se $3$ pontos virados para cima após rolar um dado, uma vez que $3 \in \{1, 3, 5 \}$,
podemos dizer que o evento "ver um número ímpar" ocorreu.

Formally, *probability* can be thought of a function that maps a set to a real value.
The probability of an event $\mathcal{A}$ in the given sample space $\mathcal{S}$,
denoted as $P(\mathcal{A})$, satisfies the following properties:

Formalmente, *probabilidade* pode ser pensada como uma função que mapeia um conjunto para um valor real.
A probabilidade de um evento $\mathcal {A}$ no espaço amostral dado $\mathcal {S}$,
denotado como $P (\mathcal {A})$, satisfaz as seguintes propriedades:

* For any event $\mathcal{A}$, its probability is never negative, i.e., $P(\mathcal{A}) \geq 0$;
* Probability of the entire sample space is $1$, i.e., $P(\mathcal{S}) = 1$;
* For any countable sequence of events $\mathcal{A}_1, \mathcal{A}_2, \ldots$ that are *mutually exclusive* ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ for all $i \neq j$), the probability that any happens is equal to the sum of their individual probabilities, i.e., $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

* Para qualquer evento $\mathcal {A}$, sua probabilidade nunca é negativa, ou seja, $P (\mathcal {A}) \geq 0$;
* A probabilidade de todo o espaço amostral é $ 1 $, ou seja, $P (\mathcal {S}) = 1$;
* Para qualquer sequência contável de eventos $\mathcal {A} _1, \mathcal {A} _2, \ldots$ que são *mutuamente exclusivos* ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ para todo $i \neq j$), a probabilidade de que aconteça é igual à soma de suas probabilidades individuais, ou seja, $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

These are also the axioms of probability theory, proposed by Kolmogorov in 1933.
Thanks to this axiom system, we can avoid any philosophical dispute on randomness;
instead, we can reason rigorously with a mathematical language.
For instance, by letting event $\mathcal{A}_1$ be the entire sample space and $\mathcal{A}_i = \emptyset$ for all $i > 1$, we can prove that $P(\emptyset) = 0$, i.e., the probability of an impossible event is $0$.

Esses também são os axiomas da teoria das probabilidades, propostos por Kolmogorov em 1933.
Graças a este sistema de axiomas, podemos evitar qualquer disputa filosófica sobre aleatoriedade;
em vez disso, podemos raciocinar rigorosamente com uma linguagem matemática.
Por exemplo, permitindo que o evento $\mathcal{A}_1$ seja todo o espaço da amostra e $\mathcal{A}_i = \emptyset$ para todos $i> 1$, podemos provar que $P(\emptyset) = 0$, ou seja, a probabilidade de um evento impossível é $0$.

### Random Variables

In our random experiment of casting a die, we introduced the notion of a *random variable*. A random variable can be pretty much any quantity and is not deterministic. It could take one value among a set of possibilities in a random experiment.
Consider a random variable $X$ whose value is in the sample space $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ of rolling a die. We can denote the event "seeing a $5$" as $\{X = 5\}$ or $X = 5$, and its probability as $P(\{X = 5\})$ or $P(X = 5)$.
By $P(X = a)$, we make a distinction between the random variable $X$ and the values (e.g., $a$) that $X$ can take.
However, such pedantry results in a cumbersome notation.
For a compact notation,
on one hand, we can just denote $P(X)$ as the *distribution* over the random variable $X$:
the distribution tells us the probability that $X$ takes any value.
On the other hand,
we can simply write $P(a)$ to denote the probability that a random variable takes the value $a$.
Since an event in probability theory is a set of outcomes from the sample space,
we can specify a range of values for a random variable to take.
For example, $P(1 \leq X \leq 3)$ denotes the probability of the event $\{1 \leq X \leq 3\}$,
which means $\{X = 1, 2, \text{or}, 3\}$. Equivalently, $P(1 \leq X \leq 3)$ represents the probability that the random variable $X$ can take a value from $\{1, 2, 3\}$.

Em nosso experimento aleatório de lançar um dado, introduzimos a noção de uma * variável aleatória *. Uma variável aleatória pode ser praticamente qualquer quantidade e não é determinística. Pode assumir um valor entre um conjunto de possibilidades em um experimento aleatório.
Considere uma variável aleatória $X$ cujo valor está no espaço amostral $\mathcal {S} = \{1, 2, 3, 4, 5, 6 \}$ do lançamento de um dado. Podemos denotar o evento "vendo $5$" como $\{X = 5 \}$ ou $X = 5$, e sua probabilidade como $P (\{X = 5 \})$ ou $P (X = 5)$.
Por $P (X = a)$, fazemos uma distinção entre a variável aleatória $X$ e os valores (por exemplo, $a$) que $X$ pode assumir.
No entanto, esse pedantismo resulta em uma notação complicada.
Para uma notação compacta,
por um lado, podemos apenas denotar $P (X)$ como a *distribuição* sobre a variável aleatória $X$:
a distribuição nos diz a probabilidade de que $X$ assuma qualquer valor.
Por outro lado,
podemos simplesmente escrever $P (a)$ para denotar a probabilidade de uma variável aleatória assumir o valor $a$.
Uma vez que um evento na teoria da probabilidade é um conjunto de resultados do espaço amostral,
podemos especificar um intervalo de valores para uma variável aleatória assumir.
Por exemplo, $P(1 \leq X \leq 3)$ denota a probabilidade do evento $\{1 \leq X \leq 3\}$,
o que significa $\{X = 1, 2, \text{or}, 3\}$. De forma equivalente, $\{X = 1, 2, \text{or}, 3\}$ representa a probabilidade de que a variável aleatória $X$ possa assumir um valor de $\{1, 2, 3\}$.

Note that there is a subtle difference between *discrete* random variables, like the sides of a die, and *continuous* ones, like the weight and the height of a person. There is little point in asking whether two people have exactly the same height. If we take precise enough measurements you will find that no two people on the planet have the exact same height. In fact, if we take a fine enough measurement, you will not have the same height when you wake up and when you go to sleep. So there is no purpose in asking about the probability
that someone is 1.80139278291028719210196740527486202 meters tall. Given the world population of humans the probability is virtually 0. It makes more sense in this case to ask whether someone's height falls into a given interval, say between 1.79 and 1.81 meters. In these cases we quantify the likelihood that we see a value as a *density*. The height of exactly 1.80 meters has no probability, but nonzero density. In the interval between any two different heights we have nonzero probability.
In the rest of this section, we consider probability in discrete space.
For probability over continuous random variables, you may refer to :numref:`sec_random_variables`.

Observe que há uma diferença sutil entre variáveis ​​aleatórias * discretas *, como os lados de um dado, e * contínuas *, como o peso e a altura de uma pessoa. Não adianta perguntar se duas pessoas têm exatamente a mesma altura. Se tomarmos medidas precisas o suficiente, você descobrirá que duas pessoas no planeta não têm exatamente a mesma altura. Na verdade, se fizermos uma medição suficientemente precisa, você não terá a mesma altura ao acordar e ao dormir. Portanto, não há nenhum propósito em perguntar sobre a probabilidade
que alguém tem 1,80139278291028719210196740527486202 metros de altura. Dada a população mundial de humanos, a probabilidade é virtualmente 0. Faz mais sentido, neste caso, perguntar se a altura de alguém cai em um determinado intervalo, digamos entre 1,79 e 1,81 metros. Nesses casos, quantificamos a probabilidade de vermos um valor como uma * densidade *. A altura de exatamente 1,80 metros não tem probabilidade, mas densidade diferente de zero. No intervalo entre quaisquer duas alturas diferentes, temos probabilidade diferente de zero.
No restante desta seção, consideramos a probabilidade no espaço discreto.
Para probabilidade sobre variáveis ​​aleatórias contínuas, você pode consultar: numref: `sec_random_variables`.

## Dealing with Multiple Random Variables

Very often, we will want to consider more than one random variable at a time.
For instance, we may want to model the relationship between diseases and symptoms. Given a disease and a symptom, say "flu" and "cough", either may or may not occur in a patient with some probability. While we hope that the probability of both would be close to zero, we may want to estimate these probabilities and their relationships to each other so that we may apply our inferences to effect better medical care.

Muitas vezes, queremos considerar mais de uma variável aleatória de cada vez.
Por exemplo, podemos querer modelar a relação entre doenças e sintomas. Dados uma doença e um sintoma, digamos "gripe" e "tosse", podem ou não ocorrer em um paciente com alguma probabilidade. Embora esperemos que a probabilidade de ambos seja próxima de zero, podemos estimar essas probabilidades e suas relações entre si para que possamos aplicar nossas inferências para obter um melhor atendimento médico.

As a more complicated example, images contain millions of pixels, thus millions of random variables. And in many cases images will come with a
label, identifying objects in the image. We can also think of the label as a
random variable. We can even think of all the metadata as random variables
such as location, time, aperture, focal length, ISO, focus distance, and camera type.
All of these are random variables that occur jointly. When we deal with multiple random variables, there are several quantities of interest.

Como um exemplo mais complicado, as imagens contêm milhões de pixels, portanto, milhões de variáveis aleatórias. E, em muitos casos, as imagens vêm com um
rótulo, identificando objetos na imagem. Também podemos pensar no rótulo como um
variável aleatória. Podemos até pensar em todos os metadados como variáveis aleatórias
como local, tempo, abertura, comprimento focal, ISO, distância de foco e tipo de câmera.
Todas essas são variáveis aleatórias que ocorrem em conjunto. Quando lidamos com múltiplas variáveis aleatórias, existem várias quantidades de interesse.

### Joint Probability

The first is called the *joint probability* $P(A = a, B=b)$. Given any values $a$ and $b$, the joint probability lets us answer, what is the probability that $A=a$ and $B=b$ simultaneously?
Note that for any values $a$ and $b$, $P(A=a, B=b) \leq P(A=a)$.
This has to be the case, since for $A=a$ and $B=b$ to happen, $A=a$ has to happen *and* $B=b$ also has to happen (and vice versa). Thus, $A=a$ and $B=b$ cannot be more likely than $A=a$ or $B=b$ individually.


O primeiro é chamado de * probabilidade conjunta * $P(A = a, B=b)$. Dados quaisquer valores $a$ e $b$, a probabilidade conjunta nos permite responder, qual é a probabilidade de que $A = a$ e $B = b$ simultaneamente?
Observe que, para quaisquer valores $a$ e $b$, $P (A = a, B = b) \leq P (A = a)$.
Tem de ser este o caso, visto que para $A = a$ e $B = b$ acontecer, $A = a$ tem que acontecer *e* $B = b$ também tem que acontecer (e vice-versa). Assim, $A = a$ e $B = b$ não podem ser mais prováveis do que $A = a$ ou $B = b$ individualmente.

### Conditional Probability

This brings us to an interesting ratio: $0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$. We call this ratio a *conditional probability*
and denote it by $P(B=b \mid A=a)$: it is the probability of $B=b$, provided that
$A=a$ has occurred.

Isso nos leva a uma razão interessante: $0 \leq \frac {P (A = a, B = b)} {P (A = a)} \leq 1$. Chamamos essa proporção de *probabilidade condicional*
e denotá-lo por $P (B = b \mid A = a)$: é a probabilidade de $B = b$, desde que
$A = a$ ocorreu.

### Bayes' theorem

Using the definition of conditional probabilities, we can derive one of the most useful and celebrated equations in statistics: *Bayes' theorem*.
It goes as follows.
By construction, we have the *multiplication rule* that $P(A, B) = P(B \mid A) P(A)$. By symmetry, this also holds for $P(A, B) = P(A \mid B) P(B)$. Assume that $P(B) > 0$. Solving for one of the conditional variables we get

Usando a definição de probabilidades condicionais, podemos derivar uma das equações mais úteis e celebradas em estatística: * Teorema de Bayes *.
É o seguinte.
Por construção, temos a * regra de multiplicação * que $P(A, B) = P(B \mid A) P(A)$. Por simetria, isso também é válido para $P(A, B) = P(A \mid B) P(B)$. Suponha que $P(B)> 0$. Resolvendo para uma das variáveis condicionais, obtemos

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

Note that here we use the more compact notation where $P(A, B)$ is a *joint distribution* and $P(A \mid B)$ is a *conditional distribution*. Such distributions can be evaluated for particular values $A = a, B=b$.

Observe que aqui usamos a notação mais compacta em que $P(A, B)$ é uma *distribuição conjunta* e $P(A \mid B)$ é uma *distribuição condicional*. Essas distribuições podem ser avaliadas para valores particulares $A = a, B = b$.

### Marginalization

Bayes' theorem is very useful if we want to infer one thing from the other, say cause and effect, but we only know the properties in the reverse direction, as we will see later in this section. One important operation that we need, to make this work, is *marginalization*.
It is the operation of determining $P(B)$ from $P(A, B)$. We can see that the probability of $B$ amounts to accounting for all possible choices of $A$ and aggregating the joint probabilities over all of them:

O teorema de Bayes é muito útil se quisermos inferir uma coisa da outra, digamos causa e efeito, mas só conhecemos as propriedades na direção reversa, como veremos mais adiante nesta seção. Uma operação importante de que precisamos para fazer esse trabalho é a *marginalização*.
É a operação de determinar $P(B)$ de $P(A, B)$. Podemos ver que a probabilidade de $B$ equivale a contabilizar todas as escolhas possíveis de $A$ e agregar as probabilidades conjuntas de todas elas:

$$P(B) = \sum_{A} P(A, B),$$

which is also known as the *sum rule*. The probability or distribution as a result of marginalization is called a *marginal probability* or a *marginal distribution*.

que também é conhecida como * regra da soma *. A probabilidade ou distribuição como resultado da marginalização é chamada de *probabilidade marginal* ou *distribuição marginal*.

### Independence

Another useful property to check for is *dependence* vs. *independence*.
Two random variables $A$ and $B$ being independent
means that the occurrence of one event of $A$
does not reveal any information about the occurrence of an event of $B$.
In this case $P(B \mid A) = P(B)$. Statisticians typically express this as $A \perp  B$. From Bayes' theorem, it follows immediately that also $P(A \mid B) = P(A)$.
In all the other cases we call $A$ and $B$ dependent. For instance, two successive rolls of a die are independent. In contrast, the position of a light switch and the brightness in the room are not (they are not perfectly deterministic, though, since we could always have a broken light bulb, power failure, or a broken switch).

Outra propriedade útil para verificar é * dependência * vs. * independência *.
Duas variáveis aleatórias $A$ e $B$ sendo independentes
significa que a ocorrência de um evento de $A$
não revela nenhuma informação sobre a ocorrência de um evento de $B$.
Neste caso $P(B \mid A) = P(B)$. Os estatísticos normalmente expressam isso como $A \perp  B$. Do teorema de Bayes, segue imediatamente que também $P(A \mid B) = P(A)$.
Em todos os outros casos, chamamos $A$ e $B$ de dependente. Por exemplo, duas jogadas sucessivas de um dado são independentes. Em contraste, a posição de um interruptor de luz e a luminosidade da sala não são (eles não são perfeitamente determinísticos, pois podemos sempre ter uma lâmpada quebrada, falha de energia ou um interruptor quebrado).

Since $P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$ is equivalent to $P(A, B) = P(A)P(B)$, two random variables are independent if and only if their joint distribution is the product of their individual distributions.
Likewise, two random variables $A$ and $B$ are *conditionally independent* given another random variable $C$
if and only if $P(A, B \mid C) = P(A \mid C)P(B \mid C)$. This is expressed as $A \perp B \mid C$.

Dado que $P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$ é equivalente a $P(A, B) = P(A)P(B)$, duas variáveis aleatórias são independentes se e somente se sua distribuição conjunta é o produto de suas distribuições individuais.
Da mesma forma, duas variáveis aleatórias $A$ e $B$ são *condicionalmente independentes* dada outra variável aleatória $C$
se e somente se $P(A, B \mid C) = P(A \mid C)P(B \mid C)$. Isso é expresso como $A \perp B \mid C$.

### Aplicação
:label:`subsec_probability_hiv_app`

Let us put our skills to the test. Assume that a doctor administers an HIV test to a patient. This test is fairly accurate and it fails only with 1% probability if the patient is healthy but reporting him as diseased. Moreover,
it never fails to detect HIV if the patient actually has it. We use $D_1$ to indicate the diagnosis ($1$ if positive and $0$ if negative) and $H$ to denote the HIV status ($1$ if positive and $0$ if negative).
:numref:`conditional_prob_D1` lists such conditional probabilities.

Vamos colocar nossas habilidades à prova. Suponha que um médico administre um teste de HIV a um paciente. Este teste é bastante preciso e falha apenas com 1% de probabilidade se o paciente for saudável, mas relatá-lo como doente. Além disso,
nunca deixa de detectar o HIV se o paciente realmente o tiver. Usamos $D_1$ para indicar o diagnóstico ($1$ se positivo e $0$ se negativo) e $H$ para denotar o estado de HIV ($1$ se positivo e $0$ se negativo).
: numref: `conditional_prob_D1` lista tais probabilidades condicionais.
:
Conditional probability of $P(D_1 \mid H)$.

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`

Note that the column sums are all 1 (but the row sums are not), since the conditional probability needs to sum up to 1, just like the probability. Let us work out the probability of the patient having HIV if the test comes back positive, i.e., $P(H = 1 \mid D_1 = 1)$. Obviously this is going to depend on how common the disease is, since it affects the number of false alarms. Assume that the population is quite healthy, e.g., $P(H=1) = 0.0015$. To apply Bayes' theorem, we need to apply marginalization and the multiplication rule to determine

Observe que as somas das colunas são todas 1 (mas as somas das linhas não), uma vez que a probabilidade condicional precisa somar 1, assim como a probabilidade. Vamos calcular a probabilidade de o paciente ter HIV se o teste der positivo, ou seja, $P(H = 1 \mid D_1 = 1)$. Obviamente, isso vai depender de quão comum é a doença, já que afeta o número de alarmes falsos. Suponha que a população seja bastante saudável, por exemplo, $P(H=1) = 0.0015$. Para aplicar o teorema de Bayes, precisamos aplicar a marginalização e a regra de multiplicação para determinar

$$\begin{aligned}
&P(D_1 = 1) \\
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

Thus, we get

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1)\\ =& \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} \\ =& 0.1306 \end{aligned}.$$

In other words, there is only a 13.06% chance that the patient
actually has HIV, despite using a very accurate test.
As we can see, probability can be counterintuitive.

Em outras palavras, há apenas 13,06% de chance de que o paciente
realmente tem HIV, apesar de usar um teste muito preciso.
Como podemos ver, a probabilidade pode ser contra-intuitiva.

What should a patient do upon receiving such terrifying news? Likely, the patient
would ask the physician to administer another test to get clarity. The second
test has different characteristics and it is not as good as the first one, as shown in :numref:`conditional_prob_D2`.

O que o paciente deve fazer ao receber notícias tão terríveis? Provavelmente, o paciente
pediria ao médico para administrar outro teste para obter clareza. O segundo
teste tem características diferentes e não é tão bom quanto o primeiro, como mostrado em :numref: conditional_prob_D2`.

:Conditional probability of $P(D_2 \mid H)$.

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_2 = 1 \mid H)$|            0.98 |         0.03 |
|$P(D_2 = 0 \mid H)$|            0.02 |         0.97 |
:label:`conditional_prob_D2`

Unfortunately, the second test comes back positive, too.
Let us work out the requisite probabilities to invoke Bayes' theorem
by assuming the conditional independence:

Infelizmente, o segundo teste também deu positivo.
Vamos trabalhar as probabilidades necessárias para invocar o teorema de Bayes
assumindo a independência condicional:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 0) \\
=& P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)  \\
=& 0.0003,
\end{aligned}
$$

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 1) \\
=& P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)  \\
=& 0.98.
\end{aligned}
$$

Now we can apply marginalization and the multiplication rule:

Agora podemos aplicar a marginalização e a regra de multiplicação:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1) \\
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$

In the end, the probability of the patient having HIV given both positive tests is

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1, D_2 = 1)\\
=& \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)} \\
=& 0.8307.
\end{aligned}
$$

That is, the second test allowed us to gain much higher confidence that not all is well. Despite the second test being considerably less accurate than the first one, it still significantly improved our estimate.

Ou seja, o segundo teste nos permitiu ganhar uma confiança muito maior de que nem tudo está bem. Apesar do segundo teste ser consideravelmente menos preciso do que o primeiro, ele ainda melhorou significativamente nossa estimativa.



## Expectation and Variance

To summarize key characteristics of probability distributions,
we need some measures.
The *expectation* (or average) of the random variable $X$ is denoted as

Para resumir as principais características das distribuições de probabilidade,
precisamos de algumas medidas.
A *expectativa* (ou média) da variável aleatória $X$ é denotada como

$$E[X] = \sum_{x} x P(X = x).$$

When the input of a function $f(x)$ is a random variable drawn from the distribution $P$ with different values $x$,
the expectation of $f(x)$ is computed as

Quando a entrada de uma função $f (x)$ é uma variável aleatória retirada da distribuição $P$ com valores diferentes $x$,
a expectativa de $f (x)$ é calculada como

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x).$$


In many cases we want to measure by how much the random variable $X$ deviates from its expectation. This can be quantified by the variance

Em muitos casos, queremos medir o quanto a variável aleatória $X$ se desvia de sua expectativa. Isso pode ser quantificado pela variação

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.$$

Its square root is called the *standard deviation*.
The variance of a function of a random variable measures
by how much the function deviates from the expectation of the function,
as different values $x$ of the random variable are sampled from its distribution:

Sua raiz quadrada é chamada de *desvio padrão*.
A variância de uma função de uma variável aleatória mede
pelo quanto a função se desvia da expectativa da função,
como diferentes valores $x$ da variável aleatória são amostrados de sua distribuição:

$$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].$$


## Summary

* We can sample from probability distributions.
* We can analyze multiple random variables using joint distribution, conditional distribution, Bayes' theorem, marginalization, and independence assumptions.
* Expectation and variance offer useful measures to summarize key characteristics of probability distributions.

* Podemos obter amostras de distribuições de probabilidade.
* Podemos analisar múltiplas variáveis aleatórias usando distribuição conjunta, distribuição condicional, teorema de Bayes, marginalização e suposições de independência.
* A expectativa e a variância oferecem medidas úteis para resumir as principais características das distribuições de probabilidade.

## Exercises

1. We conducted $m=500$ groups of experiments where each group draws $n=10$ samples. Vary $m$ and $n$. Observe and analyze the experimental results.
2. Given two events with probability $P(\mathcal{A})$ and $P(\mathcal{B})$, compute upper and lower bounds on $P(\mathcal{A} \cup \mathcal{B})$ and $P(\mathcal{A} \cap \mathcal{B})$. (Hint: display the situation using a [Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram).)
3. Assume that we have a sequence of random variables, say $A$, $B$, and $C$, where $B$ only depends on $A$, and $C$ only depends on $B$, can you simplify the joint probability $P(A, B, C)$? (Hint: this is a [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain).)
4. In :numref:`subsec_probability_hiv_app`, the first test is more accurate. Why not run the first test twice rather than run both the first and second tests?
5. 
6. Conduzimos $m = 500$ grupos de experimentos onde cada grupo extrai $n = 10$ amostras. Varie $m$ e $n$. Observe e analise os resultados experimentais.
7. Dados dois eventos com probabilidade $P(\mathcal{A})$ e $P(\mathcal{B})$, calcule os limites superior e inferior em $P(\mathcal{A} \cup \mathcal{B})$ e $P(\mathcal{A} \cap \mathcal{B})$. (Dica: exiba a situação usando um [Diagrama de Venn](https://en.wikipedia.org/wiki/Venn_diagram).)
8. Suponha que temos uma sequência de variáveis aleatórias, digamos $A$, $B$ e $C$, onde $B$ depende apenas de $A$ e $C$ depende apenas de $B$, você pode simplificar a probabilidade conjunta $P (A, B, C)$? (Dica: esta é uma [Cadeia de Markov](https://en.wikipedia.org/wiki/Markov_chain).)
9. Em :numref:`subsec_probability_hiv_app`, o primeiro teste é mais preciso. Por que não executar o primeiro teste duas vezes em vez de executar o primeiro e o segundo testes?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/198)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMzcyODg3MTU4LDgxOTkyNDAxOF19
-->
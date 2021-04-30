# Incorporação de Palavras (word2vec)
:label:`sec_word2vec`

Uma linguagem natural é um sistema complexo que usamos para expressar significados. Nesse sistema, as palavras são a unidade básica do significado linguístico. Como o próprio nome indica, um vetor de palavras é um vetor usado para representar uma palavra. Também pode ser considerado o vetor de características de uma palavra. A técnica de mapear palavras em vetores de números reais também é conhecida como incorporação de palavras. Nos últimos anos, a incorporação de palavras tornou-se gradualmente um conhecimento básico no processamento de linguagem natural.

## Por que não usar vetores one-hot?

Usamos vetores one-hot para representar palavras (caracteres são palavras) em
:numref:`sec_rnn_scratch`.
Lembre-se de que quando assumimos o número de palavras diferentes em um
dicionário (o tamanho do dicionário) é $N$, cada palavra pode corresponder uma a uma
com inteiros consecutivos de 0 a $N-1$. Esses inteiros que correspondem a
as palavras são chamadas de índices das palavras. Assumimos que o índice de uma palavra
é $i$. A fim de obter a representação vetorial one-hot da palavra, criamos
um vetor de 0s com comprimento de $N$ e defina o elemento $i$ como 1. Desta forma,
cada palavra é representada como um vetor de comprimento $N$ que pode ser usado diretamente por
a rede neural.

Embora os vetores de uma palavra quente sejam fáceis de construir, eles geralmente não são uma boa escolha. Uma das principais razões é que os vetores de uma palavra quente não podem expressar com precisão a semelhança entre palavras diferentes, como a semelhança de cosseno que usamos comumente. Para os vetores $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, suas semelhanças de cosseno são os cossenos dos ângulos entre eles:

$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$

Uma vez que a similaridade de cosseno entre os vetores one-hot de quaisquer duas palavras diferentes é 0, é difícil usar o vetor one-hot para representar com precisão a similaridade entre várias palavras diferentes.

[Word2vec](https://code.google.com/archive/p/word2vec/) é uma ferramenta que viemos
para resolver o problema acima. Ele representa cada palavra com um
vetor de comprimento fixo e usa esses vetores para melhor indicar a similaridade e
relações de analogia entre palavras diferentes. A ferramenta Word2vec contém dois
modelos: skip-gram :cite:`Mikolov.Sutskever.Chen.ea.2013` e bolsa contínua de
words (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013`. Em seguida, vamos dar um
observe os dois modelos e seus métodos de treinamento.

## O Modelo Skip-Gram

O modelo skip-gram assume que uma palavra pode ser usada para gerar as palavras que a cercam em uma sequência de texto. Por exemplo, assumimos que a sequência de texto é "o", "homem", "ama", "seu" e "filho". Usamos "amores" como palavra-alvo central e definimos o tamanho da janela de contexto para 2. Conforme mostrado em :numref:`fig_skip_gram`, dada a palavra-alvo central "amores", o modelo de grama de salto está preocupado com a probabilidade condicional para gerando as palavras de contexto, "o", "homem", "seu" e "filho", que estão a uma distância de no máximo 2 palavras, que é

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$

Assumimos que, dada a palavra-alvo central, as palavras de contexto são geradas independentemente umas das outras. Neste caso, a fórmula acima pode ser reescrita como

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![O modelo skip-gram se preocupa com a probabilidade condicional de gerar palavras de contexto para uma determinada palavra-alvo central.](../img/skip-gram.svg)
:label:`fig_skip_gram`

No modelo skip-gram, cada palavra é representada como dois vetores de dimensão $d$, que são usados para calcular a probabilidade condicional. Assumimos que a palavra está indexada como $i$ no dicionário, seu vetor é representado como $\mathbf{v}_i\in\mathbb{R}^d$ quando é a palavra alvo central, e $\mathbf{u}_i\in\mathbb{R}^d$ quando é uma palavra de contexto. Deixe a palavra alvo central $w_c$ e a palavra de contexto $w_o$ serem indexadas como $c$ e $o$ respectivamente no dicionário. A probabilidade condicional de gerar a palavra de contexto para a palavra alvo central fornecida pode ser obtida executando uma operação softmax no produto interno do vetor:

$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},$$

onde o índice de vocabulário definido $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$. Suponha que uma sequência de texto de comprimento $T$ seja fornecida, onde a palavra no passo de tempo $t$ é denotada como $w^{(t)}$. Suponha que as palavras de contexto sejam geradas independentemente, dadas as palavras centrais. Quando o tamanho da janela de contexto é $m$, a função de verossimilhança do modelo skip-gram é a probabilidade conjunta de gerar todas as palavras de contexto dadas qualquer palavra central

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

Aqui, qualquer intervalo de tempo menor que 1 ou maior que $T$ pode ser ignorado.

### Treinamento do modelo Skip-Gram

Os parâmetros do modelo skip-gram são o vetor da palavra alvo central e o vetor da palavra de contexto para cada palavra individual. No processo de treinamento, aprenderemos os parâmetros do modelo maximizando a função de verossimilhança, também conhecida como estimativa de máxima verossimilhança. Isso é equivalente a minimizar a seguinte função de perda:

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).$$


Se usarmos o SGD, em cada iteração vamos escolher uma subsequência mais curta por meio de amostragem aleatória para calcular a perda para essa subsequência e, em seguida, calcular o gradiente para atualizar os parâmetros do modelo. A chave do cálculo de gradiente é calcular o gradiente da probabilidade condicional logarítmica para o vetor de palavras central e o vetor de palavras de contexto. Por definição, primeiro temos


$$\log P(w_o \mid w_c) =
\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$

Por meio da diferenciação, podemos obter o gradiente $\mathbf{v}_c$ da fórmula acima.

$$
\begin{aligned}
\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}
&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\
&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\
&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.
\end{aligned}
$$

Seu cálculo obtém a probabilidade condicional para todas as palavras no dicionário dada a palavra alvo central $w_c$. Em seguida, usamos o mesmo método para obter os gradientes para outros vetores de palavras.

Após o treinamento, para qualquer palavra do dicionário com índice $i$, vamos obter seus conjuntos de vetores de duas palavras $\mathbf{v}_i$ e $\mathbf{u}_i$. Em aplicações de processamento de linguagem natural, o vetor de palavra-alvo central no modelo skip-gram é geralmente usado como o vetor de representação de uma palavra.


## O modelo do conjcontínuo de palavras (CBOW)

The continuous bag of words (CBOW) model is similar to the skip-gram model. The biggest difference is that the CBOW model assumes that the central target word is generated based on the context words before and after it in the text sequence. With the same text sequence "the", "man", "loves", "his" and "son", in which "loves" is the central target word, given a context window size of 2, the CBOW model is concerned with the conditional probability of generating the target word "loves" based on the context words "the", "man", "his" and "son"(as shown in :numref:`fig_cbow`), such as

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$

![The CBOW model cares about the conditional probability of generating the central target word from given context words.  ](../img/cbow.svg)
:label:`fig_cbow`

Since there are multiple context words in the CBOW model, we will average their word vectors and then use the same method as the skip-gram model to compute the conditional probability. We assume that $\mathbf{v_i}\in\mathbb{R}^d$ and $\mathbf{u_i}\in\mathbb{R}^d$ are the context word vector and central target word vector of the word with index $i$ in the dictionary (notice that the symbols are opposite to the ones in the skip-gram model). Let central target word $w_c$ be indexed as $c$, and context words $w_{o_1}, \ldots, w_{o_{2m}}$ be indexed as $o_1, \ldots, o_{2m}$ in the dictionary. Thus, the conditional probability of generating a central target word from the given context word is

$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$


For brevity, denote $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$, and $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$. The equation above can be simplified as

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$

Given a text sequence of length $T$, we assume that the word at time step $t$ is $w^{(t)}$, and the context window size is $m$.  The likelihood function of the CBOW model is the probability of generating any central target word from the context words.

$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

### CBOW Model Training

CBOW model training is quite similar to skip-gram model training.  The maximum likelihood estimation of the CBOW model is equivalent to minimizing the loss function.

$$  -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

Notice that

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$

Through differentiation, we can compute the logarithm of the conditional probability of the gradient of any context word vector $\mathbf{v}_{o_i}$($i = 1, \ldots, 2m$) in the formula above.

$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$

We then use the same method to obtain the gradients for other word vectors. Unlike the skip-gram model, we usually use the context word vector as the representation vector for a word in the CBOW model.

## Summary

* A word vector is a vector used to represent a word. The technique of mapping words to vectors of real numbers is also known as word embedding.
* Word2vec includes both the continuous bag of words (CBOW) and skip-gram models. The skip-gram model assumes that context words are generated based on the central target word. The CBOW model assumes that the central target word is generated based on the context words.


## Exercises

1. What is the computational complexity of each gradient? If the dictionary contains a large volume of words, what problems will this cause?
1. There are some fixed phrases in the English language which consist of multiple words, such as "new york". How can you train their word vectors? Hint: See section 4 in the Word2vec paper :cite:`Mikolov.Sutskever.Chen.ea.2013`.
1. Use the skip-gram model as an example to think about the design of a word2vec model. What is the relationship between the inner product of two word vectors and the cosine similarity in the skip-gram model? For a pair of words with close semantical meaning, why it is likely for their word vector cosine similarity to be high?



[Discussions](https://discuss.d2l.ai/t/381)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3MjQzNzg1MTddfQ==
-->
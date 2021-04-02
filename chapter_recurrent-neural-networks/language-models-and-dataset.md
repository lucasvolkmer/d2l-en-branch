# Modelos de Linguagem e o *Dataset*
:label:`sec_language_model`


Em :numref:`sec_text_preprocessing`,  vemos como mapear dados de texto em tokens, onde esses tokens podem ser vistos como uma sequência de observações discretas, como palavras ou caracteres.
Suponha que os tokens em uma sequência de texto de comprimento $T$ sejam, por sua vez, $x_1, x_2, \ldots, x_T$. 
Então, na sequência de texto,
$x_t$($1 \leq t \leq T$) pode ser considerado como a observação ou rótulo no intervalo de tempo $t$. Dada essa sequência de texto,
o objetivo de um *modelo de linguagem* é estimar a probabilidade conjunta da sequência

$$P(x_1, x_2, \ldots, x_T).$$


Os modelos de linguagem são incrivelmente úteis. Por exemplo, um modelo de linguagem ideal seria capaz de gerar texto natural sozinho, simplesmente desenhando um token por vez $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$.
Bem ao contrário do macaco usando uma máquina de escrever, todo texto emergente de tal modelo passaria como linguagem natural, por exemplo, texto em inglês. Além disso, seria suficiente para gerar um diálogo significativo, simplesmente condicionando o texto a fragmentos de diálogo anteriores.
Claramente, ainda estamos muito longe de projetar tal sistema, uma vez que seria necessário *compreender* o texto em vez de apenas gerar conteúdo gramaticalmente sensato.

No entanto, os modelos de linguagem são de grande utilidade, mesmo em sua forma limitada.
Por exemplo, as frases "reconhecer a fala" e "destruir uma bela praia" [^1] soam muito semelhantes.
Isso pode causar ambiguidade no reconhecimento de fala,
o que é facilmente resolvido por meio de um modelo de linguagem que rejeita a segunda tradução como bizarra.
Da mesma forma, em um algoritmo de sumarização de documentos
vale a pena saber que "cachorro morde homem" é muito mais frequente do que "homem morde cachorro", ou que "quero comer vovó" é uma afirmação um tanto perturbadora, enquanto "quero comer, vovó" é muito mais benigna.

[^1]: Traduzido de  *to recognize speech* e *to wreck a nice beach*


## Aprendendo um Modelo de Linguagem

A questão óbvia é como devemos modelar um documento, ou mesmo uma sequência de tokens.
Suponha que tokenizemos dados de texto no nível da palavra.
Podemos recorrer à análise que aplicamos aos modelos de sequência em :numref:`sec_sequence`.
Vamos começar aplicando regras básicas de probabilidade:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

Por exemplo,
a probabilidade de uma sequência de texto contendo quatro palavras seria dada como:

$$P(\text{deep}, \text{learning}, \text{é}, \text{divertido}) =  P(\text{deep}) P(\text{learning}  \mid  \text{deep}) P(\text{é}  \mid  \text{deep}, \text{learning}) P(\text{divertido}  \mid  \text{deep}, \text{learning}, \text{é}).$$


Para calcular o modelo de linguagem, precisamos calcular a
probabilidade de palavras e a probabilidade condicional de uma palavra dada
as poucas palavras anteriores.
Essas probabilidades são essencialmente
parâmetros do modelo de linguagem.

Aqui nós
supomos que o conjunto de dados de treinamento é um grande corpus de texto, como todas as
entradas da Wikipedia, [Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg),
e todo o texto postado no
wede.
A probabilidade das palavras pode ser calculada a partir da palavra relativa
frequência de uma determinada palavra no conjunto de dados de treinamento.
Por exemplo, a estimativa $\hat{P}(\text{deep})$ pode ser calculada como o
probabilidade de qualquer frase que comece com a palavra "deep". Uma
abordagem ligeiramente menos precisa seria contar todas as ocorrências de
a palavra "deep" e dividi-la pelo número total de palavras em
o corpus.
Isso funciona muito bem, especialmente para
palavras. Continuando, podemos tentar estimar

$$\hat{P}(\text{learning} \mid \text{deep}) = \frac{n(\text{deep, learning})}{n(\text{deep})},$$


onde $n(x)$ e $n(x, x')$ são o número de ocorrências de *singletons*
e pares de palavras consecutivas, respectivamente.
Infelizmente, estimando a
probabilidade de um par de palavras é um pouco mais difícil, uma vez que
as ocorrências de "deep learning" são muito menos frequentes. No
em particular, para algumas combinações incomuns de palavras, pode ser complicado
encontrar ocorrências suficientes para obter estimativas precisas.
As coisas pioram com as combinações de três palavras e além.
Haverá muitas combinações plausíveis de três palavras que provavelmente não veremos em nosso conjunto de dados.
A menos que forneçamos alguma solução para atribuir tais combinações de palavras contagens diferentes de zero, não poderemos usá-las em um modelo de linguagem. Se o conjunto de dados for pequeno ou se as palavras forem muito raras, podemos não encontrar nem mesmo uma delas.

Uma estratégia comum é realizar alguma forma de *suavização de Laplace*.
A solução é
adicionar uma pequena constante a todas as contagens.
Denote por $n$ o número total de palavras em
o conjunto de treinamento
e $m$ o número de palavras únicas.
Esta solução ajuda com singletons, por exemplo, via

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

Here $\epsilon_1,\epsilon_2$, and $\epsilon_3$ are hyperparameters.
Take $\epsilon_1$ as an example:
when $\epsilon_1 = 0$, no smoothing is applied;
when $\epsilon_1$ approaches positive infinity,
$\hat{P}(x)$ approaches the uniform probability $1/m$. 
The above is a rather primitive variant of what
other techniques can accomplish :cite:`Wood.Gasthaus.Archambeau.ea.2011`.


Unfortunately, models like this get unwieldy rather quickly
for the following reasons. First, we need to store all counts.
Second, this entirely ignores the meaning of the words. For
instance, "cat" and "feline" should occur in related contexts.
It is quite difficult to adjust such models to additional contexts,
whereas, deep learning based language models are well suited to
take this into account.
Last, long word
sequences are almost certain to be novel, hence a model that simply
counts the frequency of previously seen word sequences is bound to perform poorly there.

## Markov Models and $n$-grams

Before we discuss solutions involving deep learning, we need some more terminology and concepts. Recall our discussion of Markov Models in :numref:`sec_sequence`.
Let us apply this to language modeling. A distribution over sequences satisfies the Markov property of first order if $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$. Higher orders correspond to longer dependencies. This leads to a number of approximations that we could apply to model a sequence:

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

The probability formulae that involve one, two, and three variables are typically referred to as *unigram*, *bigram*, and *trigram* models, respectively. In the following, we will learn how to design better models.

## Natural Language Statistics

Let us see how this works on real data.
We construct a vocabulary based on the time machine dataset as introduced in :numref:`sec_text_preprocessing` 
and print the top 10 most frequent words.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
#@tab all
tokens = d2l.tokenize(d2l.read_time_machine())
# Since each text line is not necessarily a sentence or a paragraph, we
# concatenate all text lines 
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
```

As we can see, the most popular words are actually quite boring to look at.
They are often referred to as *stop words* and thus filtered out.
Nonetheless, they still carry meaning and we will still use them.
Besides, it is quite clear that the word frequency decays rather rapidly. The $10^{\mathrm{th}}$ most frequent word is less than $1/5$ as common as the most popular one. To get a better idea, we plot the figure of the word frequency.

```{.python .input}
#@tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

We are on to something quite fundamental here: the word frequency decays rapidly in a well-defined way.
After dealing with the first few words as exceptions, all the remaining words roughly follow a straight line on a log-log plot. This means that words satisfy *Zipf's law*,
which states that the frequency $n_i$ of the $i^\mathrm{th}$ most frequent word
is:

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

which is equivalent to

$$\log n_i = -\alpha \log i + c,$$

where $\alpha$ is the exponent that characterizes the distribution and $c$ is a constant.
This should already give us pause if we want to model words by count statistics and smoothing.
After all, we will significantly overestimate the frequency of the tail, also known as the infrequent words. But what about the other word combinations, such as bigrams, trigrams, and beyond?
Let us see whether the bigram frequency behaves in the same manner as the unigram frequency.

```{.python .input}
#@tab all
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

One thing is notable here. Out of the ten most frequent word pairs, nine are composed of both stop words and only one is relevant to the actual book---"the time". Furthermore, let us see whether the trigram frequency behaves in the same manner.

```{.python .input}
#@tab all
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

Last, let us visualize the token frequency among these three models: unigrams, bigrams, and trigrams.

```{.python .input}
#@tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

This figure is quite exciting for a number of reasons. First, beyond unigram words, sequences of words also appear to be following Zipf's law, albeit with a smaller exponent $\alpha$ in :eqref:`eq_zipf_law`, depending on the sequence length.
Second, the number of distinct $n$-grams is not that large. This gives us hope that there is quite a lot of structure in language.
Third, many $n$-grams occur very rarely, which makes Laplace smoothing rather unsuitable for language modeling. Instead, we will use deep learning based models.


## Reading Long Sequence Data

Since sequence data are by their very nature sequential, we need to address
the issue of processing it.
We did so in a rather ad-hoc manner in :numref:`sec_sequence`.
When sequences get too long to be processed by models
all at once,
we may wish to split such sequences for reading.
Now let us describe general strategies.
Before introducing the model,
let us assume that we will use a neural network to train a language model,
where the network processes a minibatch of sequences with predefined length, say $n$ time steps, at a time.
Now the question is how to read minibatches of features and labels at random.

To begin with,
since a text sequence can be arbitrarily long,
such as the entire *The Time Machine* book,
we can partition such a long sequence into subsequences
with the same number of time steps.
When training our neural network,
a minibatch of such subsequences
will be fed into the model.
Suppose that the network processes a subsequence
of $n$ time steps
at a time.
:numref:`fig_timemachine_5gram`
shows all the different ways to obtain subsequences
from an original text sequence, where $n=5$ and a token at each time step corresponds to a character.
Note that we have quite some freedom since we could pick an arbitrary offset that indicates the initial position.

![Different offsets lead to different subsequences when splitting up text.](../img/timemachine-5gram.svg)
:label:`fig_timemachine_5gram`

Hence, which one should we pick from :numref:`fig_timemachine_5gram`?
In fact, all of them are equally good.
However, if we pick just one offset,
there is limited coverage of all the possible subsequences
for training our network.
Therefore,
we can start with a random offset to partition a sequence
to get both *coverage* and *randomness*.
In the following,
we describe how to accomplish this for both
*random sampling* and *sequential partitioning* strategies.


### Random Sampling

In random sampling, each example is a subsequence arbitrarily captured on the original long sequence.
The subsequences from two adjacent random minibatches
during iteration
are not necessarily adjacent on the original sequence.
For language modeling,
the target is to predict the next token based on what tokens we have seen so far, hence the labels are the original sequence, shifted by one token.

The following code randomly generates a minibatch from the data each time.
Here, the argument `batch_size` specifies the number of subsequence examples in each minibatch
and `num_steps` is the predefined number of time steps
in each subsequence.

```{.python .input}
#@tab all
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield d2l.tensor(X), d2l.tensor(Y)
```

Let us manually generate a sequence from 0 to 34.
We assume that
the batch size and numbers of time steps are 2 and 5,
respectively.
This means that we can generate $\lfloor (35 - 1) / 5 \rfloor= 6$ feature-label subsequence pairs. With a minibatch size of 2, we only get 3 minibatches.

```{.python .input}
#@tab all
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

### Sequential Partitioning

In addition to random sampling of the original sequence, we can also ensure that 
the subsequences from two adjacent minibatches
during iteration
are adjacent on the original sequence.
This strategy preserves the order of split subsequences when iterating over minibatches, hence is called sequential partitioning.

```{.python .input}
#@tab mxnet, pytorch
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

```{.python .input}
#@tab tensorflow
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs = d2l.reshape(Xs, (batch_size, -1))
    Ys = d2l.reshape(Ys, (batch_size, -1))
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

Using the same settings,
let us print features `X` and labels `Y` for each minibatch of subsequences read by sequential partitioning.
Note that
the subsequences from two adjacent minibatches
during iteration
are indeed adjacent on the original sequence.

```{.python .input}
#@tab all
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

Now we wrap the above two sampling functions to a class so that we can use it as a data iterator later.

```{.python .input}
#@tab all
class SeqDataLoader:  #@save
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

Last, we define a function `load_data_time_machine` that returns both the data iterator and the vocabulary, so we can use it similarly as other other functions with the `load_data` prefix, such as `d2l.load_data_fashion_mnist` defined in :numref:`sec_fashion_mnist`.

```{.python .input}
#@tab all
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

## Summary

* Language models are key to natural language processing.
* $n$-grams provide a convenient model for dealing with long sequences by truncating the dependence.
* Long sequences suffer from the problem that they occur very rarely or never.
* Zipf's law governs the word distribution for not only unigrams but also the other $n$-grams.
* There is a lot of structure but not enough frequency to deal with infrequent word combinations efficiently via Laplace smoothing.
* The main choices for reading long sequences are random sampling and sequential partitioning. The latter can ensure that the subsequences from two adjacent minibatches during iteration are adjacent on the original sequence.


## Exercises

1. Suppose there are $100,000$ words in the training dataset. How much word frequency and multi-word adjacent frequency does a four-gram need to store?
1. How would you model a dialogue?
1. Estimate the exponent of Zipf's law for unigrams, bigrams, and trigrams.
1. What other methods can you think of for reading long sequence data?
1. Consider the random offset that we use for reading long sequences.
    1. Why is it a good idea to have a random offset?
    1. Does it really lead to a perfectly uniform distribution over the sequences on the document?
    1. What would you have to do to make things even more uniform?
1. If we want a sequence example to be a complete sentence, what kind of problem does this introduce in minibatch sampling? How can we fix the problem?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTUyMTAxODA5NiwtNDM2MzUzMzQ1LC02Mj
k0NTY4NDJdfQ==
-->
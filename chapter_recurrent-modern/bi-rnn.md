# Redes Neurais Recorrentes Bidirecionais
:label:`sec_bi_rnn`

No aprendizado sequencial,
até agora, assumimos que nosso objetivo é modelar a próxima saída, dado o que vimos até agora, por exemplo, no contexto de uma série temporal ou no contexto de um modelo de linguagem. Embora este seja um cenário típico, não é o único que podemos encontrar. Para ilustrar o problema, considere as três tarefas a seguir para preencher o espaço em branco em uma sequência de texto:

* Eu estou`___`.
* Eu estou `___` faminto.
* Eu estou `___` faminto, e poderia comer meio porco.

Dependendo da quantidade de informações disponíveis, podemos preencher os espaços em branco com palavras muito diferentes, como "feliz", "não" e "muito".
Claramente, o final da frase (se disponível) transmite informações significativas sobre qual palavra escolher.
Um modelo de sequência que é incapaz de tirar vantagem disso terá um desempenho ruim em tarefas relacionadas.
Por exemplo, para se sair bem no reconhecimento de entidade nomeada (por exemplo, para reconhecer se "Verde" se refere a "Sr. Verde" ou à cor)
contexto de longo alcance é igualmente vital.
Para obter alguma inspiração para abordar o problema, façamos um desvio para modelos gráficos probabilísticos.


## Programação dinâmica em modelos de Markov ocultos

Esta subseção serve para ilustrar o problema de programação dinâmica. Os detalhes técnicos específicos não importam para a compreensão dos modelos de aprendizagem profunda
mas ajudam a motivar por que se pode usar o aprendizado profundo e por que se pode escolher arquiteturas específicas.

Se quisermos resolver o problema usando modelos gráficos probabilísticos, poderíamos, por exemplo, projetar um modelo de variável latente como segue.
A qualquer momento, passo $t$,
assumimos que existe alguma variável latente $h_t$ que governa nossa emissão observada $x_t$ via $P(x_t \mid h_t)$.
Além disso, qualquer transição $h_t \to h_{t+1}$ é dada por alguma probabilidade de transição de estado $P(h_{t+1} \mid h_{t})$. Este modelo gráfico probabilístico é então um *modelo Markov oculto* como em :numref: =`fig_hmm`.

![Um modelo de Markov oculto.](../img/hmm.svg)
:label:`fig_hmm`

Desse modo,
para uma sequência de observações $T$, temos a seguinte distribuição de probabilidade conjunta sobre os estados observados e ocultos:

$$P(x_1, \ldots, x_T, h_1, \ldots, h_T) = \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t), \text{ onde} P(h_1 \mid h_0) = P(h_1).$$
:eqlabel:`eq_hmm_jointP`

Agora suponha que observamos todos os $x_i$ com exceção de alguns $x_j$ e nosso objetivo é calcular $P(x_j \mid x_{-j})$, onde $x_{-j} = (x_1, \ldots, x_{j-1}, x_{j+1}, \ldots, x_{T})$.
Uma vez que não há variável latente
em $P(x_j \mid x_{-j})$,
nós consideramos resumir
todas as combinações possíveis de escolhas para $h_1, \ldots, h_T$.
No caso de qualquer $h_i$ poder assumir valores distintos de $k$ (um número finito de estados), isso significa que precisamos somar mais de $k^T$ termos --- normalmente missão impossível! Felizmente, existe uma solução elegante para isso: *programação dinâmica*.

Para ver como funciona,
considere somar as variáveis latentes
$h_1, \ldots, h_T$ por sua vez.
De acordo com :eqref:`eq_hmm_jointP`,
isso produz:

$$\begin{aligned}
    &P(x_1, \ldots, x_T) \\
    =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_2, \ldots, h_T} \underbrace{\left[\sum_{h_1} P(h_1) P(x_1 \mid h_1) P(h_2 \mid h_1)\right]}_{\pi_2(h_2) \stackrel{\mathrm{def}}{=}}
    P(x_2 \mid h_2) \prod_{t=3}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t) \\
    =& \sum_{h_3, \ldots, h_T} \underbrace{\left[\sum_{h_2} \pi_2(h_2) P(x_2 \mid h_2) P(h_3 \mid h_2)\right]}_{\pi_3(h_3)\stackrel{\mathrm{def}}{=}}
    P(x_3 \mid h_3) \prod_{t=4}^T P(h_t \mid h_{t-1}) P(x_t \mid h_t)\\
    =& \dots \\
    =& \sum_{h_T} \pi_T(h_T) P(x_T \mid h_T).
\end{aligned}$$

Em geral, temos a *recursão direta* como

$$\pi_{t+1}(h_{t+1}) = \sum_{h_t} \pi_t(h_t) P(x_t \mid h_t) P(h_{t+1} \mid h_t).$$

A recursão é inicializada como $\pi_1(h_1) = P(h_1)$. Em termos abstratos, isso pode ser escrito como $\pi_{t+1} = f(\pi_t, x_t)$, onde $f$ é alguma função aprendível. Isso se parece muito com a equação de atualização nos modelos de variáveis latentes que discutimos até agora no contexto de RNNs!

De forma totalmente análoga à recursão direta,
nós também podemos
soma sobre o mesmo conjunto de variáveis latentes com uma recursão para trás. Isso produz:

$$\begin{aligned}
    & P(x_1, \ldots, x_T) \\
     =& \sum_{h_1, \ldots, h_T} P(x_1, \ldots, x_T, h_1, \ldots, h_T) \\
    =& \sum_{h_1, \ldots, h_T} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot P(h_T \mid h_{T-1}) P(x_T \mid h_T) \\
    =& \sum_{h_1, \ldots, h_{T-1}} \prod_{t=1}^{T-1} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_T} P(h_T \mid h_{T-1}) P(x_T \mid h_T)\right]}_{\rho_{T-1}(h_{T-1})\stackrel{\mathrm{def}}{=}} \\
    =& \sum_{h_1, \ldots, h_{T-2}} \prod_{t=1}^{T-2} P(h_t \mid h_{t-1}) P(x_t \mid h_t) \cdot
    \underbrace{\left[\sum_{h_{T-1}} P(h_{T-1} \mid h_{T-2}) P(x_{T-1} \mid h_{T-1}) \rho_{T-1}(h_{T-1}) \right]}_{\rho_{T-2}(h_{T-2})\stackrel{\mathrm{def}}{=}} \\
    =& \ldots \\
    =& \sum_{h_1} P(h_1) P(x_1 \mid h_1)\rho_{1}(h_{1}).
\end{aligned}$$

Podemos, portanto, escrever a *recursão para trás* como

$$\rho_{t-1}(h_{t-1})= \sum_{h_{t}} P(h_{t} \mid h_{t-1}) P(x_{t} \mid h_{t}) \rho_{t}(h_{t}),$$

com inicialização $\rho_T(h_T) = 1$. 
Ambas as recursões para frente e para trás nos permitem somar mais de $T$ variáveis latentes em $\mathcal{O}(kT)$ (linear) tempo sobre todos os valores de $(h_1, \ldots, h_T)$ em vez de em tempo exponencial .
Este é um dos grandes benefícios da inferência probabilística com modelos gráficos.
Isto é
também uma instância muito especial de
um algoritmo geral de passagem de mensagens :cite:`Aji.McEliece.2000`.
Combinando recursões para frente e para trás, somos capazes de calcular

$$P(x_j \mid x_{-j}) \propto \sum_{h_j} \pi_j(h_j) \rho_j(h_j) P(x_j \mid h_j).$$

Observe que, em termos abstratos, a recursão para trás pode ser escrita como $\rho_{t-1} = g(\rho_t, x_t)$, onde $g$ é uma função que pode ser aprendida. Novamente, isso se parece muito com uma equação de atualização, apenas rodando para trás, ao contrário do que vimos até agora em RNNs. Na verdade, os modelos ocultos de Markov se beneficiam do conhecimento de dados futuros, quando estiverem disponíveis. Os cientistas de processamento de sinais distinguem entre os dois casos de conhecimento e não conhecimento de observações futuras como interpolação vs. extrapolação.
Veja o capítulo introdutório do livro sobre algoritmos sequenciais de Monte Carlo para mais detalhes :cite:`Doucet.De-Freitas.Gordon.2001`.


## Modelo Bid

If we want to have a mechanism in RNNs that offers comparable look-ahead ability as in hidden Markov models, we need to modify the RNN design that we have seen so far. Fortunately, this is easy conceptually. Instead of running an RNN only in the forward mode starting from the first token, we start another one from the last token running from back to front. 
*Bidirectional RNNs* add a hidden layer that passes information in a backward direction to more flexibly process such information. :numref:`fig_birnn` illustrates the architecture of a bidirectional RNN with a single hidden layer.

![Architecture of a bidirectional RNN.](../img/birnn.svg)
:label:`fig_birnn`

In fact, this is not too dissimilar to the forward and backward recursions in the dynamic programing of hidden Markov models. 
The main distinction is that in the previous case these equations had a specific statistical meaning.
Now they are devoid of such easily accessible interpretations and we can just treat them as 
generic and learnable functions.
This transition epitomizes many of the principles guiding the design of modern deep networks: first, use the type of functional dependencies of classical statistical models, and then parameterize them in a generic form.


### Definition

Bidirectional RNNs were introduced by :cite:`Schuster.Paliwal.1997`. 
For a detailed discussion of the various architectures see also the paper :cite:`Graves.Schmidhuber.2005`.
Let us look at the specifics of such a network.


For any time step $t$, 
given a minibatch input $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (number of examples: $n$, number of inputs in each example: $d$) and let the hidden layer activation function be $\phi$. In the bidirectional architecture, we assume that the forward and backward hidden states for this time step are $\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ and $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$, respectively,
where $h$ is the number of hidden units.
The forward and backward hidden state updates are as follows:


$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
\end{aligned}
$$

where the weights $\mathbf{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \text{ and } \mathbf{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$, and biases $\mathbf{b}_h^{(f)} \in \mathbb{R}^{1 \times h} \text{ and } \mathbf{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$ are all the model parameters.

Next, we concatenate the forward and backward hidden states $\overrightarrow{\mathbf{H}}_t$ and $\overleftarrow{\mathbf{H}}_t$
to obtain the hidden state $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$ to be fed into the output layer.
In deep bidirectional RNNs with multiple hidden layers,
such information
is passed on as *input* to the next bidirectional layer. Last, the output layer computes the output $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (number of outputs: $q$):

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

Here, the weight matrix $\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$ and the bias $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ are the model parameters of the output layer. In fact, the two directions can have different numbers of hidden units.

### Computational Cost and Applications

One of the key features of a bidirectional RNN is that information from both ends of the sequence is used to estimate the output. That is, we use information from both future and past observations to predict the current one. 
In the case of next token prediction this is not quite what we want.
After all, we do not have the luxury of knowing the next to next token when predicting the next one. Hence, if we were to use a bidirectional RNN naively we would not get a very good accuracy: during training we have past and future data to estimate the present. During test time we only have past data and thus poor accuracy. We will illustrate this in an experiment below.

To add insult to injury, bidirectional RNNs are also exceedingly slow.
The main reasons for this are that 
the forward propagation
requires both forward and backward recursions
in bidirectional layers
and that the backpropagation is dependent on the outcomes of the forward propagation. Hence, gradients will have a very long dependency chain.

In practice bidirectional layers are used very sparingly and only for a narrow set of applications, such as filling in missing words, annotating tokens (e.g., for named entity recognition), and encoding sequences wholesale as a step in a sequence processing pipeline (e.g., for machine translation).
In :numref:`sec_bert` and :numref:`sec_sentiment_rnn`,
we will introduce how to use bidirectional RNNs
to encode text sequences.



## Training a Bidirectional RNN for a Wrong Application

If we were to ignore all advice regarding the fact that bidirectional RNNs use past and future data and simply apply it to language models, 
we will get estimates with acceptable perplexity. Nonetheless, the ability of the model to predict future tokens is severely compromised as the experiment below illustrates. 
Despite reasonable perplexity, it only generates gibberish even after many iterations.
We include the code below as a cautionary example against using them in the wrong context.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
lstm_layer = rnn.LSTM(num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# Train the model
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

The output is clearly unsatisfactory for the reasons described above. 
For a
discussion of more effective uses of bidirectional RNNs, please see the sentiment
analysis application 
in :numref:`sec_sentiment_rnn`.

## Summary

* In bidirectional RNNs, the hidden state for each time step is simultaneously determined by the data prior to and after the current time step.
* Bidirectional RNNs bear a striking resemblance with the forward-backward algorithm in probabilistic graphical models.
* Bidirectional RNNs are mostly useful for sequence encoding and the estimation of observations given bidirectional context.
* Bidirectional RNNs are very costly to train due to long gradient chains.

## Exercises

1. If the different directions use a different number of hidden units, how will the shape of $\mathbf{H}_t$ change?
1. Design a bidirectional RNN with multiple hidden layers.
1. Polysemy is common in natural languages. For example, the word "bank" has different meanings in contexts “i went to the bank to deposit cash” and “i went to the bank to sit down”. How can we design a neural network model such that given a context sequence and a word, a vector representation of the word in the context will be returned? What type of neural architectures is preferred for handling polysemy?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1059)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3MzcyNTkzOTcsLTUzNTQ3ODQxM119
-->
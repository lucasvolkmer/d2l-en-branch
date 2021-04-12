# Long Short-Term Memory (LSTM)
:label:`sec_lstm`

The challenge to address long-term information preservation and short-term input
skipping in latent variableMemória Longa de Curto Prazo (LSTM)
:label:`sec_lstm`

O desafio de abordar a preservação de informações de longo prazo e entrada de curto prazo
pular em modelos has existed for a long time. One of the
earliest approaches to address this was the
long short-term memory (LSTM) :cite:`Hochreiter.Schmidhuber.1997`. It shares many of the properties of the
GRU.
Interestingly, LSTMs have a slightly morede variáveis latentes existe há muito tempo. Um dos
as primeiras abordagens para resolver isso foram
memória longa de curto prazo (LSTM) :cite:`Hochreiter.Schmidhuber.1997`. Ele compartilha muitas das propriedades do
GRU.
Curiosamente, os LSTMs têm um aspecto um pouco mais complexo
design than GRUs but predates GRUs by almost twodo que GRUs, mas antecede GRUs em quase duas deécadeas.



## Gated Memory CellCélula de Memória Bloqueada

Indiscutivelmente, o design da LSTM é inspirado
pelas portas lógicas de um computador.
LSTM introduz uma *célula de memória* (ou *célula* para abreviar)
que tem a mesma forma que o estado oculto
(algumas literaturas consideram a célula de memória
como um tipo especial de estado oculto),
projetado para registrar informações adicionais.
Para controlar a célula de memória
precisamos de vários portões.
Um portão é necessário para ler as entradas do
célula.
Vamos nos referir a isso como o
*portão de saída*.
Uma segunda porta é necessária para decidir quando ler os dados para o
célula.
Chamamos isso de *porta de entrada*.
Por último, precisamos de um mecanismo para redefinir
o conteúdo da célula, governado por um *portão de esquecimento*.
A motivação para tal
design é o mesmo das GRUs,
ou seja, ser capaz de decidir quando lembrar e
quando ignorar entradas no estado oculto por meio de um mecanismo dedicado. Deixe-nos ver
como isso funciona na prática.

### Porta de entrada, porta de esquecimento e porta de saída

Assim como em GRUs,
os dados que alimentam as portas LSTM são
a entrada na etapa de tempo atual e
o estado oculto da etapa de tempo anterior,
conforme ilustrado em :numref:`lstm_0`.
Eles são processados por
três camadas totalmente conectadas com uma função de ativação sigmóide para calcular os valores de
a entrada, esqueça. e portas de saída.
Como resultado, os valores das três portas
estão na faixa de $(0, 1)$.

![Calculando a porta de entrada, a porta de esquecimento e a porta de saída em um modelo LSTM.](../img/lstm-0.svg)
:label:`lstm_0`

Matematicamente,
suponha que existam $h$ unidades ocultas, o tamanho do lote é $n$ e o número de entradas é $d$.
Assim, a entrada é $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ e o estado oculto da etapa de tempo anterior é $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$. Correspondentemente, as portas na etapa de tempo $t$
são definidos da seguinte forma: a porta de entrada é $\mathbf{I}_t \in \mathbb{R}^{n \times h}$, a porta de esquecimento é $\mathbf{F}_t \in \mathbb{R}^{n \times h}$, e a porta de saída é $\mathbf{O}_t \in \mathbb{R}^{n \times h}$. Eles são calculados da seguinte forma:

$$
\begin{aligned}
\mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i),\\
\mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f),\\
\mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o),
\end{aligned}
$$

onde $\mathbf{W}_{xi}, \mathbf{W}_{xf}, \mathbf{W}_{xo} \in \mathbb{R}^{d \times h}$ e $\mathbf{W}_{hi}, \mathbf{W}_{hf}, \mathbf{W}_{ho} \in \mathbb{R}^{h \times h}$ são parâmetros de pesos e $\mathbf{b}_i, \mathbf{b}_f, \mathbf{b}_o \in \mathbb{R}^{1 \times h}$ são parâmetros viéses.

### Célula de Memória Candidata

Em seguida, projetamos a célula de memória. Como ainda não especificamos a ação das várias portas, primeiro introduzimos a célula de memória *candidata* $\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$. Seu cálculo é semelhante ao das três portas descritas acima, mas usando uma função $\tanh$ com um intervalo de valores para $(-1,1)$ como a função de ativação. Isso leva à seguinte equação na etapa de tempo $t$:

$$\tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c),$$

onde $\mathbf{W}_{xc} \in \mathbb{R}^{d \times h}$ and $\mathbf{W}_{hc} \in \mathbb{R}^{h \times h}$ são parâmetros de pesos $\mathbf{b}_c \in \mathbb{R}^{1 \times h}$ é um parâmetro de viés.

Uma ilustração rápida da célula de memória candidata é mostrada em :numref:`lstm_1`.

![Computando a célula de memória candidata em um modelo LSTM.](../img/lstm-1.svg)
:label:`lstm_1`

### Célula de Memória

Em GRUs, temos um mecanismo para controlar a entrada e o esquecimento (ou salto).
De forma similar,
em LSTMs, temos duas portas dedicadas para tais propósitos: a porta de entrada $\mathbf{I}_t$ governa o quanto levamos os novos dados em conta via $\tilde{\mathbf{C}}_t$ e a porta de esquecer $\mathbf{F}_t$ aborda quanto do conteúdo da célula de memória antiga $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$ retemos. Usando o mesmo truque de multiplicação pontual de antes, chegamos à seguinte equação de atualização:

$$\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.$$

Se a porta de esquecimento é sempre aproximadamente 1 e a porta de entrada é sempre aproximadamente 0, as células de memória anteriores $\mathbf{C}_{t-1}$ serão salvas ao longo do tempo e passadas para o intervalo de tempo atual.
Este projeto é introduzido para aliviar o problema do gradiente de desaparecimento e para melhor capturar
dependências de longo alcance dentro de sequências.

Assim, chegamos ao diagrama de fluxo em :numref:`lstm_2`.

![Calculando a célula de memória em um modelo LSTM.](../img/lstm-2.svg)

:label:`lstm_2`


### Estado Oculto

Por último, precisamos definir como calcular o estado oculto $\mathbf{H}_t \in \mathbb{R}^{n \times h}$. É aqui que a porta de saída entra em ação. No LSTM, é simplesmente uma versão bloqueada do $\tanh$ da célula de memória.
Isso garante que os valores de $\mathbf{H}_t$ estejam sempre no intervalo $(-1,1)$.

$$\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).$$

Sempre que a porta de saída se aproxima de 1, passamos efetivamente todas as informações da memória para o preditor, enquanto para a porta de saída próxima de 0, retemos todas as informações apenas dentro da célula de memória e não realizamos nenhum processamento posterior.

:numref:`lstm_3` tem uma ilustração gráfica do fluxo de dados.

![Calculando o estado oculto em um modelo LSTM.](../img/lstm-3.svg)
:label:`lstm_3`



## Implementação do zero

Agora, vamos implementar um LSTM do zero.
Da mesma forma que os experimentos em :numref:`sec_rnn_scratch`,
primeiro carregamos o conjunto de dados da máquina do tempo.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

### Initializing Model Parameters

Next we need to define and initialize the model parameters. As previously, the hyperparameter `num_hiddens` defines the number of hidden units. We initialize weights following a Gaussian distribution with 0.01 standard deviation, and we set the biases to 0.

```{.python .input}
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

### Defining the Model

In the initialization function, the hidden state of the LSTM needs to return an *additional* memory cell with a value of 0 and a shape of (batch size, number of hidden units). Hence we get the following state initialization.

```{.python .input}
def init_lstm_state(batch_size, num_hiddens, device):
    return (np.zeros((batch_size, num_hiddens), ctx=device),
            np.zeros((batch_size, num_hiddens), ctx=device))
```

```{.python .input}
#@tab pytorch
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
```

The actual model is defined just like what we discussed before: providing three gates and an auxiliary memory cell. Note that only the hidden state is passed to the output layer. The memory cell $\mathbf{C}_t$ does not directly participate in the output computation.

```{.python .input}
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = npx.sigmoid(np.dot(X, W_xi) + np.dot(H, W_hi) + b_i)
        F = npx.sigmoid(np.dot(X, W_xf) + np.dot(H, W_hf) + b_f)
        O = npx.sigmoid(np.dot(X, W_xo) + np.dot(H, W_ho) + b_o)
        C_tilda = np.tanh(np.dot(X, W_xc) + np.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * np.tanh(C)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H, C)
```

```{.python .input}
#@tab pytorch
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

### Training and Prediction

Let us train an LSTM as same as what we did in :numref:`sec_gru`, by instantiating the `RNNModelScratch` class as introduced in :numref:`sec_rnn_scratch`.

```{.python .input}
#@tab all
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## Concise Implementation

Using high-level APIs,
we can directly instantiate an `LSTM` model.
This encapsulates all the configuration details that we made explicit above. The code is significantly faster as it uses compiled operators rather than Python for many details that we spelled out in detail before.

```{.python .input}
lstm_layer = rnn.LSTM(num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

LSTMs are the prototypical latent variable autoregressive model with nontrivial state control.
Many variants thereof have been proposed over the years, e.g., multiple layers, residual connections, different types of regularization. However, training LSTMs and other sequence models (such as GRUs) are quite costly due to the long range dependency of the sequence.
Later we will encounter alternative models such as transformers that can be used in some cases.


## Summary

* LSTMs have three types of gates: input gates, forget gates, and output gates that control the flow of information.
* The hidden layer output of LSTM includes the hidden state and the memory cell. Only the hidden state is passed into the output layer. The memory cell is entirely internal.
* LSTMs can alleviate vanishing and exploding gradients.


## Exercises

1. Adjust the hyperparameters and analyze the their influence on running time, perplexity, and the output sequence.
1. How would you need to change the model to generate proper words as opposed to sequences of characters?
1. Compare the computational cost for GRUs, LSTMs, and regular RNNs for a given hidden dimension. Pay special attention to the training and inference cost.
1. Since the candidate memory cell ensures that the value range is between $-1$ and $1$ by  using the $\tanh$ function, why does the hidden state need to use the $\tanh$ function again to ensure that the output value range is between $-1$ and $1$?
1. Implement an LSTM model for time series prediction rather than character sequence prediction.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/343)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1057)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjQ2NDgxODY4XX0=
-->
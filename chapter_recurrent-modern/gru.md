# Gated Recurrent Units (GRU)
:label:`sec_gru`

* We might encounter a situation where an early observation is highly
  significant for predicting all future observations. Consider the somewhat
  contrived case where the first observation contains a checksum and the goal is
  to discern whether the checksum is correct at the end of the sequence. In this
  case, the influence of the first token is vital. We would like to have some
  mechanisms for storing vital early information in a *memory cell*. Without such
  a mechanism, we will have to assign a very large gradient to this observation,
  since it affects all the subsequent observations.
* We might encounter situations where some tokens carry no pertinent
  observation. For instance, when parsing a web page there might be auxiliary
  HTML code that is irrelevant for the purpose of assessing the sentiment
  conveyed on the page. We would like to have some mechanism for *skipping* such
  tokens in the latent state representation.
* We might encounter situations where there is a logical break between parts of
  a sequence. For instance, there might be a transition between chapters in a
  book, or a transition between a bear and a bull market for securities. In
  this case it would be nice to have a means of *resetting* our internal state
  representation.

Em :numref:`sec_bptt`,
discutimos como os gradientes são calculados
em RNNs.
Em particular, descobrimos que produtos longos de matrizes podem levar
para gradientes desaparecendo ou explodindo.
Vamos pensar brevemente sobre como
anomalias de gradiente significam na prática:

* Podemos encontrar uma situação em que uma observação precoce é altamente
  significativo para prever todas as observações futuras. Considere o pouco
  caso inventado em que a primeira observação contém uma soma de verificação e o objetivo é
  para discernir se a soma de verificação está correta no final da sequência. Nisso
  caso, a influência do primeiro token é vital. Gostaríamos de ter algum
  mecanismos para armazenar informações precoces vitais em uma *célula de memória*. Sem tal
  um mecanismo, teremos que atribuir um gradiente muito grande a esta observação,
  uma vez que afeta todas as observações subsequentes.
* Podemos encontrar situações em que alguns tokens não carreguem
  observação. Por exemplo, ao analisar uma página da web, pode haver
  Código HTML que é irrelevante para o propósito de avaliar o sentimento
  transmitido na página. Gostaríamos de ter algum mecanismo para *pular* tais
  tokens na representação do estado latente.
* Podemos encontrar situações em que haja uma quebra lógica entre as partes do
  uma sequência. Por exemplo, pode haver uma transição entre capítulos em um
  livro, ou uma transição entre um mercado de valores em baixa e em alta. No
  neste caso seria bom ter um meio de *redefinir* nosso estado interno
  representação.

Vários métodos foram propostos para resolver isso. Uma das mais antigas é a memória de curto prazo longa :cite:`Hochreiter.Schmidhuber.1997` que nós
iremos discutir em :numref:`sec_lstm`. A unidade recorrente fechada (GRU)
:cite:`Cho.Van-Merrienboer.Bahdanau.ea.2014` é um pouco mais simplificado
variante que muitas vezes oferece desempenho comparável e é significativamente mais rápido para
compute :cite:`Chung.Gulcehre.Cho.ea.2014`.
Por sua simplicidade, comecemos com o GRU.

## Estado Oculto Fechado

The key distinction between vanilla RNNs and GRUs
is that the latter support gating of the hidden state.
This means that we have dedicated mechanisms for
when a hidden state should be *updated* and
also when it should be *reset*.
These mechanisms are learned and they address the concerns listed above.
For instance, if the first token is of great importance
we will learn not to update the hidden state after the first observation.
Likewise, we will learn to skip irrelevant temporary observations.
Last, we will learn to reset the latent state whenever needed.
We discuss this in detail below.

A principal distinção entre RNNs vanilla e GRUs
é que o último suporta o bloqueio do estado oculto.
Isso significa que temos mecanismos dedicados para
quando um estado oculto deve ser *atualizado* e
também quando deve ser *redefinido*.
Esses mecanismos são aprendidos e atendem às questões listadas acima.
Por exemplo, se o primeiro token é de grande importância
aprenderemos a não atualizar o estado oculto após a primeira observação.
Da mesma forma, aprenderemos a pular observações temporárias irrelevantes.
Por último, aprenderemos a redefinir o estado latente sempre que necessário.
Discutimos isso em detalhes abaixo.

### Reset Gate and Update Gate

The first thing we need to introduce are
the *reset gate* and the *update gate*.
We engineer them to be vectors with entries in $(0, 1)$
such that we can perform convex combinations.
For instance,
a reset gate would allow us to control how much of the previous state we might still want to remember.
Likewise, an update gate would allow us to control how much of the new state is just a copy of the old state.

A primeira coisa que precisamos apresentar é
a * porta de reinicialização * e a * porta de atualização *.
Nós os projetamos para serem vetores com entradas em $ (0, 1) $
para que possamos realizar combinações convexas.
Por exemplo,
uma porta de reinicialização nos permitiria controlar quanto do estado anterior ainda podemos querer lembrar.
Da mesma forma, uma porta de atualização nos permitiria controlar quanto do novo estado é apenas uma cópia do antigo estado.

We begin by engineering these gates.
:numref:`fig_gru_1` illustrates the inputs for both
the reset and update gates in a GRU, given the input
of the current time step
and the hidden state of the previous time step.
The outputs of two gates
are given by two fully-connected layers
with a sigmoid activation function.

Começamos projetando esses portões.
:numref:`fig_gru_1` ilustra as entradas para ambos
as portas de reset e atualização em uma GRU, dada a entrada
da etapa de tempo atual
e o estado oculto da etapa de tempo anterior.
As saídas de duas portas
são fornecidos por duas camadas totalmente conectadas
com uma função de ativação sigmóide.

![Calculando a porta de reinicialização e a porta de atualização em um modelo GRU.](../img/gru-1.svg)
:label:`fig_gru_1`

Mathematically,
for a given time step $t$,
suppose that the input is
a minibatch
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (number of examples: $n$, number of inputs: $d$) and the hidden state of the previous time step is $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ (number of hidden units: $h$). Then, the reset gate $\mathbf{R}_t \in \mathbb{R}^{n \times h}$ and update gate $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$ are computed as follows:

Matematicamente,
para um determinado intervalo de tempo $ t $,
suponha que a entrada seja
um minibatch
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (número de exemplos: $ n $, número de entradas: $ d $) e o estado oculto da etapa de tempo anterior é $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ (número de unidades ocultas: $ h $). Então, reinicie o portão $\mathbf{R}_t \in \mathbb{R}^{n \times h}$ e atualize o portão $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$ são calculados da seguinte forma:

$$
\begin{aligned}
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),
\end{aligned}
$$

onde $\mathbf{W}_{xr}, \mathbf{W}_{xz} \in \mathbb{R}^{d \times h}$ e
$\mathbf{W}_{hr}, \mathbf{W}_{hz} \in \mathbb{R}^{h \times h}$ são pesos de parâmetros e $\mathbf{b}_r, \mathbf{b}_z \in \mathbb{R}^{1 \times h}$ são viéses.
Note that broadcasting (see :numref:`subsec_broadcasting`) is triggered during the summation.
We use sigmoid functions (as introduced in :numref:`sec_mlp`) to transform input values to the interval $(0, 1)$.
Observe que a transmissão (consulte :numref:`subsec_broadcasting`) é acionada durante a soma.
Usamos funções sigmóides (como introduzidas em :numref:`sec_mlp`) para transformar os valores de entrada no intervalo $(0, 1)$.

### Candidate Hidden State Estado Oculto do Candidato

Next, let us
integrate the reset gate $\mathbf{R}_t$ with
the regular latent state updating mechanism
in :eqref:`rnn_h_with_state`.
It leads to the following
*candidate hidden state*

Em seguida, vamos
integrar a porta de reset $\mathbf {R} _t$ com
o mecanismo regular de atualização de estado latente
in :eqref:`rnn_h_with_state`.
Isso leva ao seguinte
*estado oculto candidato*
$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$ at time step $t$:

$$\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h),$$
:eqlabel:`gru_tilde_H`

onde $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$ and $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$
são parâmetros de pesos,
$\mathbf{b}_h \in \mathbb{R}^{1 \times h}$
is the bias,
and the symbol $\odot$ is the Hadamard (elementwise) product operator.
Here we use a nonlinearity in the form of tanh to ensure that the values in the candidate hidden state remain in the interval $(-1, 1)$.

é o preconceito,
e o símbolo $\odot$ é o operador de produto Hadamard (elementwise).
Aqui, usamos uma não linearidade na forma de tanh para garantir que os valores no estado oculto candidato permaneçam no intervalo $(-1, 1)$.

The result is a *candidate* since we still need to incorporate the action of the update gate.
Comparing with :eqref:`rnn_h_with_state`,
now the influence of the previous states
can be reduced with the
elementwise multiplication of
$\mathbf{R}_t$ and $\mathbf{H}_{t-1}$
in :eqref:`gru_tilde_H`.
Whenever the entries in the reset gate $\mathbf{R}_t$ are close to 1, we recover a vanilla RNN such as in :eqref:`rnn_h_with_state`.
For all entries of the reset gate $\mathbf{R}_t$ that are close to 0, the candidate hidden state is the result of an MLP with $\mathbf{X}_t$ as the input. Any pre-existing hidden state is thus *reset* to defaults.

:numref:`fig_gru_2` illustrates the computational flow after applying the reset gate.

O resultado é um *candidato*, pois ainda precisamos incorporar a ação da porta de atualização.
Comparando com :eqref:`rnn_h_with_state`,
agora a influência dos estados anteriores
pode ser reduzido com o
multiplicação elementar de
$\mathbf{R}_t$ e$\mathbf{H}_{t-1}$
em :eqref:`gru_tilde_H`.
Sempre que as entradas na porta de reset $\mathbf{R}_t$ estão perto de 1, recuperamos um RNN vanilla como em :eqref:`rnn_h_with_state`.
Para todas as entradas da porta de reset $\mathbf{R}_t$ que estão próximas de 0, o estado oculto candidato é o resultado de um MLP com $\mathbf{X}_t$ como a entrada. Qualquer estado oculto pré-existente é, portanto, *redefinido* para os padrões.

:numref:`fig_gru_2` ilustra o fluxo computacional após aplicar a porta de reset.

![Calculando o estado oculto candidato em um modelo GRU.](../img/gru-2.svg)
:label:`fig_gru_2`


### Estados Escondidos

Finally, we need to incorporate the effect of the update gate $\mathbf{Z}_t$. This determines the extent to which the new hidden state $\mathbf{H}_t \in \mathbb{R}^{n \times h}$ is just the old state $\mathbf{H}_{t-1}$ and by how much the new candidate state $\tilde{\mathbf{H}}_t$ is used.
The update gate $\mathbf{Z}_t$ can be used for this purpose, simply by taking elementwise convex combinations between both $\mathbf{H}_{t-1}$ and $\tilde{\mathbf{H}}_t$.
This leads to the final update equation for the GRU:

Finalmente, precisamos incorporar o efeito da porta de atualização $\mathbf{Z}_t$. Isso determina até que ponto o novo estado oculto $\mathbf{H}_t \in \mathbb{R}^{n \times h}$ é apenas o antigo estado $\mathbf{H}_{t-1}$ e em quanto o novo estado candidato $\tilde{\mathbf{H}}_t$ é usado.
A porta de atualização $\mathbf{Z}_t$ pode ser usada para este propósito, simplesmente tomando combinações convexas elementwise entre $\mathbf{H}_{t-1}$ e $\tilde{\mathbf{H}}_t$.
Isso leva à equação de atualização final para a GRU:

$$\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t.$$


Whenever the update gate $\mathbf{Z}_t$ is close to 1, we simply retain the old state. In this case the information from $\mathbf{X}_t$ is essentially ignored, effectively skipping time step $t$ in the dependency chain. In contrast, whenever $\mathbf{Z}_t$ is close to 0, the new latent state $\mathbf{H}_t$ approaches the candidate latent state $\tilde{\mathbf{H}}_t$. These designs can help us cope with the vanishing gradient problem in RNNs and better capture dependencies for sequences with large time step distances.
For instance,
if the update gate has been close to 1
for all the time steps of an entire subsequence,
the old hidden state at the time step of its beginning
will be easily retained and passed
to its end,
regardless of the length of the subsequence.

Sempre que a porta de atualização $\mathbf{Z}_t$ está próxima de 1, simplesmente mantemos o estado antigo. Neste caso, a informação de $\mathbf{X}_t$ é essencialmente ignorada, pulando efetivamente o passo de tempo $t$ na cadeia de dependências. Em contraste, sempre que $\mathbf{Z}_t$ está próximo de 0, o novo estado latente $\mathbf{H}_t$ se aproxima do estado latente candidato $\tilde{\mathbf{H}}_t$. Esses projetos podem nos ajudar a lidar com o problema do gradiente de desaparecimento em RNNs e melhor capturar dependências para sequências com grandes distâncias de intervalo de tempo.
Por exemplo,
se a porta de atualização estiver perto de 1
para todas as etapas de tempo de uma subsequência inteira,
o antigo estado oculto na etapa do tempo de seu início
será facilmente retido e aprovado
até o fim,
independentemente do comprimento da subsequência.



:numref:`fig_gru_3` illustrates the computational flow after the update gate is in action.

![Computing the hidden state in a GRU model.](../img/gru-3.svg)
:label:`fig_gru_3`


In summary, GRUs have the following two distinguishing features:

* Reset gates help capture short-term dependencies in sequences.
* Update gates help capture long-term dependencies in sequences.

## Implementation from Scratch

To gain a better understanding of the GRU model, let us implement it from scratch. We begin by reading the time machine dataset that we used in :numref:`sec_rnn_scratch`. The code for reading the dataset is given below.

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

The next step is to initialize the model parameters.
We draw the weights from a Gaussian distribution
with standard deviation to be 0.01 and set the bias to 0. The hyperparameter `num_hiddens` defines the number of hidden units.
We instantiate all weights and biases relating to the update gate, the reset gate, the candidate hidden state,
and the output layer.

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                np.zeros(num_hiddens, ctx=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                d2l.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

### Defining the Model

Now we will define the hidden state initialization function `init_gru_state`. Just like the `init_rnn_state` function defined in :numref:`sec_rnn_scratch`, this function returns a tensor with a shape (batch size, number of hidden units) whose values are all zeros.

```{.python .input}
def init_gru_state(batch_size, num_hiddens, device):
    return (np.zeros(shape=(batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

Now we are ready to define the GRU model.
Its structure is the same as that of the basic RNN cell, except that the update equations are more complex.

```{.python .input}
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = npx.sigmoid(np.dot(X, W_xz) + np.dot(H, W_hz) + b_z)
        R = npx.sigmoid(np.dot(X, W_xr) + np.dot(H, W_hr) + b_r)
        H_tilda = np.tanh(np.dot(X, W_xh) + np.dot(R * H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

### Training and Prediction

Training and prediction work in exactly the same manner as in :numref:`sec_rnn_scratch`.
After training,
we print out the perplexity on the training set
and the predicted sequence following
the provided prefixes "time traveller" and "traveller", respectively.

```{.python .input}
#@tab all
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## Concise Implementation

In high-level APIs,
we can directly
instantiate a GPU model.
This encapsulates all the configuration detail that we made explicit above.
The code is significantly faster as it uses compiled operators rather than Python for many details that we spelled out before.

```{.python .input}
gru_layer = rnn.GRU(num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

```{.python .input}
#@tab pytorch
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

## Summary

* Gated RNNs can better capture dependencies for sequences with large time step distances.
* Reset gates help capture short-term dependencies in sequences.
* Update gates help capture long-term dependencies in sequences.
* GRUs contain basic RNNs as their extreme case whenever the reset gate is switched on. They can also skip subsequences by turning on the update gate.


## Exercises

1. Assume that we only want to use the input at time step $t'$ to predict the output at time step $t > t'$. What are the best values for the reset and update gates for each time step?
1. Adjust the hyperparameters and analyze the their influence on running time, perplexity, and the output sequence.
1. Compare runtime, perplexity, and the output strings for `rnn.RNN` and `rnn.GRU` implementations with each other.
1. What happens if you implement only parts of a GRU, e.g., with only a reset gate or only an update gate?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/342)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1056)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTM0NjA3NjkxNiw4NDU0NDY4MTEsLTc3NT
E4NDMyMF19
-->
# *Dropout*
:label:`sec_dropout`

Em :numref:`sec_weight_decay`,
introduzimos a abordagem clássica
para regularizar modelos estatísticos
penalizando a norma $L_2$ dos pesos.
Em termos probabilísticos, poderíamos justificar esta técnica
argumentando que assumimos uma crença anterior
que os pesos tomam valores de
uma distribuição gaussiana com média zero.
Mais intuitivamente, podemos argumentar
que encorajamos o modelo a espalhar seus pesos
entre muitas características, em vez de depender demais
em um pequeno número de associações potencialmente espúrias.

## *Overfitting* Revisitado


Diante de mais características do que exemplos,
modelos lineares tendem a fazer *overfitting*.
Mas dados mais exemplos do que características,
geralmente podemos contar com modelos lineares para não ajustar demais.
Infelizmente, a confiabilidade com a qual
os modelos lineares generalizam têm um custo.
Aplicados ingenuamente, os modelos lineares não levam
em conta as interações entre as características.
Para cada recurso, um modelo linear deve atribuir
um peso positivo ou negativo, ignorando o contexto.

Em textos tradicionais, esta tensão fundamental
entre generalização e flexibilidade
é descrito como a *compensação de variação de polarização*.
Modelos lineares têm alta polarização: eles podem representar apenas uma pequena classe de funções.
No entanto, esses modelos têm baixa variação: eles fornecem resultados semelhantes
em diferentes amostras aleatórias dos dados.

Redes neurais profundas habitam o oposto
fim do espectro de polarização-variância.
Ao contrário dos modelos lineares, as redes neurais
não se limitam a examinar cada recurso individualmente.
Eles podem aprender interações entre grupos de recursos.
Por exemplo, eles podem inferir que
“Nigéria” e “Western Union” aparecendo
juntos em um e-mail indicam spam
mas separadamente eles não o fazem.


Mesmo quando temos muito mais exemplos do que características,
redes neurais profundas são capazes de fazer *overfitting*.
Em 2017, um grupo de pesquisadores demonstrou
a extrema flexibilidade das redes neurais
treinando redes profundas em imagens rotuladas aleatoriamente.
Apesar da ausência de qualquer padrão verdadeiro
ligando as entradas às saídas,
eles descobriram que a rede neural otimizada pelo gradiente descendente estocástico
poderia rotular todas as imagens no conjunto de treinamento perfeitamente.
Considere o que isso significa.
Se os rótulos forem atribuídos uniformemente
aleatoriamente e há 10 classes,
então nenhum classificador pode fazer melhor
precisão de 10% nos dados de validação.
A lacuna de generalização aqui é de 90%.
Se nossos modelos são tão expressivos que
podem fazer tanto *overftitting*, então, quando deveríamos
esperar que eles não se ajustem demais?

Os fundamentos matemáticos para
as propriedades de generalização intrigantes
de redes profundas permanecem questões de pesquisa em aberto,
e encorajamos os leitores orientados teoricamente
para se aprofundar no assunto.
Por enquanto, nos voltamos para a investigação de
ferramentas práticas que tendem a
melhorar empiricamente a generalização de redes profundas.

## Robustez por Meio de Perturbações


Vamos pensar brevemente sobre o que nós
esperamos de um bom modelo preditivo.
Queremos que ele funcione bem com dados não vistos.
A teoria da generalização clássica
sugere que para fechar a lacuna entre
treinar e testar o desempenho,
devemos ter como objetivo um modelo simples.
A simplicidade pode vir na forma
de um pequeno número de dimensões.
Exploramos isso ao discutir as
funções de base monomial de modelos lineares
em :numref:`sec_model_selection`.
Além disso, como vimos ao discutir o *weight decay*
(regularização $L_2$) em :numref:`sec_weight_decay`,
a norma (inversa) dos parâmetros também
representa uma medida útil de simplicidade.
Outra noção útil de simplicidade é suavidade,
ou seja, que a função não deve ser sensível
a pequenas mudanças em suas entradas.
Por exemplo, quando classificamos imagens,
esperaríamos que adicionar algum ruído aleatório
aos pixels seja inofensivo.

Em 1995, Christopher Bishop formalizou
essa ideia quando ele provou que o treinamento com ruído de entrada
equivale à regularização de Tikhonov :cite:`Bishop.1995`.
Este trabalho traçou uma conexão matemática clara
entre o requisito de que uma função seja suave (e, portanto, simples),
e a exigência de que seja resiliente
a perturbações na entrada.

Então, em 2014, Srivastava et al. :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`
desenvolveram uma ideia inteligente de como aplicar a ideia de Bishop
às camadas internas de uma rede também.
Ou seja, eles propuseram injetar ruído
em cada camada da rede
antes de calcular a camada subsequente durante o treinamento.
Eles perceberam que durante o treinamento
uma rede profunda com muitas camadas,
injetando ruído reforça suavidade apenas no mapeamento de entrada-saída.


A ideia deles, chamada *dropout*, envolve
injetar ruído durante a computação de
cada camada interna durante a propagação direta,
e se tornou uma técnica padrão
para treinar redes neurais.
O método é chamado *dropout* porque nós literalmente
*abandonamos[^1]* alguns neurônios durante o treinamento.
Ao longo do treinamento, em cada iteração,
*dropout* padrão consiste em zerar
alguma fração dos nós em cada camada
antes de calcular a camada subsequente.

[^1]: A tradução do termo *drop out* do inglês pode ser interpretada, neste contexto, como abandonar, mas durante o texto, optou-se por usar o termo em inglês.

Para ser claro, estamos impondo
nossa própria narrativa com o link para Bishop.
O artigo original em *dropout*
oferece intuição através de uma surpreendente
analogia com a reprodução sexual.
Os autores argumentam que o *overfitting* da rede neural
é caracterizado por um estado em que
cada camada depende de um específico
padrão de ativações na camada anterior,
chamando essa condição de *co-adaptação*.
A desistência, eles afirmam, acaba com a co-adaptação
assim como a reprodução sexual é argumentada para
quebrar genes co-adaptados.

O principal desafio é como injetar esse ruído.
Uma ideia é injetar o ruído de uma maneira *imparcial*
de modo que o valor esperado de cada camada --- enquanto fixa
os outros --- seja igual ao valor que teria o ruído ausente.

No trabalho de Bishop, ele adicionou ruído gaussiano
às entradas de um modelo linear.
A cada iteração de treinamento, ele adicionava ruído
amostrado a partir de uma distribuição com média zero
$\epsilon \sim \mathcal{N}(0,\sigma^2)$ à entrada $\mathbf{x}$,
produzindo um ponto perturbado $\mathbf{x}' = \mathbf{x} + \epsilon$.
Na expectativa, $E[\mathbf{x}'] = \mathbf{x}$.

Na regularização de *dropout*  padrão,
um tira o *bias* de cada camada normalizando
pela fração de nós que foram retidos (não descartados).
Em outras palavras,
com *probabilidade de dropout* $p$,
cada ativação intermediária $h$ é substituída por
uma variável aleatória $h'$ como segue:

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ with probability } p \\
    \frac{h}{1-p} & \text{ otherwise}
\end{cases}
\end{aligned}
$$

By design, the expectation remains unchanged, i.e., $E[h'] = h$.

## Dropout in Practice

Recall the MLP with a hidden layer and 5 hidden units
in :numref:`fig_mlp`.
When we apply dropout to a hidden layer,
zeroing out each hidden unit with probability $p$,
the result can be viewed as a network
containing only a subset of the original neurons.
In :numref:`fig_dropout2`, $h_2$ and $h_5$ are removed.
Consequently, the calculation of the outputs
no longer depends on $h_2$ or $h_5$
and their respective gradient also vanishes
when performing backpropagation.
In this way, the calculation of the output layer
cannot be overly dependent on any
one element of $h_1, \ldots, h_5$.

![MLP before and after dropout.](../img/dropout2.svg)
:label:`fig_dropout2`

Typically, we disable dropout at test time.
Given a trained model and a new example,
we do not drop out any nodes
and thus do not need to normalize.
However, there are some exceptions:
some researchers use dropout at test time as a heuristic
for estimating the *uncertainty* of neural network predictions:
if the predictions agree across many different dropout masks,
then we might say that the network is more confident.

## Implementation from Scratch

To implement the dropout function for a single layer,
we must draw as many samples
from a Bernoulli (binary) random variable
as our layer has dimensions,
where the random variable takes value $1$ (keep)
with probability $1-p$ and $0$ (drop) with probability $p$.
One easy way to implement this is to first draw samples
from the uniform distribution $U[0, 1]$.
Then we can keep those nodes for which the corresponding
sample is greater than $p$, dropping the rest.

In the following code, we (**implement a `dropout_layer` function
that drops out the elements in the tensor input `X`
with probability `dropout`**),
rescaling the remainder as described above:
dividing the survivors by `1.0-dropout`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return np.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = (torch.Tensor(X.shape).uniform_(0, 1) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return tf.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

We can [**test out the `dropout_layer` function on a few examples**].
In the following lines of code,
we pass our input `X` through the dropout operation,
with probabilities 0, 0.5, and 1, respectively.

```{.python .input}
X = np.arange(16).reshape(2, 8)
print(dropout_layer(X, 0))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1))
```

```{.python .input}
#@tab pytorch
X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
print(X)
print(dropout_layer(X, 0.))
print(dropout_layer(X, 0.5))
print(dropout_layer(X, 1.))
```

### Defining Model Parameters

Again, we work with the Fashion-MNIST dataset
introduced in :numref:`sec_fashion_mnist`.
We [**define an MLP with
two hidden layers containing 256 units each.**]

```{.python .input}
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens1))
b1 = np.zeros(num_hiddens1)
W2 = np.random.normal(scale=0.01, size=(num_hiddens1, num_hiddens2))
b2 = np.zeros(num_hiddens2)
W3 = np.random.normal(scale=0.01, size=(num_hiddens2, num_outputs))
b3 = np.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
```

```{.python .input}
#@tab tensorflow
num_outputs, num_hiddens1, num_hiddens2 = 10, 256, 256
```

### Defining the Model

The model below applies dropout to the output
of each hidden layer (following the activation function).
We can set dropout probabilities for each layer separately.
A common trend is to set
a lower dropout probability closer to the input layer.
Below we set it to 0.2 and 0.5 for the first
and second hidden layers, respectively.
We ensure that dropout is only active during training.

```{.python .input}
dropout1, dropout2 = 0.2, 0.5

def net(X):
    X = X.reshape(-1, num_inputs)
    H1 = npx.relu(np.dot(X, W1) + b1)
    # Use dropout only when training the model
    if autograd.is_training():
        # Add a dropout layer after the first fully connected layer
        H1 = dropout_layer(H1, dropout1)
    H2 = npx.relu(np.dot(H1, W2) + b2)
    if autograd.is_training():
        # Add a dropout layer after the second fully connected layer
        H2 = dropout_layer(H2, dropout2)
    return np.dot(H2, W3) + b3
```

```{.python .input}
#@tab pytorch
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # Use dropout only when training the model
        if self.training == True:
            # Add a dropout layer after the first fully connected layer
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # Add a dropout layer after the second fully connected layer
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

```{.python .input}
#@tab tensorflow
dropout1, dropout2 = 0.2, 0.5

class Net(tf.keras.Model):
    def __init__(self, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()
        self.input_layer = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(num_hiddens1, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(num_hiddens2, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, training=None):
        x = self.input_layer(inputs)
        x = self.hidden1(x)
        if training:
            x = dropout_layer(x, dropout1)
        x = self.hidden2(x)
        if training:
            x = dropout_layer(x, dropout2)
        x = self.output_layer(x)
        return x

net = Net(num_outputs, num_hiddens1, num_hiddens2)
```

### [**Training and Testing**]

This is similar to the training and testing of MLPs described previously.

```{.python .input}
num_epochs, lr, batch_size = 10, 0.5, 256
loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr, batch_size = 10, 0.5, 256
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## [**Concise Implementation**]

With high-level APIs, all we need to do is add a `Dropout` layer
after each fully-connected layer,
passing in the dropout probability
as the only argument to its constructor.
During training, the `Dropout` layer will randomly
drop out outputs of the previous layer
(or equivalently, the inputs to the subsequent layer)
according to the specified dropout probability.
When not in training mode,
the `Dropout` layer simply passes the data through during testing.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Dense(256, activation="relu"),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # Add a dropout layer after the first fully connected layer
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # Add a dropout layer after the second fully connected layer
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the first fully connected layer
    tf.keras.layers.Dropout(dropout1),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # Add a dropout layer after the second fully connected layer
    tf.keras.layers.Dropout(dropout2),
    tf.keras.layers.Dense(10),
])
```

Next, we [**train and test the model**].

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## Summary

* Beyond controlling the number of dimensions and the size of the weight vector, dropout is yet another tool to avoid overfitting. Often they are used jointly.
* Dropout replaces an activation $h$ with a random variable with expected value $h$.
* Dropout is only used during training.


## Exercises

1. What happens if you change the dropout probabilities for the first and second layers? In particular, what happens if you switch the ones for both layers? Design an experiment to answer these questions, describe your results quantitatively, and summarize the qualitative takeaways.
1. Increase the number of epochs and compare the results obtained when using dropout with those when not using it.
1. What is the variance of the activations in each hidden layer when dropout is and is not applied? Draw a plot to show how this quantity evolves over time for both models.
1. Why is dropout not typically used at test time?
1. Using the model in this section as an example, compare the effects of using dropout and weight decay. What happens when dropout and weight decay are used at the same time? Are the results additive? Are there diminished returns (or worse)? Do they cancel each other out?
1. What happens if we apply dropout to the individual weights of the weight matrix rather than the activations?
1. Invent another technique for injecting random noise at each layer that is different from the standard dropout technique. Can you develop a method that outperforms dropout on the Fashion-MNIST dataset (for a fixed architecture)?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/261)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTkzMTI0NjQ5NiwtMTgyODYwODg5OSwyNj
E1MDQzNjgsNjg1MDk2ODA4LC0xNzM5MTE4NDEwLDEzODAxNzg1
NTAsLTk4MDg3NzM1N119
-->
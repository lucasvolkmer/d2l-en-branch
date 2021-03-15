# Implementação de Perceptrons Multicamadas do Zero
:label:`sec_mlp_scratch`

Agora que caracterizamos
perceptrons multicamadas (MLPs) matematicamente,
vamos tentar implementar um nós mesmos. Para comparar com nossos resultados anteriores
alcançado com regressão *softmax*
(:numref:`sec_softmax_scratch`),
vamos continuar a trabalhar com
o conjunto de dados de classificação de imagens Fashion-MNIST
(:numref:`sec_fashion_mnist`).

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Inicializando os Parâmetros do Modelo


Lembre-se de que o Fashion-MNIST contém 10 classes,
e que cada imagem consiste em uma graded $28 \times 28 = 784$ de valores de pixel em tons de cinza.
Novamente, vamos desconsiderar a estrutura espacial
entre os pixels por enquanto,
então podemos pensar nisso simplesmente como um conjunto de dados de classificação
com 784 características de entrada e 10 classes.
Para começar, iremos [**implementar um MLP
com uma camada oculta e 256 unidades ocultas.**]
Observe que podemos considerar essas duas quantidades
como hiperparâmetros.
Normalmente, escolhemos larguras de camada em potências de 2,
que tendem a ser computacionalmente eficientes porque
de como a memória é alocada e endereçada no hardware.

Novamente, iremos representar nossos parâmetros com vários tensores.
Observe que *para cada camada*, devemos acompanhar
uma matriz de ponderação e um vetor de polarização.
Como sempre, alocamos memória
para os gradientes da perda com relação a esses parâmetros.

```{.python .input}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab tensorflow
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

params = [W1, b1, W2, b2]
```

## Função de Ativação

Para ter certeza de que sabemos como tudo funciona,
iremos [**implementar a ativação ReLU**] nós mesmos
usar a função máxima em vez de
invocar a função embutida `relu` diretamente.

```{.python .input}
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
#@tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
#@tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

## Modelo

Porque estamos desconsiderando a estrutura espacial,
nós `remodelamos` cada imagem bidimensional em
um vetor plano de comprimento `num_inputs`.
Finalmente, nós (**implementamos nosso modelo**)
com apenas algumas linhas de código.

```{.python .input}
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

```{.python .input}
#@tab pytorch
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # Here '@' stands for matrix multiplication
    return (H@W2 + b2)
```

```{.python .input}
#@tab tensorflow
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
```

## Função de Perda

To ensure numerical stability,
and because we already implemented
the softmax function from scratch
(:numref:`sec_softmax_scratch`),
we leverage the integrated function from high-level APIs
for calculating the softmax and cross-entropy loss.
Recall our earlier discussion of these intricacies
in :numref:`subsec_softmax-implementation-revisited`.
We encourage the interested reader
to examine the source code for the loss function
to deepen their knowledge of implementation details.

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
```

## Training

Fortunately, [**the training loop for MLPs
is exactly the same as for softmax regression.**]
Leveraging the `d2l` package again,
we call the `train_ch3` function
(see :numref:`sec_softmax_scratch`),
setting the number of epochs to 10
and the learning rate to 0.1.

```{.python .input}
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 10, 0.1
updater = d2l.Updater([W1, W2, b1, b2], lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

To evaluate the learned model,
we [**apply it on some test data**].

```{.python .input}
#@tab all
d2l.predict_ch3(net, test_iter)
```

## Summary

* We saw that implementing a simple MLP is easy, even when done manually.
* However, with a large number of layers, implementing MLPs from scratch can still get messy (e.g., naming and keeping track of our model's parameters).


## Exercises

1. Change the value of the hyperparameter `num_hiddens` and see how this hyperparameter influences your results. Determine the best value of this hyperparameter, keeping all others constant.
1. Try adding an additional hidden layer to see how it affects the results.
1. How does changing the learning rate alter your results? Fixing the model architecture and other hyperparameters (including number of epochs), what learning rate gives you the best results?
1. What is the best result you can get by optimizing over all the hyperparameters (learning rate, number of epochs, number of hidden layers, number of hidden units per layer) jointly?
1. Describe why it is much more challenging to deal with multiple hyperparameters.
1. What is the smartest strategy you can think of for structuring a search over multiple hyperparameters?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbOTUzMzcyMDExLC0xNjk4NDU4MTQyLC0xMz
kzNDY2NjcyLDE1MjkzMTcxNzRdfQ==
-->
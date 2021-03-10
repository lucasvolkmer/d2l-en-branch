# Implementação Concisa de Regressão Linear
:label:`sec_linear_concise`


Amplo e intenso interesse em *deep learning* nos últimos anos
inspiraram empresas, acadêmicos e amadores
para desenvolver uma variedade de estruturas de código aberto maduras
para automatizar o trabalho repetitivo de implementação
algoritmos de aprendizagem baseados em gradiente.
Em :numref:`sec_linear_scratch`, contamos apenas com
(i) tensores para armazenamento de dados e álgebra linear;
e (ii) auto diferenciação para cálculo de gradientes.
Na prática, porque iteradores de dados, funções de perda, otimizadores,
e camadas de rede neural
são tão comuns que as bibliotecas modernas também implementam esses componentes para nós.

Nesta seção, (**mostraremos como implementar
o modelo de regressão linear**) de:numref:`sec_linear_scratch`
(**de forma concisa, usando APIs de alto nível**) de estruturas de *deep learning*.


## Gerando the Dataset

Para começar, vamos gerar o mesmo conjunto de dados como em
:numref:`sec_linear_scratch`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## Lendo o Dataset

Em vez de usar nosso próprio iterador,
podemos [**chamar a API existente em uma estrutura para ler os dados.**]
Passamos *`features`* e *`labels`* como argumentos e especificamos *`batch_size`*
ao instanciar um objeto iterador de dados.
Além disso, o valor booleano `is_train`
indica se ou não
queremos que o objeto iterador de dados embaralhe os dados
em cada época (passe pelo conjunto de dados).

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

```{.python .input}
#@tab all
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

Now we can use `data_iter` in much the same way as we called
the `data_iter` function in :numref:`sec_linear_scratch`.
To verify that it is working, we can read and print
the first minibatch of examples.
Comparing with :numref:`sec_linear_scratch`,
here we use `iter` to construct a Python iterator and use `next` to obtain the first item from the iterator.

```{.python .input}
#@tab all
next(iter(data_iter))
```

## Definindo o Modelo


Quando implementamos a regressão linear do zero
em :numref:`sec_linear_scratch`,
definimos nossos parâmetros de modelo explicitamente
e codificamos os cálculos para produzir saída
usando operações básicas de álgebra linear.
Você *deveria* saber como fazer isso.
Mas quando seus modelos ficam mais complexos,
e uma vez que você tem que fazer isso quase todos os dias,
você ficará feliz com a ajuda.
A situação é semelhante a codificar seu próprio blog do zero.
Fazer uma ou duas vezes é gratificante e instrutivo,
mas você seria um péssimo desenvolvedor da web
se toda vez que você precisava de um blog você passava um mês
reinventando tudo.

Para operações padrão, podemos [**usar as camadas predefinidas de uma estrutura,**]
o que nos permite focar especialmente
nas camadas usadas para construir o modelo
em vez de ter que se concentrar na implementação.
Vamos primeiro definir uma variável de modelo `net`,
que se refere a uma instância da classe `Sequential`.
A classe `Sequential` define um contêiner
para várias camadas que serão encadeadas.
Dados dados de entrada, uma instância `Sequential` passa por
a primeira camada, por sua vez passando a saída
como entrada da segunda camada e assim por diante.
No exemplo a seguir, nosso modelo consiste em apenas uma camada,
portanto, não precisamos realmente de `Sequencial`.
Mas como quase todos os nossos modelos futuros
envolverão várias camadas,
vamos usá-lo de qualquer maneira apenas para familiarizá-lo
com o fluxo de trabalho mais padrão.

Lembre-se da arquitetura de uma rede de camada única, conforme mostrado em :numref:`fig_single_neuron`.
Diz-se que a camada está *totalmente conectada*
porque cada uma de suas entradas está conectada a cada uma de suas saídas
por meio de uma multiplicação de matriz-vetor.

:begin_tab:`mxnet`
No Gluon, a camada totalmente conectada é definida na classe `Densa`.
Uma vez que queremos apenas gerar uma única saída escalar,
nós definimos esse número para 1.

É importante notar que, por conveniência,
Gluon não exige que especifiquemos
a forma de entrada para cada camada.
Então, aqui, não precisamos dizer ao Gluon
quantas entradas vão para esta camada linear.
Quando tentamos primeiro passar dados por meio de nosso modelo,
por exemplo, quando executamos `net (X)` mais tarde,
o Gluon irá inferir automaticamente o número de entradas para cada camada.
Descreveremos como isso funciona com mais detalhes posteriormente.
:end_tab:

: begin_tab: `pytorch`
No PyTorch, a camada totalmente conectada é definida na classe `Linear`. Observe que passamos dois argumentos para `nn.Linear`. O primeiro especifica a dimensão do recurso de entrada, que é 2, e o segundo é a dimensão do recurso de saída, que é um escalar único e, portanto, 1.
:end_tab:

:begin_tab:`tensorflow`
No Keras, a camada totalmente conectada é definida na classe `Dense`. Como queremos gerar apenas uma única saída escalar, definimos esse número como 1.

É importante notar que, por conveniência,
Keras não exige que especifiquemos
a forma de entrada para cada camada.
Então, aqui, não precisamos dizer a Keras
quantas entradas vão para esta camada linear.
Quando tentamos primeiro passar dados por meio de nosso modelo,
por exemplo, quando executamos `net (X)` mais tarde,
Keras inferirá automaticamente o número de entradas para cada camada.
Descreveremos como isso funciona com mais detalhes posteriormente.
:end_tab:

```{.python .input}
# `nn` is an abbreviation for neural networks
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# `nn` is an abbreviation for neural networks
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# `keras` is the high-level API for TensorFlow
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

## Inicializando os Parâmetros do Modelo


Antes de usar `net`, precisamos (**inicializar os parâmetros do modelo,**)
como os pesos e *bias* no modelo de regressão linear.
As estruturas de *deep learning* geralmente têm uma maneira predefinida de inicializar os parâmetros.
Aqui especificamos que cada parâmetro de peso
deve ser amostrado aleatoriamente a partir de uma distribuição normal
com média 0 e desvio padrão 0,01.
O parâmetro bias será inicializado em zero.

:begin_tab:`mxnet`
Vamos importar o módulo *`initializer`* do MXNet.
Este módulo fornece vários métodos para inicialização de parâmetros do modelo.
Gluon disponibiliza `init` como um atalho (abreviatura)
para acessar o pacote `initializer`.
Nós apenas especificamos como inicializar o peso chamando `init.Normal (sigma = 0,01)`.
Os parâmetros de polarização são inicializados em zero por padrão.
:end_tab:

:begin_tab:`pytorch`
As we have specified the input and output dimensions when constructing `nn.Linear`. Now we access the parameters directly to specify their initial values. We first locate the layer by `net[0]`, which is the first layer in the network, and then use the `weight.data` and `bias.data` methods to access the parameters. Next we use the replace methods `normal_` and `fill_` to overwrite parameter values.
:end_tab:

:begin_tab:`tensorflow`
O módulo *`initializers`* no TensorFlow fornece vários métodos para a inicialização dos parâmetros do modelo. A maneira mais fácil de especificar o método de inicialização no Keras é ao criar a camada especificando *`kernel_initializer`*. Aqui, recriamos o `net` novamente.
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

:begin_tab:`mxnet`
The code above may look straightforward but you should note
that something strange is happening here.
We are initializing parameters for a network
even though Gluon does not yet know
how many dimensions the input will have!
It might be 2 as in our example or it might be 2000.
Gluon lets us get away with this because behind the scene,
the initialization is actually *deferred*.
The real initialization will take place only
when we for the first time attempt to pass data through the network.
Just be careful to remember that since the parameters
have not been initialized yet,
we cannot access or manipulate them.
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
The code above may look straightforward but you should note
that something strange is happening here.
We are initializing parameters for a network
even though Keras does not yet know
how many dimensions the input will have!
It might be 2 as in our example or it might be 2000.
Keras lets us get away with this because behind the scenes,
the initialization is actually *deferred*.
The real initialization will take place only
when we for the first time attempt to pass data through the network.
Just be careful to remember that since the parameters
have not been initialized yet,
we cannot access or manipulate them.
:end_tab:

## Defining the Loss Function

:begin_tab:`mxnet`
In Gluon, the `loss` module defines various loss functions.
In this example, we will use the Gluon
implementation of squared loss (`L2Loss`).
:end_tab:

:begin_tab:`pytorch`
[**The `MSELoss` class computes the mean squared error, also known as squared $L_2$ norm.**]
By default it returns the average loss over examples.
:end_tab:

:begin_tab:`tensorflow`
The `MeanSquaredError` class computes the mean squared error, also known as squared $L_2$ norm.
By default it returns the average loss over examples.
:end_tab:

```{.python .input}
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()
```

## Defining the Optimization Algorithm

:begin_tab:`mxnet`
Minibatch stochastic gradient descent is a standard tool
for optimizing neural networks
and thus Gluon supports it alongside a number of
variations on this algorithm through its `Trainer` class.
When we instantiate `Trainer`,
we will specify the parameters to optimize over
(obtainable from our model `net` via `net.collect_params()`),
the optimization algorithm we wish to use (`sgd`),
and a dictionary of hyperparameters
required by our optimization algorithm.
Minibatch stochastic gradient descent just requires that
we set the value `learning_rate`, which is set to 0.03 here.
:end_tab:

:begin_tab:`pytorch`
Minibatch stochastic gradient descent is a standard tool
for optimizing neural networks
and thus PyTorch supports it alongside a number of
variations on this algorithm in the `optim` module.
When we (**instantiate an `SGD` instance,**)
we will specify the parameters to optimize over
(obtainable from our net via `net.parameters()`), with a dictionary of hyperparameters
required by our optimization algorithm.
Minibatch stochastic gradient descent just requires that
we set the value `lr`, which is set to 0.03 here.
:end_tab:

:begin_tab:`tensorflow`
Minibatch stochastic gradient descent is a standard tool
for optimizing neural networks
and thus Keras supports it alongside a number of
variations on this algorithm in the `optimizers` module.
Minibatch stochastic gradient descent just requires that
we set the value `learning_rate`, which is set to 0.03 here.
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)
```

## Training

You might have noticed that expressing our model through
high-level APIs of a deep learning framework
requires comparatively few lines of code.
We did not have to individually allocate parameters,
define our loss function, or implement minibatch stochastic gradient descent.
Once we start working with much more complex models,
advantages of high-level APIs will grow considerably.
However, once we have all the basic pieces in place,
[**the training loop itself is strikingly similar
to what we did when implementing everything from scratch.**]

To refresh your memory: for some number of epochs,
we will make a complete pass over the dataset (`train_data`),
iteratively grabbing one minibatch of inputs
and the corresponding ground-truth labels.
For each minibatch, we go through the following ritual:

* Generate predictions by calling `net(X)` and calculate the loss `l` (the forward propagation).
* Calculate gradients by running the backpropagation.
* Update the model parameters by invoking our optimizer.

For good measure, we compute the loss after each epoch and print it to monitor progress.

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab tensorflow
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

Below, we [**compare the model parameters learned by training on finite data
and the actual parameters**] that generated our dataset.
To access parameters,
we first access the layer that we need from `net`
and then access that layer's weights and bias.
As in our from-scratch implementation,
note that our estimated parameters are
close to their ground-truth counterparts.

```{.python .input}
w = net[0].weight.data()
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'error in estimating b: {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('error in estimating w:', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('error in estimating w', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('error in estimating b', true_b - b)
```

## Summary

:begin_tab:`mxnet`
* Using Gluon, we can implement models much more concisely.
* In Gluon, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers, and the `loss` module defines many common loss functions.
* MXNet's module `initializer` provides various methods for model parameter initialization.
* Dimensionality and storage are automatically inferred, but be careful not to attempt to access parameters before they have been initialized.
:end_tab:

:begin_tab:`pytorch`
* Using PyTorch's high-level APIs, we can implement models much more concisely.
* In PyTorch, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers and common loss functions.
* We can initialize the parameters by replacing their values with methods ending with `_`.
:end_tab:

:begin_tab:`tensorflow`
* Using TensorFlow's high-level APIs, we can implement models much more concisely.
* In TensorFlow, the `data` module provides tools for data processing, the `keras` module defines a large number of neural network layers and common loss functions.
* TensorFlow's module `initializers` provides various methods for model parameter initialization.
* Dimensionality and storage are automatically inferred (but be careful not to attempt to access parameters before they have been initialized).
:end_tab:

## Exercises

:begin_tab:`mxnet`
1. If we replace `l = loss(output, y)` with `l = loss(output, y).mean()`, we need to change `trainer.step(batch_size)` to `trainer.step(1)` for the code to behave identically. Why?
1. Review the MXNet documentation to see what loss functions and initialization methods are provided in the modules `gluon.loss` and `init`. Replace the loss by Huber's loss.
1. How do you access the gradient of `dense.weight`?

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. If we replace `nn.MSELoss(reduction='sum')` with `nn.MSELoss()`, how can we change the learning rate for the code to behave identically. Why?
1. Review the PyTorch documentation to see what loss functions and initialization methods are provided. Replace the loss by Huber's loss.
1. How do you access the gradient of `net[0].weight`?

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. Review the TensorFlow documentation to see what loss functions and initialization methods are provided. Replace the loss by Huber's loss.

[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMzYzNjY2MzMsLTY1Mjk5MTk1OCwtMjE0NT
k5NTMwN119
-->
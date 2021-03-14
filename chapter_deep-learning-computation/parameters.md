# Gerenciamento de Parâmetros

Depois de escolher uma arquitetura
e definir nossos hiperparâmetros,
passamos para o ciclo de treinamento,
onde nosso objetivo é encontrar valores de parâmetro
que minimizam nossa função de perda.
Após o treinamento, precisaremos desses parâmetros
para fazer previsões futuras.
Além disso, às vezes desejamos
para extrair os parâmetros
seja para reutilizá-los em algum outro contexto,
para salvar nosso modelo em disco para que
pode ser executado em outro *software*,
ou para exame na esperança de
ganhar compreensão científica.

Na maioria das vezes, seremos capazes de
ignorar os detalhes essenciais
de como os parâmetros são declarados
e manipulado, contando com estruturas de *Deep Learning*
para fazer o trabalho pesado.
No entanto, quando nos afastamos de
arquiteturas empilhadas com camadas padrão,
às vezes precisaremos 
declarar e manipular parâmetros.
Nesta seção, cobrimos o seguinte:

* Parâmetros de acesso para depuração, diagnóstico e visualizações.
* Inicialização de parâmetros.
* Parâmetros de compartilhamento em diferentes componentes do modelo.

Começamos nos concentrando em um MLP com uma camada oculta.

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X)  # Forward computation
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X)
```

## Acesso a Parâmetros

Vamos começar explicando como acessar os parâmetros
dos modelos que você já conhece.
Quando um modelo é definido por meio da classe `Sequential`,
podemos primeiro acessar qualquer camada indexando
no modelo como se fosse uma lista.
Os parâmetros de cada camada são convenientemente
localizado em seu atributo.
Podemos inspecionar os parâmetros da segunda camada totalmente conectada da seguinte maneira.

```{.python .input}
print(net[1].params)
```

```{.python .input}
#@tab pytorch
print(net[2].state_dict())
```

```{.python .input}
#@tab tensorflow
print(net.layers[2].weights)
```

A saída nos diz algumas coisas importantes.
Primeiro, esta camada totalmente conectada
contém dois parâmetros,
correspondendo aos
pesos e vieses, respectivamente.
Ambos são armazenados como *floats* de precisão simples (float32).
Observe que os nomes dos parâmetros
nos permitem identificar de forma única
parâmetros de cada camada,
mesmo em uma rede contendo centenas de camadas.


### Parâmetros Direcionados

Note that each parameter is represented
as an instance of the parameter class.
To do anything useful with the parameters,
we first need to access the underlying numerical values.
There are several ways to do this.
Some are simpler while others are more general.
The following code extracts the bias
from the second neural network layer, which returns a parameter class instance, and 
further accesses that parameter's value.

Observe que cada parâmetro é representado
como uma instância da classe de parâmetro.
Para fazer algo útil com os parâmetros,
primeiro precisamos acessar os valores numéricos subjacentes.
Existem várias maneiras de fazer isso.
Alguns são mais simples, enquanto outros são mais gerais.
O código a seguir extrai o viés
da segunda camada de rede neural, que retorna uma instância de classe de parâmetro, e
acessa posteriormente o valor desse parâmetro.

```{.python .input}
print(type(net[1].bias))
print(net[1].bias)
print(net[1].bias.data())
```

```{.python .input}
#@tab pytorch
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

```{.python .input}
#@tab tensorflow
print(type(net.layers[2].weights[1]))
print(net.layers[2].weights[1])
print(tf.convert_to_tensor(net.layers[2].weights[1]))
```

:begin_tab:`mxnet,pytorch`
Parameters are complex objects,
containing values, gradients,
and additional information.
That's why we need to request the value explicitly.

In addition to the value, each parameter also allows us to access the gradient. Because we have not invoked backpropagation for this network yet, it is in its initial state.

Os parâmetros são objetos complexos,
contendo valores, gradientes,
e informações adicionais.
É por isso que precisamos solicitar o valor explicitamente.

Além do valor, cada parâmetro também nos permite acessar o gradiente. Como ainda não invocamos a retropropagação para esta rede, ela está em seu estado inicial.
:end_tab:

```{.python .input}
net[1].weight.grad()
```

```{.python .input}
#@tab pytorch
net[2].weight.grad == None
```

### All Parameters at Once

When we need to perform operations on all parameters,
accessing them one-by-one can grow tedious.
The situation can grow especially unwieldy
when we work with more complex blocks (e.g., nested blocks),
since we would need to recurse
through the entire tree to extract
each sub-block's parameters. Below we demonstrate accessing the parameters of the first fully-connected layer vs. accessing all layers.

Quando precisamos realizar operações em todos os parâmetros,
acessá-los um por um pode se tornar tedioso.
A situação pode ficar especialmente complicada
quando trabalhamos com blocos mais complexos (por exemplo, blocos aninhados),
uma vez que precisaríamos recurse
através de toda a árvore para extrair
parâmetros de cada sub-bloco. Abaixo, demonstramos como acessar os parâmetros da primeira camada totalmente conectada versus acessar todas as camadas.

```{.python .input}
print(net[0].collect_params())
print(net.collect_params())
```

```{.python .input}
#@tab pytorch
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

```{.python .input}
#@tab tensorflow
print(net.layers[1].weights)
print(net.get_weights())
```

This provides us with another way of accessing the parameters of the network as follows.

Isso nos fornece outra maneira de acessar os parâmetros da rede como segue.

```{.python .input}
net.collect_params()['dense1_bias'].data()
```

```{.python .input}
#@tab pytorch
net.state_dict()['2.bias'].data
```

```{.python .input}
#@tab tensorflow
net.get_weights()[1]
```

### Collecting Parameters from Nested Blocks

Let us see how the parameter naming conventions work
if we nest multiple blocks inside each other.
For that we first define a function that produces blocks
(a block factory, so to speak) and then
combine these inside yet larger blocks.

Vamos ver como funcionam as convenções de nomenclatura de parâmetros
se aninharmos vários blocos uns dentro dos outros.
Para isso, primeiro definimos uma função que produz blocos
(uma fábrica de blocos, por assim dizer) e então
combine-os dentro de blocos ainda maiores.

```{.python .input}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for _ in range(4):
        # Nested here
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(X)
```

```{.python .input}
#@tab pytorch
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # Nested here
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```

```{.python .input}
#@tab tensorflow
def block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu)],
        name=name)

def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        # Nested here
        net.add(block1(name=f'block-{i}'))
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(1))
rgnet(X)
```

Now that we have designed the network,
let us see how it is organized.

Agora que projetamos a rede,
vamos ver como está organizado.

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

```{.python .input}
#@tab pytorch
print(rgnet)
```

```{.python .input}
#@tab tensorflow
print(rgnet.summary())
```

Since the layers are hierarchically nested,
we can also access them as though
indexing through nested lists.
For instance, we can access the first major block,
within it the second sub-block,
and within that the bias of the first layer,
with as follows.

Uma vez que as camadas são aninhadas hierarquicamente,
também podemos acessá-los como se
indexação por meio de listas aninhadas.
Por exemplo, podemos acessar o primeiro bloco principal,
dentro dele o segundo sub-bloco,
e dentro disso o viés da primeira camada,
com o seguinte.

```{.python .input}
rgnet[0][1][0].bias.data()
```

```{.python .input}
#@tab pytorch
rgnet[0][1][0].bias.data
```

```{.python .input}
#@tab tensorflow
rgnet.layers[0].layers[1].layers[1].weights[1]
```

## Parameter Initialization

Now that we know how to access the parameters,
let us look at how to initialize them properly.
We discussed the need for proper initialization in :numref:`sec_numerical_stability`.
The deep learning framework provides default random initializations to its layers.
However, we often want to initialize our weights
according to various other protocols. The framework provides most commonly
used protocols, and also allows to create a custom initializer.

Agora que sabemos como acessar os parâmetros,
vamos ver como inicializá-los corretamente.
Discutimos a necessidade de inicialização adequada em: numref: `sec_numerical_stability`.
A estrutura de aprendizado profundo fornece inicializações aleatórias padrão para suas camadas.
No entanto, muitas vezes queremos inicializar nossos pesos
de acordo com vários outros protocolos. A estrutura fornece mais comumente
protocolos usados e também permite criar um inicializador personalizado.

:begin_tab:`mxnet`
By default, MXNet initializes weight parameters by randomly drawing from a uniform distribution $U(-0.07, 0.07)$,
clearing bias parameters to zero.
MXNet's `init` module provides a variety
of preset initialization methods.

Por padrão, MXNet inicializa os parâmetros de peso ao desenhar aleatoriamente de uma distribuição uniforme $U(-0.07, 0.07)$,
limpar os parâmetros de polarização para zero.
O módulo `init` do MXNet oferece uma variedade
de métodos de inicialização predefinidos.
:end_tab:

:begin_tab:`pytorch`
By default, PyTorch initializes weight and bias matrices
uniformly by drawing from a range that is computed according to the input and output dimension.
PyTorch's `nn.init` module provides a variety
of preset initialization methods.

Por padrão, o PyTorch inicializa matrizes de ponderação e polarização
uniformemente extraindo de um intervalo que é calculado de acordo com a dimensão de entrada e saída.
O módulo `nn.init` do PyTorch oferece uma variedade
de métodos de inicialização predefinidos.
:end_tab:

:begin_tab:`tensorflow`
By default, Keras initializes weight matrices uniformly by drawing from a range that is computed according to the input and output dimension, and the bias parameters are all set to zero.
TensorFlow provides a variety of initialization methods both in the root module and the `keras.initializers` module.

Por padrão, Keras inicializa matrizes de ponderação uniformemente, tirando de um intervalo que é calculado de acordo com a dimensão de entrada e saída, e os parâmetros de polarização são todos definidos como zero.
O TensorFlow oferece uma variedade de métodos de inicialização no módulo raiz e no módulo `keras.initializers`.
:end_tab:

### Built-in Initialization

Let us begin by calling on built-in initializers.
The code below initializes all weight parameters
as Gaussian random variables
with standard deviation 0.01, while bias parameters cleared to zero.

Vamos começar chamando inicializadores integrados.
O código abaixo inicializa todos os parâmetros de peso
como variáveis aleatórias gaussianas
com desvio padrão de 0,01, enquanto os parâmetros de polarização são zerados.

```{.python .input}
# Here `force_reinit` ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

We can also initialize all the parameters
to a given constant value (say, 1).

Também podemos inicializar todos os parâmetros
a um determinado valor constante (digamos, 1).

```{.python .input}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

We can also apply different initializers for certain blocks.
For example, below we initialize the first layer
with the Xavier initializer
and initialize the second layer
to a constant value of 42.

Também podemos aplicar inicializadores diferentes para certos blocos.
Por exemplo, abaixo inicializamos a primeira camada
com o inicializador Xavier
e inicializar a segunda camada
para um valor constante de 42.

```{.python .input}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
#@tab pytorch
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(1)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

### Custom Initialization

Sometimes, the initialization methods we need
are not provided by the deep learning framework.
In the example below, we define an initializer
for any weight parameter $w$ using the following strange distribution:

Às vezes, os métodos de inicialização de que precisamos
não são fornecidos pela estrutura de aprendizado profundo.
No exemplo abaixo, definimos um inicializador
para qualquer parâmetro de peso $ w $ usando a seguinte distribuição estranha:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ with probability } \frac{1}{4} \\
            0    & \text{ with probability } \frac{1}{2} \\
        U(-10, -5) & \text{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Here we define a subclass of the `Initializer` class.
Usually, we only need to implement the `_init_weight` function
which takes a tensor argument (`data`)
and assigns to it the desired initialized values.

Aqui definimos uma subclasse da classe `Initializer`.
Normalmente, só precisamos implementar a função `_init_weight`
que leva um argumento tensor (`dados`)
e atribui a ele os valores inicializados desejados.
:end_tab:

:begin_tab:`pytorch`
Again, we implement a `my_init` function to apply to `net`.

Novamente, implementamos uma função `my_init` para aplicar a` net`.
:end_tab:

:begin_tab:`tensorflow`
Here we define a subclass of `Initializer` and implement the `__call__`
function that return a desired tensor given the shape and data type.

Aqui nós definimos uma subclasse de `Initializer` e implementamos o` __call__`
função que retorna um tensor desejado de acordo com a forma e o tipo de dados.
:end_tab:

```{.python .input}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
#@tab pytorch
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) 
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
#@tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor        

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

Note that we always have the option
of setting parameters directly.

Observe que sempre temos a opção
de definir parâmetros diretamente.

```{.python .input}
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
#@tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
#@tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

:begin_tab:`mxnet`
A note for advanced users:
if you want to adjust parameters within an `autograd` scope,
you need to use `set_data` to avoid confusing
the automatic differentiation mechanics.

Uma observação para usuários avançados:
se você quiser ajustar os parâmetros dentro de um escopo `autograd`,
você precisa usar `set_data` para evitar confundir
a mecânica de diferenciação automática.
:end_tab:

## Tied Parameters

Often, we want to share parameters across multiple layers.
Let us see how to do this elegantly.
In the following we allocate a dense layer
and then use its parameters specifically
to set those of another layer.

Freqüentemente, queremos compartilhar parâmetros em várias camadas.
Vamos ver como fazer isso com elegância.
A seguir, alocamos uma camada densa
e usar seus parâmetros especificamente
para definir os de outra camada.

```{.python .input}
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
#@tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
#@tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

:begin_tab:`mxnet,pytorch`
This example shows that the parameters
of the second and third layer are tied.
They are not just equal, they are
represented by the same exact tensor.
Thus, if we change one of the parameters,
the other one changes, too.
You might wonder,
when parameters are tied
what happens to the gradients?
Since the model parameters contain gradients,
the gradients of the second hidden layer
and the third hidden layer are added together
during backpropagation.

Este exemplo mostra que os parâmetros
da segunda e terceira camadas são amarrados.
Eles não são apenas iguais, eles são
representado pelo mesmo tensor exato.
Assim, se mudarmos um dos parâmetros,
o outro também muda.
Você pode se perguntar,
quando os parâmetros são amarrados
o que acontece com os gradientes?
Uma vez que os parâmetros do modelo contêm gradientes,
os gradientes da segunda camada oculta
e a terceira camada oculta são adicionadas juntas
durante a retropropagação.
:end_tab:

## Summary

* We have several ways to access, initialize, and tie model parameters.
* We can use custom initialization.

* Temos várias maneiras de acessar, inicializar e vincular os parâmetros do modelo.
* Podemos usar inicialização personalizada.


## Exercises

1. Use the `FancyMLP` model defined in :numref:`sec_model_construction` and access the parameters of the various layers.
1. Look at the initialization module document to explore different initializers.
1. Construct an MLP containing a shared parameter layer and train it. During the training process, observe the model parameters and gradients of each layer.
1. Why is sharing parameters a good idea?


1. Use o modelo `FancyMLP` definido em: numref:` sec_model_construction` e acesse os parâmetros das várias camadas.
1. Observe o documento do módulo de inicialização para explorar diferentes inicializadores.
1. Construa um MLP contendo uma camada de parâmetros compartilhados e treine-o. Durante o processo de treinamento, observe os parâmetros do modelo e gradientes de cada camada.
1. Por que compartilhar parâmetros é uma boa ideia?
:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbOTUyODk0MTc2XX0=
-->
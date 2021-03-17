# Camadas Personalizadas

One factor behind deep learning's success
is the availability of a wide range of layers
that can be composed in creative ways
to design architectures suitable
for a wide variety of tasks.
For instance, researchers have invented layers
specifically for handling images, text,
looping over sequential data,
and
performing dynamic programming.
Sooner or later, you will encounter or invent
a layer that does not exist yet in the deep learning framework.
In these cases, you must build a custom layer.
In this section, we show you how.

Um fator por trás do sucesso do *Deep Learning*
é a disponibilidade de uma ampla gama de camadas
que pode ser composto de maneiras criativas
projetar arquiteturas adequadas
para uma ampla variedade de tarefas.
Por exemplo, os pesquisadores inventaram camadas
especificamente para lidar com imagens, texto,
loop sobre dados sequenciais,
e
realizando programação dinâmica.
Mais cedo ou mais tarde, você encontrará ou inventará
uma camada que ainda não existe na estrutura de *Deep Learning*.
Nesses casos, você deve construir uma camada personalizada.
Nesta seção, mostramos como.

## Layers without Parameters

To start, we construct a custom layer
that does not have any parameters of its own.
This should look familiar if you recall our
introduction to block in :numref:`sec_model_construction`.
The following `CenteredLayer` class simply
subtracts the mean from its input.
To build it, we simply need to inherit
from the base layer class and implement the forward propagation function.

Para começar, construímos uma camada personalizada
que não possui parâmetros próprios.
Isso deve parecer familiar, se você se lembra de nosso
introdução ao bloco em: numref: `sec_model_construction`.
A seguinte classe `CenteredLayer` simplesmente
subtrai a média de sua entrada.
Para construí-lo, simplesmente precisamos herdar
da classe da camada base e implementar a função de propagação direta.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)
```

Let us verify that our layer works as intended by feeding some data through it.

Vamos verificar se nossa camada funciona conforme o esperado, alimentando alguns dados por meio dela.

```{.python .input}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab pytorch
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```

```{.python .input}
#@tab tensorflow
layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))
```

We can now incorporate our layer as a component
in constructing more complex models.

Agora podemos incorporar nossa camada como um componente
na construção de modelos mais complexos.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```

```{.python .input}
#@tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

As an extra sanity check, we can send random data
through the network and check that the mean is in fact 0.
Because we are dealing with floating point numbers,
we may still see a very small nonzero number
due to quantization.

Como uma verificação extra de sanidade, podemos enviar dados aleatórios
através da rede e verifique se a média é de fato 0.
Porque estamos lidando com números de ponto flutuante,
ainda podemos ver um número muito pequeno diferente de zero
devido à quantização.

```{.python .input}
Y = net(np.random.uniform(size=(4, 8)))
Y.mean()
```

```{.python .input}
#@tab pytorch
Y = net(torch.rand(4, 8))
Y.mean()
```

```{.python .input}
#@tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

## Layers with Parameters

Now that we know how to define simple layers,
let us move on to defining layers with parameters
that can be adjusted through training.
We can use built-in functions to create parameters, which
provide some basic housekeeping functionality.
In particular, they govern access, initialization,
sharing, saving, and loading model parameters.
This way, among other benefits, we will not need to write
custom serialization routines for every custom layer.

Agora que sabemos como definir camadas simples,
vamos prosseguir para a definição de camadas com parâmetros
que pode ser ajustado por meio de treinamento.
Podemos usar funções integradas para criar parâmetros, que
fornecem algumas funcionalidades básicas de manutenção.
Em particular, eles governam o acesso, inicialização,
compartilhar, salvar e carregar parâmetros do modelo.
Dessa forma, entre outros benefícios, não precisaremos escrever
rotinas de serialização personalizadas para cada camada personalizada.

Now let us implement our own version of the  fully-connected layer.
Recall that this layer requires two parameters,
one to represent the weight and the other for the bias.
In this implementation, we bake in the ReLU activation as a default.
This layer requires to input arguments: `in_units` and `units`, which
denote the number of inputs and outputs, respectively.

Agora, vamos implementar nossa própria versão da camada totalmente conectada.
Lembre-se de que esta camada requer dois parâmetros,
um para representar o peso e outro para o viés.
Nesta implementação, preparamos a ativação do ReLU como padrão.
Esta camada requer a entrada de argumentos: `in_units` e` unidades`, que
denotam o número de entradas e saídas, respectivamente.

```{.python .input}
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
#@tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
#@tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

Next, we instantiate the `MyDense` class
and access its model parameters.

Em seguida, instanciamos a classe `MyDense`
e acessar seus parâmetros de modelo.

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
#@tab pytorch
dense = MyLinear(5, 3)
dense.weight
```

```{.python .input}
#@tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

We can directly carry out forward propagation calculations using custom layers.

Podemos realizar cálculos de propagação direta usando camadas personalizadas.

```{.python .input}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
#@tab pytorch
dense(torch.rand(2, 5))
```

```{.python .input}
#@tab tensorflow
dense(tf.random.uniform((2, 5)))
```

We can also construct models using custom layers.
Once we have that we can use it just like the built-in fully-connected layer.

Também podemos construir modelos usando camadas personalizadas.
Assim que tivermos isso, podemos usá-lo como a camada totalmente conectada integrada.

```{.python .input}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

## Summary

* We can design custom layers via the basic layer class. This allows us to define flexible new layers that behave differently from any existing layers in the library.
* Once defined, custom layers can be invoked in arbitrary contexts and architectures.
* Layers can have local parameters, which can be created through built-in functions.
* 
* Podemos projetar camadas personalizadas por meio da classe de camada básica. Isso nos permite definir novas camadas flexíveis que se comportam de maneira diferente de quaisquer camadas existentes na biblioteca.
* Uma vez definidas, as camadas personalizadas podem ser chamadas em contextos e arquiteturas arbitrários.
* As camadas podem ter parâmetros locais, que podem ser criados por meio de funções integradas.


## Exercises

1. Design a layer that takes an input and computes a tensor reduction,
   i.e., it returns $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
2. Design a layer that returns the leading half of the Fourier coefficients of the data.

3. Projete uma camada que recebe uma entrada e calcula uma redução de tensor,
    ou seja, ele retorna $ y_k = \ sum_ {i, j} W_ {ijk} x_i x_j $.
4. Projete uma camada que retorne a metade anterior dos coeficientes de Fourier dos dados.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/279)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExNDU2NTM3OThdfQ==
-->
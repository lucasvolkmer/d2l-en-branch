# Camadas e Blocos
:label:`sec_model_construction`

Quando introduzimos as redes neurais pela primeira vez,
focamos em modelos lineares com uma única saída.
Aqui, todo o modelo consiste em apenas um único neurônio.
Observe que um único neurônio
(i) leva algum conjunto de entradas;
(ii) gera uma saída escalar correspondente;
e (iii) tem um conjunto de parâmetros associados que podem ser atualizados
para otimizar alguma função objetivo de interesse.
Então, quando começamos a pensar em redes com múltiplas saídas,
nós alavancamos a aritmética vetorizada
para caracterizar uma camada inteira de neurônios.
Assim como os neurônios individuais,
camadas (i) recebem um conjunto de entradas,
(ii) gerar resultados correspondentes,
e (iii) são descritos por um conjunto de parâmetros ajustáveis.
Quando trabalhamos com a regressão softmax,
uma única camada era ela própria o modelo.
No entanto, mesmo quando subsequentemente
introduziu MLPs,
ainda podemos pensar no modelo como
mantendo esta mesma estrutura básica.

Curiosamente, para MLPs,
todo o modelo e suas camadas constituintes
compartilham essa estrutura.
Todo o modelo recebe entradas brutas (os recursos),
gera resultados (as previsões),
e possui parâmetros
(os parâmetros combinados de todas as camadas constituintes).
Da mesma forma, cada camada individual ingere entradas
(fornecido pela camada anterior)
gera saídas (as entradas para a camada subsequente),
e possui um conjunto de parâmetros ajustáveis que são atualizados
de acordo com o sinal que flui para trás
da camada subsequente.

Embora você possa pensar que neurônios, camadas e modelos
dê-nos abstrações suficientes para cuidar de nossos negócios,
Acontece que muitas vezes achamos conveniente
para falar sobre componentes que são
maior do que uma camada individual
mas menor do que o modelo inteiro.
Por exemplo, a arquitetura ResNet-152,
que é muito popular na visão computacional,
possui centenas de camadas.
Essas camadas consistem em padrões repetidos de *grupos de camadas*. Implementar uma camada de rede por vez pode se tornar tedioso.
Essa preocupação não é apenas hipotética --- tal
padrões de projeto são comuns na prática.
A arquitetura ResNet mencionada acima
venceu as competições de visão computacional ImageNet e COCO 2015
para reconhecimento e detecção :cite:`He.Zhang.Ren.ea.2016`
e continua sendo uma arquitetura indispensável para muitas tarefas de visão.
Arquiteturas semelhantes nas quais as camadas são organizadas
em vários padrões repetidos
agora são onipresentes em outros domínios,
incluindo processamento de linguagem natural e fala.

To implement these complex networks,
we introduce the concept of a neural network *block*.
A block could describe a single layer,
a component consisting of multiple layers,
or the entire model itself!
One benefit of working with the block abstraction
is that they can be combined into larger artifacts,
often recursively. This is illustrated in :numref:`fig_blocks`. By defining code to generate blocks
of arbitrary complexity on demand,
we can write surprisingly compact code
and still implement complex neural networks.

Para implementar essas redes complexas,
introduzimos o conceito de uma rede neural *bloco*.
Um bloco pode descrever uma única camada,
um componente que consiste em várias camadas,
ou o próprio modelo inteiro!
Uma vantagem de trabalhar com a abstração de bloco
é que eles podem ser combinados em artefatos maiores,
frequentemente recursivamente. Isso é ilustrado em :numref:`fig_blocks`. Definindo o código para gerar blocos
de complexidade arbitrária sob demanda,
podemos escrever código surpreendentemente compacto
e ainda implementar redes neurais complexas.

![Multiple layers are combined into blocks, forming repeating patterns of larger models.](../img/blocks.svg)
:label:`fig_blocks`


From a programing standpoint, a block is represented by a *class*.
Any subclass of it must define a forward propagation function
that transforms its input into output
and must store any necessary parameters.
Note that some blocks do not require any parameters at all.
Finally a block must possess a backpropagation function,
for purposes of calculating gradients.
Fortunately, due to some behind-the-scenes magic
supplied by the auto differentiation
(introduced in :numref:`sec_autograd`)
when defining our own block,
we only need to worry about parameters
and the forward propagation function.

Do ponto de vista da programação, um bloco é representado por uma * classe *.
Qualquer subclasse dele deve definir uma função de propagação direta
que transforma sua entrada em saída
e deve armazenar todos os parâmetros necessários.
Observe que alguns blocos não requerem nenhum parâmetro.
Finalmente, um bloco deve possuir uma função de retropropagação,
para fins de cálculo de gradientes.
Felizmente, devido a alguma magia dos bastidores
fornecido pela diferenciação automática
(introduzido em: numref: `sec_autograd`)
ao definir nosso próprio bloco,
só precisamos nos preocupar com os parâmetros
e a função de propagação direta.

To begin, we revisit the code
that we used to implement MLPs
(:numref:`sec_mlp_concise`).
The following code generates a network
with one fully-connected hidden layer
with 256 units and ReLU activation,
followed by a fully-connected output layer
with 10 units (no activation function).

Para começar, revisitamos o código
que usamos para implementar MLPs
(: numref: `sec_mlp_concise`).
O código a seguir gera uma rede
com uma camada oculta totalmente conectada
com 256 unidades e ativação ReLU,
seguido por uma camada de saída totalmente conectada
com 10 unidades (sem função de ativação).

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X)
```

:begin_tab:`mxnet`
In this example, we constructed
our model by instantiating an `nn.Sequential`,
assigning the returned object to the `net` variable.
Next, we repeatedly call its `add` function,
appending layers in the order
that they should be executed.
In short, `nn.Sequential` defines a special kind of `Block`,
the class that presents a block in Gluon.
It maintains an ordered list of constituent `Block`s.
The `add` function simply facilitates
the addition of each successive `Block` to the list.
Note that each layer is an instance of the `Dense` class
which is itself a subclass of `Block`.
The forward propagation (`forward`) function is also remarkably simple:
it chains each `Block` in the list together,
passing the output of each as the input to the next.
Note that until now, we have been invoking our models
via the construction `net(X)` to obtain their outputs.
This is actually just shorthand for `net.forward(X)`,
a slick Python trick achieved via
the `Block` class's `__call__` function.

Neste exemplo, nós construímos
nosso modelo instanciando um `nn.Sequential`,
atribuindo o objeto retornado à variável `net`.
Em seguida, chamamos repetidamente sua função `add`,
anexando camadas no pedido
que eles devem ser executados.
Em suma, `nn.Sequential` define um tipo especial de` Block`,
a classe que apresenta um bloco em Gluon.
Ele mantém uma lista ordenada de `Block`s constituintes.
A função `add` simplesmente facilita
a adição de cada `Bloco` sucessivo à lista.
Observe que cada camada é uma instância da classe `Dense`
que é uma subclasse de `Block`.
A função de propagação direta (`para frente`) também é notavelmente simples:
ele encadeia cada `Bloco` na lista,
passando a saída de cada um como entrada para o próximo.
Observe que, até agora, temos invocado nossos modelos
através da construção `net (X)` para obter seus resultados.
Na verdade, isso é apenas um atalho para `net.forward (X)`,
um truque Python habilidoso alcançado via
a função `__call__` da classe` Block`.
:end_tab:

:begin_tab:`pytorch`
In this example, we constructed
our model by instantiating an `nn.Sequential`, with layers in the order
that they should be executed passed as arguments.
In short, `nn.Sequential` defines a special kind of `Module`,
the class that presents a block in PyTorch.
It maintains an ordered list of constituent `Module`s.
Note that each of the two fully-connected layers is an instance of the `Linear` class
which is itself a subclass of `Module`.
The forward propagation (`forward`) function is also remarkably simple:
it chains each block in the list together,
passing the output of each as the input to the next.
Note that until now, we have been invoking our models
via the construction `net(X)` to obtain their outputs.
This is actually just shorthand for `net.__call__(X)`.

Neste exemplo, nós construímos
nosso modelo instanciando um `nn.Sequential`, com camadas na ordem
que eles devem ser executados passados como argumentos.
Em suma, `nn.Sequential` define um tipo especial de` Módulo`,
a classe que apresenta um bloco em PyTorch.
Ele mantém uma lista ordenada de `Módulos` constituintes.
Observe que cada uma das duas camadas totalmente conectadas é uma instância da classe `Linear`
que é uma subclasse de `Módulo`.
A função de propagação direta (`para frente`) também é notavelmente simples:
ele encadeia cada bloco da lista,
passando a saída de cada um como entrada para o próximo.
Observe que, até agora, temos invocado nossos modelos
através da construção `net (X)` para obter seus resultados.
Na verdade, isso é apenas um atalho para `net .__ call __ (X)`.
:end_tab:

:begin_tab:`tensorflow`
In this example, we constructed
our model by instantiating an `keras.models.Sequential`, with layers in the order
that they should be executed passed as arguments.
In short, `Sequential` defines a special kind of `keras.Model`,
the class that presents a block in Keras.
It maintains an ordered list of constituent `Model`s.
Note that each of the two fully-connected layers is an instance of the `Dense` class
which is itself a subclass of `Model`.
The forward propagation (`call`) function is also remarkably simple:
it chains each block in the list together,
passing the output of each as the input to the next.
Note that until now, we have been invoking our models
via the construction `net(X)` to obtain their outputs.
This is actually just shorthand for `net.call(X)`,
a slick Python trick achieved via
the Block class's `__call__` function.

Neste exemplo, nós construímos
nosso modelo instanciando um `keras.models.Sequential`, com camadas na ordem
que eles devem ser executados passados como argumentos.
Em suma, `Sequential` define um tipo especial de` keras.Model`,
a classe que apresenta um bloco em Keras.
Ele mantém uma lista ordenada de `Model`s constituintes.
Observe que cada uma das duas camadas totalmente conectadas é uma instância da classe `Dense`
que é uma subclasse de `Model`.
A função de propagação direta (`chamada`) também é extremamente simples:
ele encadeia cada bloco da lista,
passando a saída de cada um como entrada para o próximo.
Observe que, até agora, temos invocado nossos modelos
através da construção `net (X)` para obter seus resultados.
Na verdade, isso é apenas um atalho para `net.call (X)`,
um truque Python habilidoso alcançado via
a função `__call__` da classe Block.
:end_tab:

## A Custom Block

Perhaps the easiest way to develop intuition
about how a block works
is to implement one ourselves.
Before we implement our own custom block,
we briefly summarize the basic functionality
that each block must provide:

Talvez a maneira mais fácil de desenvolver intuição
sobre como funciona um bloco
é implementar um nós mesmos.
Antes de implementar nosso próprio bloco personalizado,
resumimos brevemente a funcionalidade básica
que cada bloco deve fornecer:

1. Ingest input data as arguments to its forward propagation function.
2. Generate an output by having the forward propagation function return a value. Note that the output may have a different shape from the input. For example, the first fully-connected layer in our model above ingests an      input of arbitrary dimension but returns an output of dimension 256.
3. Calculate the gradient of its output with respect to its input, which can be accessed via its backpropagation function. Typically this happens automatically.
4. Store and provide access to those parameters necessary
   to execute the forward propagation computation.
5. Initialize model parameters as needed.

1. Ingerir dados de entrada como argumentos para sua função de propagação direta.
1. Gere uma saída fazendo com que a função de propagação direta retorne um valor. Observe que a saída pode ter uma forma diferente da entrada. Por exemplo, a primeira camada totalmente conectada em nosso modelo acima ingere uma entrada de dimensão arbitrária, mas retorna uma saída de dimensão 256.
1. Calcule o gradiente de sua saída em relação à sua entrada, que pode ser acessado por meio de sua função de retropropagação. Normalmente, isso acontece automaticamente.
1. Armazene e forneça acesso aos parâmetros necessários
    para executar o cálculo de propagação direta.
1. Inicialize os parâmetros do modelo conforme necessário.

In the following snippet,
we code up a block from scratch
corresponding to an MLP
with one hidden layer with 256 hidden units,
and a 10-dimensional output layer.
Note that the `MLP` class below inherits the class that represents a block.
We will heavily rely on the parent class's functions,
supplying only our own constructor (the `__init__` function in Python) and the forward propagation function.

No seguinte snippet,
nós codificamos um bloco do zero
correspondendo a um MLP
com uma camada oculta com 256 unidades ocultas,
e uma camada de saída de 10 dimensões.
Observe que a classe `MLP` abaixo herda a classe que representa um bloco.
Vamos contar muito com as funções da classe pai,
fornecendo apenas nosso próprio construtor (a função `__init__` em Python) e a função de propagação direta.

```{.python .input}
class MLP(nn.Block):
    # Declare a layer with model parameters. Here, we declare two
    # fully-connected layers
    def __init__(self, **kwargs):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.out = nn.Dense(10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # Hidden layer
        self.out = nn.Linear(256, 10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Block` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        # Hidden layer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def call(self, X):
        return self.out(self.hidden((X)))
```

Let us first focus on the forward propagation function.
Note that it takes `X` as the input,
calculates the hidden representation
with the activation function applied,
and outputs its logits.
In this `MLP` implementation,
both layers are instance variables.
To see why this is reasonable, imagine
instantiating two MLPs, `net1` and `net2`,
and training them on different data.
Naturally, we would expect them
to represent two different learned models.

Vamos primeiro nos concentrar na função de propagação direta.
Observe que leva `X` como entrada,
calcula a representação oculta
com a função de ativação aplicada,
e produz seus logits.
Nesta implementação `MLP`,
ambas as camadas são variáveis de instância.
Para ver por que isso é razoável, imagine
instanciando dois MLPs, `net1` e` net2`,
e treiná-los em dados diferentes.
Naturalmente, esperaríamos que eles
para representar dois modelos aprendidos diferentes.

We instantiate the MLP's layers
in the constructor
and subsequently invoke these layers
on each call to the forward propagation function.
Note a few key details.
First, our customized `__init__` function
invokes the parent class's `__init__` function
via `super().__init__()`
sparing us the pain of restating
boilerplate code applicable to most blocks.
We then instantiate our two fully-connected layers,
assigning them to `self.hidden` and `self.out`.
Note that unless we implement a new operator,
we need not worry about the backpropagation function
or parameter initialization.
The system will generate these functions automatically.
Let us try this out.

Nós instanciamos as camadas do MLP
no construtor
e posteriormente invocar essas camadas
em cada chamada para a função de propagação direta.
Observe alguns detalhes importantes.
Primeiro, nossa função `__init__` personalizada
invoca a função `__init__` da classe pai
via `super () .__ init __ ()`
poupando-nos da dor de reafirmar
código padrão aplicável à maioria dos blocos.
Em seguida, instanciamos nossas duas camadas totalmente conectadas,
atribuindo-os a `self.hidden` e` self.out`.
Observe que, a menos que implementemos um novo operador,
não precisamos nos preocupar com a função de retropropagação
ou inicialização de parâmetro.
O sistema irá gerar essas funções automaticamente.
Vamos tentar fazer isso.

```{.python .input}
net = MLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MLP()
net(X)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(X)
```

A key virtue of the block abstraction is its versatility.
We can subclass a block to create layers
(such as the fully-connected layer class),
entire models (such as the `MLP` class above),
or various components of intermediate complexity.
We exploit this versatility
throughout the following chapters,
such as when addressing
convolutional neural networks.

Uma virtude fundamental da abstração em bloco é sua versatilidade.
Podemos criar uma subclasse de um bloco para criar camadas
(como a classe de camada totalmente conectada),
modelos inteiros (como a classe `MLP` acima),
ou vários componentes de complexidade intermediária.
Nós exploramos essa versatilidade
ao longo dos capítulos seguintes,
como ao abordar
redes neurais convolucionais.


## The Sequential Block

We can now take a closer look
at how the `Sequential` class works.
Recall that `Sequential` was designed
to daisy-chain other blocks together.
To build our own simplified `MySequential`,
we just need to define two key function:
1. A function to append blocks one by one to a list.
2. A forward propagation function to pass an input through the chain of blocks, in the same order as they were appended.

The following `MySequential` class delivers the same
functionality of the default `Sequential` class.

Agora podemos dar uma olhada mais de perto
em como a classe `Sequential` funciona.
Lembre-se de que `Sequential` foi projetado
para conectar outros blocos em série.
Para construir nosso próprio `MySequential` simplificado,
só precisamos definir duas funções principais:
1. Uma função para anexar blocos um a um a uma lista.
2. Uma função de propagação direta para passar uma entrada através da cadeia de blocos, na mesma ordem em que foram acrescentados.

A seguinte classe `MySequential` oferece o mesmo
funcionalidade da classe `Sequential` padrão.

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
        # Here, `block` is an instance of a `Block` subclass, and we assume 
        # that it has a unique name. We save it in the member variable
        # `_children` of the `Block` class, and its type is OrderedDict. When
        # the `MySequential` instance calls the `initialize` function, the
        # system automatically initializes all members of `_children`
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # Here, `module` is an instance of a `Module` subclass. We save it
            # in the member variable `_modules` of the `Module` class, and its
            # type is OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._modules.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # Here, `block` is an instance of a `tf.keras.layers.Layer`
            # subclass
            self.modules.append(block)

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
The `add` function adds a single block
to the ordered dictionary `_children`.
You might wonder why every Gluon `Block`
possesses a `_children` attribute
and why we used it rather than just
define a Python list ourselves.
In short the chief advantage of `_children`
is that during our block's parameter initialization,
Gluon knows to look inside the `_children`
dictionary to find sub-blocks whose
parameters also need to be initialized.

A função `add` adiciona um único bloco
para o dicionário ordenado `_children`.
Você deve estar se perguntando por que todo bloco de Gluon
possui um atributo `_children`
e por que o usamos em vez de apenas
definir uma lista Python nós mesmos.
Resumindo, a principal vantagem das `_crianças`
é que durante a inicialização do parâmetro do nosso bloco,
Gluon sabe olhar dentro das `_crianças`
dicionário para encontrar sub-blocos cujo
os parâmetros também precisam ser inicializados.
:end_tab:

:begin_tab:`pytorch`
In the `__init__` method, we add every module
to the ordered dictionary `_modules` one by one.
You might wonder why every `Module`
possesses a `_modules` attribute
and why we used it rather than just
define a Python list ourselves.
In short the chief advantage of `_modules`
is that during our module's parameter initialization,
the system knows to look inside the `_modules`
dictionary to find sub-modules whose
parameters also need to be initialized.

No método `__init__`, adicionamos todos os módulos
para o dicionário ordenado `_modules` um por um.
Você pode se perguntar por que todo `Módulo`
possui um atributo `_modules`
e por que o usamos em vez de apenas
definir uma lista Python nós mesmos.
Em suma, a principal vantagem de `_modules`
é que durante a inicialização do parâmetro do nosso módulo,
o sistema sabe olhar dentro do `_modules`
dicionário para encontrar submódulos cujo
os parâmetros também precisam ser inicializados.
:end_tab:

When our `MySequential`'s forward propagation function is invoked,
each added block is executed
in the order in which they were added.
We can now reimplement an MLP
using our `MySequential` class.

Quando a função de propagação direta de nosso `MySequential` é invocada,
cada bloco adicionado é executado
na ordem em que foram adicionados.
Agora podemos reimplementar um MLP
usando nossa classe `MySequential`.

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X)
```

Note that this use of `MySequential`
is identical to the code we previously wrote
for the `Sequential` class
(as described in :numref:`sec_mlp_concise`).

Observe que este uso de `MySequential`
é idêntico ao código que escrevemos anteriormente
para a classe `Sequential`
(conforme descrito em: numref: `sec_mlp_concise`).


## Executing Code in the Forward Propagation Function

The `Sequential` class makes model construction easy,
allowing us to assemble new architectures
without having to define our own class.
However, not all architectures are simple daisy chains.
When greater flexibility is required,
we will want to define our own blocks.
For example, we might want to execute
Python's control flow within the forward propagation function.
Moreover, we might want to perform
arbitrary mathematical operations,
not simply relying on predefined neural network layers.

A classe `Sequential` facilita a construção do modelo,
nos permitindo montar novas arquiteturas
sem ter que definir nossa própria classe.
No entanto, nem todas as arquiteturas são cadeias simples.
Quando uma maior flexibilidade é necessária,
vamos querer definir nossos próprios blocos.
Por exemplo, podemos querer executar
Fluxo de controle do Python dentro da função de propagação direta.
Além disso, podemos querer realizar
operações matemáticas arbitrárias,
não simplesmente depender de camadas de rede neural predefinidas.

You might have noticed that until now,
all of the operations in our networks
have acted upon our network's activations
and its parameters.
Sometimes, however, we might want to
incorporate terms
that are neither the result of previous layers
nor updatable parameters.
We call these *constant parameters*.
Say for example that we want a layer
that calculates the function
$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$,
where $\mathbf{x}$ is the input, $\mathbf{w}$ is our parameter,
and $c$ is some specified constant
that is not updated during optimization.
So we implement a `FixedHiddenMLP` class as follows.

Você deve ter notado que até agora,
todas as operações em nossas redes
agiram de acordo com as ativações de nossa rede
e seus parâmetros.
Às vezes, no entanto, podemos querer
incorporar termos
que não são resultado de camadas anteriores
nem parâmetros atualizáveis.
Chamamos isso de * parâmetros constantes *.
Digamos, por exemplo, que queremos uma camada
que calcula a função
$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$,
onde $\mathbf{x}$ is the input, $\mathbf{w}$ é nosso parâmetro,
e $ c $ é alguma constante especificada
que não é atualizado durante a otimização.
Portanto, implementamos uma classe `FixedHiddenMLP` como segue.

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Random weight parameters created with the `get_constant` function
        # are not updated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Use the created constant parameters, as well as the `relu` and `dot`
        # functions
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # Use the created constant parameters, as well as the `relu` and `mm`
        # functions
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters created with `tf.constant` are not updated
        # during training (i.e., constant parameters)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Use the created constant parameters, as well as the `relu` and
        # `matmul` functions
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Reuse the fully-connected layer. This is equivalent to sharing
        # parameters with two fully-connected layers
        X = self.dense(X)
        # Control flow
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

In this `FixedHiddenMLP` model,
we implement a hidden layer whose weights
(`self.rand_weight`) are initialized randomly
at instantiation and are thereafter constant.
This weight is not a model parameter
and thus it is never updated by backpropagation.
The network then passes the output of this "fixed" layer
through a fully-connected layer.

Neste modelo `FixedHiddenMLP`,
implementamos uma camada oculta cujos pesos
(`self.rand_weight`) são inicializados aleatoriamente
na instanciação e daí em diante constantes.
Este peso não é um parâmetro do modelo
e, portanto, nunca é atualizado por retropropagação.
A rede então passa a saída desta camada "fixa"
através de uma camada totalmente conectada.

Note that before returning the output,
our model did something unusual.
We ran a while-loop, testing
on the condition its $L_1$ norm is larger than $1$,
and dividing our output vector by $2$
until it satisfied the condition.
Finally, we returned the sum of the entries in `X`.
To our knowledge, no standard neural network
performs this operation.
Note that this particular operation may not be useful
in any real-world task.
Our point is only to show you how to integrate
arbitrary code into the flow of your
neural network computations.

Observe que antes de retornar a saída,
nosso modelo fez algo incomum.
Executamos um loop while, testando
na condição de que sua norma $ L_1 $ seja maior que $ 1 $,
e dividindo nosso vetor de produção por $ 2 $
até que satisfizesse a condição.
Finalmente, retornamos a soma das entradas em `X`.
Até onde sabemos, nenhuma rede neural padrão
executa esta operação.
Observe que esta operação em particular pode não ser útil
em qualquer tarefa do mundo real.
Nosso objetivo é apenas mostrar como integrar
código arbitrário no fluxo de seu
cálculos de rede neural.

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch, tensorflow
net = FixedHiddenMLP()
net(X)
```

We can mix and match various
ways of assembling blocks together.
In the following example, we nest blocks
in some creative ways.

Podemos misturar e combinar vários
maneiras de montar blocos juntos.
No exemplo a seguir, aninhamos blocos
de algumas maneiras criativas.

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
#@tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

## Efficiency

:begin_tab:`mxnet`
The avid reader might start to worry
about the efficiency of some of these operations.
After all, we have lots of dictionary lookups,
code execution, and lots of other Pythonic things
taking place in what is supposed to be
a high-performance deep learning library.
The problems of Python's [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) are well known. 
In the context of deep learning,
we may worry that our extremely fast GPU(s)
might have to wait until a puny CPU
runs Python code before it gets another job to run.
The best way to speed up Python is by avoiding it altogether.

O leitor ávido pode começar a se preocupar
sobre a eficiência de algumas dessas operações.
Afinal, temos muitas pesquisas de dicionário,
execução de código e muitas outras coisas Pythônicas
ocorrendo no que deveria ser
uma biblioteca de aprendizado profundo de alto desempenho.
Os problemas do Python [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) são bem conhecidos.
No contexto de aprendizagem profunda,
podemos nos preocupar que nossas GPU (s) extremamente rápidas
pode ter que esperar até uma CPU insignificante
executa o código Python antes de obter outro trabalho para ser executado.
A melhor maneira de acelerar o Python é evitá-lo completamente.

One way that Gluon does this is by allowing for
*hybridization*, which will be described later.
Here, the Python interpreter executes a block
the first time it is invoked.
The Gluon runtime records what is happening
and the next time around it short-circuits calls to Python.
This can accelerate things considerably in some cases
but care needs to be taken when control flow (as above)
leads down different branches on different passes through the net.
We recommend that the interested reader checks out
the hybridization section (:numref:`sec_hybridize`)
to learn about compilation after finishing the current chapter.

Uma maneira de o Gluon fazer isso é permitindo
* hibridização *, que será descrita mais tarde.
Aqui, o interpretador Python executa um bloco
na primeira vez que é invocado.
O tempo de execução do Gluon registra o que está acontecendo
e, da próxima vez, provoca um curto-circuito nas chamadas para Python.
Isso pode acelerar as coisas consideravelmente em alguns casos
mas é preciso ter cuidado ao controlar o fluxo (como acima)
conduz a diferentes ramos em diferentes passagens através da rede.
Recomendamos que o leitor interessado verifique
a seção de hibridização (: numref: `sec_hybridize`)
para aprender sobre a compilação depois de terminar o capítulo atual.
:end_tab:

:begin_tab:`pytorch`
The avid reader might start to worry
about the efficiency of some of these operations.
After all, we have lots of dictionary lookups,
code execution, and lots of other Pythonic things
taking place in what is supposed to be
a high-performance deep learning library.
The problems of Python's [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) are well known. 
In the context of deep learning,
we may worry that our extremely fast GPU(s)
might have to wait until a puny CPU
runs Python code before it gets another job to run.

O leitor ávido pode começar a se preocupar
sobre a eficiência de algumas dessas operações.
Afinal, temos muitas pesquisas de dicionário,
execução de código e muitas outras coisas Pythônicas
ocorrendo no que deveria ser
uma biblioteca de aprendizado profundo de alto desempenho.
Os problemas do [bloqueio do interpretador global] do Python (https://wiki.python.org/moin/GlobalInterpreterLock) são bem conhecidos.
No contexto de aprendizagem profunda,
podemos nos preocupar que nossas GPU (s) extremamente rápidas
pode ter que esperar até uma CPU insignificante
executa o código Python antes de obter outro trabalho para ser executado.
:end_tab:

:begin_tab:`tensorflow`
The avid reader might start to worry
about the efficiency of some of these operations.
After all, we have lots of dictionary lookups,
code execution, and lots of other Pythonic things
taking place in what is supposed to be
a high-performance deep learning library.
The problems of Python's [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) are well known. 
In the context of deep learning,
we may worry that our extremely fast GPU(s)
might have to wait until a puny CPU
runs Python code before it gets another job to run.
The best way to speed up Python is by avoiding it altogether.

O leitor ávido pode começar a se preocupar
sobre a eficiência de algumas dessas operações.
Afinal, temos muitas pesquisas de dicionário,
execução de código e muitas outras coisas Pythônicas
ocorrendo no que deveria ser
uma biblioteca de aprendizado profundo de alto desempenho.
Os problemas do Python [global interpreter lock](https://wiki.python.org/moin/GlobalInterpreterLock) são bem conhecidos.
No contexto de aprendizagem profunda,
podemos nos preocupar que nossas GPU (s) extremamente rápidas
pode ter que esperar até uma CPU insignificante
executa o código Python antes de obter outro trabalho para ser executado.
A melhor maneira de acelerar o Python é evitá-lo completamente.
:end_tab:

## Summary

* Layers are blocks.
* Many layers can comprise a block.
* Many blocks can comprise a block.
* A block can contain code.
* Blocks take care of lots of housekeeping, including parameter initialization and backpropagation.
* Sequential concatenations of layers and blocks are handled by the `Sequential` block.

* Camadas são blocos.
* Muitas camadas podem incluir um bloco.
* Muitos blocos podem incluir um bloco.
* Um bloco pode conter código.
* Os blocos cuidam de muitas tarefas domésticas, incluindo inicialização de parâmetros e retropropagação.
* As concatenações sequenciais de camadas e blocos são tratadas pelo bloco `Sequencial`.


## Exercises

1. What kinds of problems will occur if you change `MySequential` to store blocks in a Python list?
2. Implement a block that takes two blocks as an argument, say `net1` and `net2` and returns the concatenated output of both networks in the forward propagation. This is also called a parallel block.
3. Assume that you want to concatenate multiple instances of the same network. Implement a factory function that generates multiple instances of the same block and build a larger network from it.

4. Que tipos de problemas ocorrerão se você alterar `MySequential` para armazenar blocos em uma lista Python?
5. Implemente um bloco que tenha dois blocos como argumento, digamos `net1` e` net2` e retorne a saída concatenada de ambas as redes na propagação direta. Isso também é chamado de bloco paralelo.
6. Suponha que você deseja concatenar várias instâncias da mesma rede. Implemente uma função de fábrica que gere várias instâncias do mesmo bloco e construa uma rede maior a partir dele.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyNjQwMjc4MDldfQ==
-->
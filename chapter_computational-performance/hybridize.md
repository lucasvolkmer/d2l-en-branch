# Compiladores e Interpretadores
:label:`sec_hybridize`

Até agora, este livro se concentrou na programação imperativa, que faz uso de instruções como `print`,` + `ou` if` para alterar o estado de um programa. Considere o seguinte exemplo de um programa imperativo simples.

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

Python é uma linguagem interpretada. Ao avaliar `fancy_func` ele realiza as operações que compõem o corpo da função *em sequência*. Ou seja, ele avaliará `e = add (a, b)` e armazenará os resultados como a variável `e`, alterando assim o estado do programa. As próximas duas instruções `f = add (c, d)` e `g = add (e, f)` serão executadas de forma semelhante, realizando adições e armazenando os resultados como variáveis.  :numref:`fig_compute_graph` ilustra o fluxo de dados.

![Fluxo de dados em um programa imperativo.](../img/computegraph.svg)
:label:`fig_compute_graph`

Embora a programação imperativa seja conveniente, pode ser ineficiente. Por um lado, mesmo se a função `add` for repetidamente chamada em` fancy_func`, Python executará as três chamadas de função individualmente. Se elas forem executadas, digamos, em uma GPU (ou mesmo em várias GPUs), a sobrecarga decorrente do interpretador Python pode se tornar excessiva. Além disso, ele precisará salvar os valores das variáveis `e` e` f` até que todas as instruções em `fancy_func` tenham sido executadas. Isso ocorre porque não sabemos se as variáveis `e` e` f` serão usadas por outras partes do programa após as instruções `e = add (a, b)` e `f = add (c, d)` serem executadas.

## Programação Simbólica


Considere a alternativa de programação simbólica, em que a computação geralmente é realizada apenas depois que o processo foi totalmente definido. Essa estratégia é usada por vários frameworks de aprendizado profundo, incluindo Theano, Keras e TensorFlow (os dois últimos adquiriram extensões imperativas). Geralmente envolve as seguintes etapas:

1. Definir as operações a serem executadas.
1. Compilar as operações em um programa executável.
1. Fornecer as entradas necessárias e chamar o programa compilado para execução.

Isso permite uma quantidade significativa de otimização. Em primeiro lugar, podemos pular o interpretador Python em muitos casos, removendo assim um gargalo de desempenho que pode se tornar significativo em várias GPUs rápidas emparelhadas com um único thread Python em uma CPU. Em segundo lugar, um compilador pode otimizar e reescrever o código acima em `print ((1 + 2) + (3 + 4))` ou mesmo `print (10)`. Isso é possível porque um compilador consegue ver o código completo antes de transformá-lo em instruções de máquina. Por exemplo, ele pode liberar memória (ou nunca alocá-la) sempre que uma variável não for mais necessária. Ou pode transformar o código inteiramente em uma parte equivalente. Para ter uma ideia melhor, considere a seguinte simulação de programação imperativa (afinal, é Python) abaixo.

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

As diferenças entre a programação imperativa (interpretada) e a programação simbólica são as seguintes:

* A programação imperativa é mais fácil. Quando a programação imperativa é usada em Python, a maior parte do código é direta e fácil de escrever. Também é mais fácil depurar o código de programação imperativo. Isso ocorre porque é mais fácil obter e imprimir todos os valores de variáveis intermediárias relevantes ou usar as ferramentas de depuração integradas do Python.
* A programação simbólica é mais eficiente e fácil de portar. Isso torna mais fácil otimizar o código durante a compilação, além de ter a capacidade de portar o programa para um formato independente do Python. Isso permite que o programa seja executado em um ambiente não-Python, evitando, assim, quaisquer problemas de desempenho em potencial relacionados ao interpretador Python.

## Programação Híbrida


Historicamente, a maioria das estruturas de aprendizagem profunda escolhe entre uma abordagem imperativa ou simbólica. Por exemplo, Theano, TensorFlow (inspirado no último), Keras e CNTK formulam modelos simbolicamente. Por outro lado, Chainer e PyTorch adotam uma abordagem imperativa. Um modo imperativo foi adicionado ao TensorFlow 2.0 (via Eager) e Keras em revisões posteriores.

:begin_tab:`mxnet`
Ao projetar o Gluon, os desenvolvedores consideraram se seria possível combinar os benefícios de ambos os modelos de programação. Isso levou a um modelo híbrido que permite aos usuários desenvolver e depurar usando programação imperativa pura, ao mesmo tempo em que têm a capacidade de converter a maioria dos programas em programas simbólicos a serem executados quando o desempenho e a implantação de computação em nível de produto são necessários.


Na prática, isso significa que construímos modelos usando as classes `HybridBlock` ou` HybridSequential` e `HybridConcurrent`. Por padrão, eles são executados da mesma forma que as classes `Block` ou` Sequential` e `Concurrent` são executadas na programação imperativa. `HybridSequential` é uma subclasse de` HybridBlock` (assim como `Sequential` é subclasse de ` Block`). Quando a função `hybridize`  é chamada, o Gluon compila o modelo na forma usada na programação simbólica. Isso permite otimizar os componentes de computação intensiva sem sacrificar a maneira como um modelo é implementado. Ilustraremos os benefícios abaixo, focalizando apenas modelos sequenciais e blocos (a composição concorrente funciona de forma análoga).
:end_tab:

:begin_tab:`pytorch`
Como mencionado acima, PyTorch é baseado em programação imperativa e usa gráficos de computação dinâmica. Em um esforço para alavancar a portabilidade e eficiência da programação simbólica, os desenvolvedores consideraram se seria possível combinar os benefícios de ambos os modelos de programação. Isso levou a um *torchscript* que permite aos usuários desenvolver e depurar usando programação imperativa pura, ao mesmo tempo em que têm a capacidade de converter a maioria dos programas em programas simbólicos para serem executados quando o desempenho e a implantação de computação em nível de produto forem necessários.
:end_tab:

:begin_tab:`tensorflow`
O paradigma de programação imperativo agora é o padrão no Tensorflow 2, uma mudança acolhedora para aqueles que são novos na linguagem. No entanto, as mesmas técnicas de programação simbólica e gráficos computacionais subsequentes ainda existem no TensorFlow e podem ser acessados pelo decorador `tf.function` fácil de usar. Isso trouxe o paradigma de programação imperativo para o TensorFlow, permitindo que os usuários definissem funções mais intuitivas, depois as envolvessem e compilassem em gráficos computacionais automaticamente usando um recurso que a equipe do TensorFlow chama de [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph).
:end_tab:

## Híbrido-Sequencial

A maneira mais fácil de ter uma ideia de como a hibridização funciona é considerar redes profundas com várias camadas. Convencionalmente, o interpretador Python precisará executar o código para todas as camadas para gerar uma instrução que pode então ser encaminhada para uma CPU ou GPU. Para um único dispositivo de computação (rápido), isso não causa grandes problemas. Por outro lado, se usarmos um servidor avançado de 8 GPUs, como uma instância AWS P3dn.24xlarge, o Python terá dificuldade para manter todas as GPUs ocupadas. O interpretador Python de thread único torna-se o gargalo aqui. Vamos ver como podemos resolver isso para partes significativas do código, substituindo `Sequential` por `HybridSequential`. Começamos definindo um MLP simples.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# Factory for networks
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Factory for networks
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Factory for networks
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```


:begin_tab:`mxnet`
Ao chamar a função `hybridize`, podemos compilar e otimizar o cálculo no MLP. O resultado do cálculo do modelo permanece inalterado.
:end_tab:

:begin_tab: `pytorch`
Ao converter o modelo usando a função `torch.jit.script`, podemos compilar e otimizar a computação no MLP. O resultado do cálculo do modelo permanece inalterado.
:end_tab:

:begin_tab:`tensorflow`
Anteriormente, todas as funções construídas no tensorflow eram construídas como um gráfico computacional e, portanto, JIT compilado por padrão. No entanto, com o lançamento do tensorflow 2.X e tensores *eager*, este não é mais o comportamento padrão.
Podemos reativar essa funcionalidade com tf.function. tf.function é mais comumente usado como um decorador de função, no entanto, é possível chamá-lo diretamente como uma função Python normal, mostrada abaixo. O resultado do cálculo do modelo permanece inalterado.
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
Isso parece bom demais para ser verdade: simplesmente designe um bloco como `HybridSequential`, escreva o mesmo código de antes e invoque `hybridize`. Assim que isso acontecer, a rede estará otimizada (faremos um benchmark do desempenho abaixo). Infelizmente, isso não funciona magicamente para todas as camadas. Dito isso, os blocos fornecidos pelo Gluon são, por padrão, subclasses de `HybridBlock` e, portanto, hibridizáveis. Uma camada não será otimizada se, em vez disso, herdar do `Bloco`.
:end_tab:

:begin_tab:`pytorch`
Convertendo o modelo usando `torch.jit.script` Isso parece quase bom demais para ser verdade: escreva o mesmo código de antes e simplesmente converta o modelo usando` torch.jit.script`. Assim que isso acontecer, a rede estará otimizada (faremos um benchmark do desempenho abaixo).
:end_tab:

:begin_tab:`tensorflow`
Converter o modelo usando `tf.function` nos dá um poder incrível no TensorFlow: escreva o mesmo código de antes e simplesmente converta o modelo usando` tf.function`. Quando isso acontece, a rede é construída como um gráfico computacional na representação intermediária MLIR do TensorFlow e é altamente otimizada no nível do compilador para uma execução rápida (faremos o benchmark do desempenho abaixo).
Adicionar explicitamente a sinalização `jit_compile = True` à chamada `tf.function()` ativa a funcionalidade XLA (Álgebra Linear Acelerada) no TensorFlow. O XLA pode otimizar ainda mais o código compilado JIT em certas instâncias. A execução no modo gráfico é habilitada sem essa definição explícita, no entanto, o XLA pode tornar certas operações de álgebra linear grandes (na veia daquelas que vemos em aplicativos de aprendizado profundo) muito mais rápidas, particularmente em uma GPUenvironment.
:end_tab:

### Aceleração por Hibridização

To demonstrate the performance improvement gained by compilation we compare the time needed to evaluate `net(x)` before and after hybridization. Let us define a function to measure this time first. It will come handy throughout the chapter as we set out to measure (and improve) performance.

```{.python .input}
#@tab all
#@save
class Benchmark:
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
Now we can invoke the network twice, once with and once without hybridization.
:end_tab:

:begin_tab:`pytorch`
Now we can invoke the network twice, once with and once without torchscript.
:end_tab:

:begin_tab:`tensorflow`
Now we can invoke the network three times, once executed eagerly, once with graph-mode execution, and again using JIT compiled XLA.
:end_tab:

```{.python .input}
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```


:begin_tab:`mxnet`
As is observed in the above results, after a HybridSequential instance calls the `hybridize` function, computing performance is improved through the use of symbolic programming.
:end_tab:

:begin_tab:`pytorch`
As is observed in the above results, after a nn.Sequential instance is scripted using the `torch.jit.script` function, computing performance is improved through the use of symbolic programming.
:end_tab:

:begin_tab:`tensorflow`
As is observed in the above results, after a tf.keras Sequential instance is scripted using the `tf.function` function, computing performance is improved through the use of symbolic programming via graph-mode execution in tensorflow. 
:end_tab:

### Serialization

:begin_tab:`mxnet`
One of the benefits of compiling the models is that we can serialize (save) the model and its parameters to disk. This allows us to store a model in a manner that is independent of the front-end language of choice. This allows us to deploy trained models to other devices and easily use other front-end programming languages. At the same time the code is often faster than what can be achieved in imperative programming. Let us see the `export` method in action.
:end_tab:

:begin_tab:`pytorch`
One of the benefits of compiling the models is that we can serialize (save) the model and its parameters to disk. This allows us to store a model in a manner that is independent of the front-end language of choice. This allows us to deploy trained models to other devices and easily use other front-end programming languages. At the same time the code is often faster than what can be achieved in imperative programming. Let us see the `save` method in action.
:end_tab:

:begin_tab:`tensorflow`
One of the benefits of compiling the models is that we can serialize (save) the model and its parameters to disk. This allows us to store a model in a manner that is independent of the front-end language of choice. This allows us to deploy trained models to other devices and easily use other front-end programming languages or execute a trained model on a server. At the same time the code is often faster than what can be achieved in imperative programming. 
The low-level API that allows us to save in tensorflow is `tf.saved_model`. 
Let's see the `saved_model` instance in action.
:end_tab:

```{.python .input}
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```
```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
The model is decomposed into a (large binary) parameter file and a JSON description of the program required to execute to compute the model. The files can be read by other front-end languages supported by Python or MXNet, such as C++, R, Scala, and Perl. Let us have a look at the model description.
:end_tab:

```{.python .input}
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
Things are slightly more tricky when it comes to models that resemble code more closely. Basically hybridization needs to deal with control flow and Python overhead in a much more immediate manner. Moreover,

Contrary to the Block instance, which needs to use the `forward` function, for a HybridBlock instance we need to use the `hybrid_forward` function.

Earlier, we demonstrated that, after calling the `hybridize` function, the model is able to achieve superior computing performance and portability. Note, though that hybridization can affect model flexibility, in particular in terms of control flow. We will illustrate how to design more general models and also how compilation will remove spurious Python elements.
:end_tab:

```{.python .input}
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
The code above implements a simple network with 4 hidden units and 2 outputs. `hybrid_forward` takes an additional argument - the module `F`. This is needed since, depending on whether the code has been hybridized or not, it will use a slightly different library (`ndarray` or `symbol`) for processing. Both classes perform very similar functions and MXNet automatically determines the argument. To understand what is going on we print the arguments as part of the function invocation.
:end_tab:

```{.python .input}
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
Repeating the forward computation will lead to the same output (we omit details). Now let us see what happens if we invoke the `hybridize` method.
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
Instead of using `ndarray` we now use the `symbol` module for `F`. Moreover, even though the input is of `ndarray` type, the data flowing through the network is now converted to `symbol` type as part of the compilation process. Repeating the function call leads to a surprising outcome:
:end_tab:

```{.python .input}
net(x)
```

:begin_tab:`mxnet` This is quite different from what we saw previously. All print statements, as defined in `hybrid_forward` are omitted. Indeed, after hybridization the execution of `net(x)` does not involve the Python interpreter any longer. This means that any spurious Python code is omitted (such as print statements) in favor of a much more streamlined execution and better performance. Instead, MXNet directly calls the C++ backend. Also note that some functions are not supported in the `symbol` module (like `asnumpy`) and operations in-place like `a += b` and `a[:] = a + b` must be rewritten as `a = a + b`. Nonetheless, compilation of models is worth the effort whenever speed matters. The benefit can range from small percentage points to more than twice the speed, depending on the complexity of the model, the speed of the CPU and the speed and number of GPUs.

## Summary

* Imperative programming makes it easy to design new models since it is possible to write code with control flow and the ability to use a large amount of the Python software ecosystem.
* Symbolic programming requires that we specify the program and compile it before executing it. The benefit is improved performance.
* MXNet is able to combine the advantages of both approaches as needed.
* Models constructed by the `HybridSequential` and `HybridBlock` classes are able to convert imperative programs into symbolic programs by calling the `hybridize` method.


## Exercises

1. Design a network using the `HybridConcurrent` class. Alternatively look at :ref:`sec_googlenet` for a network to compose.
1. Add `x.asnumpy()` to the first line of the `hybrid_forward` function of the HybridNet class in this section. Execute the code and observe the errors you encounter. Why do they happen?
1. What happens if we add control flow, i.e., the Python statements `if` and `for` in the `hybrid_forward` function?
1. Review the models that interest you in the previous chapters and use the HybridBlock class or HybridSequential class to implement them.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/360)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTcyMjkwMTYzMCwxNTIwNzY2OTg2LDE4Mj
M1MjAyODIsLTEyMzk0Mzc4Nyw2NDU2NDY1NjYsMTgzOTc0NDkz
OCwxMDAxMTk5NDYsMTE4MTM2NjgyOV19
-->
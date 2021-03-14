# Deferred Initialization
:label:`sec_deferred_init`

Até agora, pode parecer que escapamos
por ser descuidados na configuração de nossas redes.
Especificamente, fizemos as seguintes coisas não intuitivas,
que podem não parecer que deveriam funcionar:

* Definimos as arquiteturas de rede
   sem especificar a dimensionalidade de entrada.
* Adicionamos camadas sem especificar
   a dimensão de saída da camada anterior.
* Nós até "inicializamos" esses parâmetros
   antes de fornecer informações suficientes para determinar
   quantos parâmetros nossos modelos devem conter.

You might be surprised that our code runs at all.
After all, there is no way the deep learning framework
could tell what the input dimensionality of a network would be.
The trick here is that the framework *defers initialization*,
waiting until the first time we pass data through the model,
to infer the sizes of each layer on the fly.

Você pode se surpreender com o fato de nosso código ser executado.
Afinal, não há como o *framework* de *Deep Learning*
poderia dizer qual seria a dimensionalidade de entrada de uma rede.
O truque aqui é que o framework * adia a inicialização *,
esperando até a primeira vez que passamos os dados pelo modelo,
para inferir os tamanhos de cada camada na hora.


Later on, when working with convolutional neural networks,
this technique will become even more convenient
since the input dimensionality
(i.e., the resolution of an image)
will affect the dimensionality
of each subsequent layer.
Hence, the ability to set parameters
without the need to know,
at the time of writing the code,
what the dimensionality is
can greatly simplify the task of specifying
and subsequently modifying our models.
Next, we go deeper into the mechanics of initialization.

Mais tarde, ao trabalhar com redes neurais convolucionais,
esta técnica se tornará ainda mais conveniente
desde a dimensionalidade de entrada
(ou seja, a resolução de uma imagem)
afetará a dimensionalidade
de cada camada subsequente.
Conseqüentemente, a capacidade de definir parâmetros
sem a necessidade de saber,
no momento de escrever o código,
qual é a dimensionalidade
pode simplificar muito a tarefa de especificar
e subsequentemente modificando nossos modelos.
A seguir, vamos nos aprofundar na mecânica da inicialização.


## Instantiating a Network

To begin, let us instantiate an MLP.

Para começar, vamos instanciar um MLP.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = get_net()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

At this point, the network cannot possibly know
the dimensions of the input layer's weights
because the input dimension remains unknown.
Consequently the framework has not yet initialized any parameters.
We confirm by attempting to access the parameters below.

Neste ponto, a rede não pode saber
as dimensões dos pesos da camada de entrada
porque a dimensão de entrada permanece desconhecida.
Consequentemente, a estrutura ainda não inicializou nenhum parâmetro.
Confirmamos tentando acessar os parâmetros abaixo.

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
Note that while the parameter objects exist,
the input dimension to each layer is listed as -1.
MXNet uses the special value -1 to indicate
that the parameter dimension remains unknown.
At this point, attempts to access `net[0].weight.data()`
would trigger a runtime error stating that the network
must be initialized before the parameters can be accessed.
Now let us see what happens when we attempt to initialize
parameters via the `initialize` function.

Observe que, embora os objetos de parâmetro existam,
a dimensão de entrada para cada camada é listada como -1.
MXNet usa o valor especial -1 para indicar
que a dimensão do parâmetro permanece desconhecida.
Neste ponto, tenta acessar `net [0] .weight.data ()`
desencadearia um erro de tempo de execução informando que a rede
deve ser inicializado antes que os parâmetros possam ser acessados.
Agora vamos ver o que acontece quando tentamos inicializar
parâmetros por meio da função `initialize`.
:end_tab:

:begin_tab:`tensorflow`
Note that each layer objects exist but the weights are empty.
Using `net.get_weights()` would throw an error since the weights
have not been initialized yet.

Observe que cada objeto de camada existe, mas os pesos estão vazios.
Usar `net.get_weights ()` geraria um erro, uma vez que os pesos
ainda não foram inicializados.
:end_tab:

```{.python .input}
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
As we can see, nothing has changed.
When input dimensions are unknown,
calls to initialize do not truly initialize the parameters.
Instead, this call registers to MXNet that we wish
(and optionally, according to which distribution)
to initialize the parameters.

Como podemos ver, nada mudou.
Quando as dimensões de entrada são desconhecidas,
chamadas para inicializar não inicializam verdadeiramente os parâmetros.
Em vez disso, esta chamada se registra no MXNet que desejamos
(e opcionalmente, de acordo com qual distribuição)
para inicializar os parâmetros.
:end_tab:

Next let us pass data through the network
to make the framework finally initialize parameters.

Em seguida, vamos passar os dados pela rede
para fazer o framework finalmente inicializar os parâmetros.

```{.python .input}
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

As soon as we know the input dimensionality,
20,
the framework can identify the shape of the first layer's weight matrix by plugging in the value of 20.
Having recognized the first layer's shape, the framework proceeds
to the second layer,
and so on through the computational graph
until all shapes are known.
Note that in this case,
only the first layer requires deferred initialization,
but the framework initializes sequentially.
Once all parameter shapes are known,
the framework can finally initialize the parameters.

Assim que soubermos a dimensionalidade da entrada,
20,
a estrutura pode identificar a forma da matriz de peso da primeira camada conectando o valor de 20.
Tendo reconhecido a forma da primeira camada, a estrutura prossegue
para a segunda camada,
e assim por diante através do gráfico computacional
até que todas as formas sejam conhecidas.
Observe que, neste caso,
apenas a primeira camada requer inicialização adiada,
mas a estrutura inicializa sequencialmente.
Uma vez que todas as formas dos parâmetros são conhecidas,
a estrutura pode finalmente inicializar os parâmetros.

## Summary

* Deferred initialization can be convenient, allowing the framework to infer parameter shapes automatically, making it easy to modify architectures and eliminating one common source of errors.
* We can pass data through the model to make the framework finally initialize parameters.

* A inicialização adiada pode ser conveniente, permitindo que o framework inferir formas de parâmetros automaticamente, facilitando a modificação de arquiteturas e eliminando uma fonte comum de erros.
* Podemos passar dados através do modelo para fazer o framework finalmente inicializar os parâmetros.


## Exercises

1. What happens if you specify the input dimensions to the first layer but not to subsequent layers? Do you get immediate initialization?
2. What happens if you specify mismatching dimensions?
3. What would you need to do if you have input of varying dimensionality? Hint: look at the parameter tying.

1. O que acontece se você especificar as dimensões de entrada para a primeira camada, mas não para as camadas subsequentes? Você consegue inicialização imediata?
1. O que acontece se você especificar dimensões incompatíveis?
1. O que você precisa fazer se tiver dados de dimensionalidade variável? Dica: observe a vinculação de parâmetros.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTkyOTM1MjQ1LDExMzU1ODY3NzRdfQ==
-->
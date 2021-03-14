# File I/O

So far we discussed how to process data and how
to build, train, and test deep learning models.
However, at some point, we will hopefully be happy enough
with the learned models that we will want
to save the results for later use in various contexts
(perhaps even to make predictions in deployment).
Additionally, when running a long training process,
the best practice is to periodically save intermediate results (checkpointing)
to ensure that we do not lose several days worth of computation
if we trip over the power cord of our server.
Thus it is time to learn how to load and store
both individual weight vectors and entire models.
This section addresses both issues.

Até agora, discutimos como processar dados e como
para construir, treinar e testar modelos de aprendizado profundo.
No entanto, em algum momento, esperamos ser felizes o suficiente
com os modelos aprendidos que queremos
para salvar os resultados para uso posterior em vários contextos
(talvez até mesmo para fazer previsões na implantação).
Além disso, ao executar um longo processo de treinamento,
a prática recomendada é salvar resultados intermediários periodicamente (pontos de verificação)
para garantir que não perdemos vários dias de computação
se tropeçarmos no cabo de alimentação do nosso servidor.
Portanto, é hora de aprender como carregar e armazenar
ambos os vetores de peso individuais e modelos inteiros.
Esta seção aborda ambos os problemas.

## Loading and Saving Tensors

For individual tensors, we can directly
invoke the `load` and `save` functions
to read and write them respectively.
Both functions require that we supply a name,
and `save` requires as input the variable to be saved.

Para tensores individuais, podemos diretamente
invocar as funções `load` e` save`
para ler e escrever respectivamente.
Ambas as funções exigem que forneçamos um nome,
e `save` requer como entrada a variável a ser salva.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

x = tf.range(4)
np.save("x-file.npy", x)
```

We can now read the data from the stored file back into memory.

Agora podemos ler os dados do arquivo armazenado de volta na memória.

```{.python .input}
x2 = npx.load('x-file')
x2
```

```{.python .input}
#@tab pytorch
x2 = torch.load("x-file")
x2
```

```{.python .input}
#@tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

We can store a list of tensors and read them back into memory.

Podemos armazenar uma lista de tensores e lê-los de volta na memória.
```{.python .input}
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

We can even write and read a dictionary that maps
from strings to tensors.
This is convenient when we want
to read or write all the weights in a model.

Podemos até escrever e ler um dicionário que mapeia
de cordas a tensores.
Isso é conveniente quando queremos
para ler ou escrever todos os pesos em um modelo.

```{.python .input}
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
#@tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
#@tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

## Loading and Saving Model Parameters

Saving individual weight vectors (or other tensors) is useful,
but it gets very tedious if we want to save
(and later load) an entire model.
After all, we might have hundreds of
parameter groups sprinkled throughout.
For this reason the deep learning framework provides built-in functionalities
to load and save entire networks.
An important detail to note is that this
saves model *parameters* and not the entire model.
For example, if we have a 3-layer MLP,
we need to specify the architecture separately.
The reason for this is that the models themselves can contain arbitrary code,
hence they cannot be serialized as naturally.
Thus, in order to reinstate a model, we need
to generate the architecture in code
and then load the parameters from disk.
Let us start with our familiar MLP.

Salvar vetores de peso individuais (ou outros tensores) é útil,
mas fica muito tedioso se quisermos salvar
(e depois carregue) um modelo inteiro.
Afinal, podemos ter centenas de
grupos de parâmetros espalhados por toda parte.
Por esta razão, a estrutura de aprendizagem profunda fornece funcionalidades integradas
para carregar e salvar redes inteiras.
Um detalhe importante a notar é que este
salva o modelo * parâmetros * e não o modelo inteiro.
Por exemplo, se tivermos um MLP de 3 camadas,
precisamos especificar a arquitetura separadamente.
A razão para isso é que os próprios modelos podem conter código arbitrário,
portanto, eles não podem ser serializados naturalmente.
Assim, para restabelecer um modelo, precisamos
para gerar a arquitetura em código
e carregue os parâmetros do disco.
Vamos começar com nosso MLP familiar.

```{.python .input}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

Next, we store the parameters of the model as a file with the name "mlp.params".

A seguir, armazenamos os parâmetros do modelo como um arquivo com o nome "mlp.params".

```{.python .input}
net.save_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
#@tab tensorflow
net.save_weights('mlp.params')
```

To recover the model, we instantiate a clone
of the original MLP model.
Instead of randomly initializing the model parameters,
we read the parameters stored in the file directly.

Para recuperar o modelo, instanciamos um clone
do modelo MLP original.
Em vez de inicializar aleatoriamente os parâmetros do modelo,
lemos os parâmetros armazenados no arquivo diretamente.

```{.python .input}
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))
clone.eval()
```

```{.python .input}
#@tab tensorflow
clone = MLP()
clone.load_weights("mlp.params")
```

Since both instances have the same model parameters,
the computational result of the same input `X` should be the same.
Let us verify this.

Uma vez que ambas as instâncias têm os mesmos parâmetros de modelo,
o resultado computacional da mesma entrada `X` deve ser o mesmo.
Deixe-nos verificar isso.

```{.python .input}
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab pytorch
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab tensorflow
Y_clone = clone(X)
Y_clone == Y
```

## Summary

* The `save` and `load` functions can be used to perform file I/O for tensor objects.
* We can save and load the entire sets of parameters for a network via a parameter dictionary.
* Saving the architecture has to be done in code rather than in parameters.

* As funções `save` e` load` podem ser usadas para executar E / S de arquivo para objetos tensores.
* Podemos salvar e carregar todos os conjuntos de parâmetros de uma rede por meio de um dicionário de parâmetros.
* Salvar a arquitetura deve ser feito em código e não em parâmetros.
## Exercises

1. Even if there is no need to deploy trained models to a different device, what are the practical benefits of storing model parameters?
2. Assume that we want to reuse only parts of a network to be incorporated into a network of a different architecture. How would you go about using, say the first two layers from a previous network in a new network?
3. How would you go about saving the network architecture and parameters? What restrictions would you impose on the architecture?

4. Mesmo se não houver necessidade de implantar modelos treinados em um dispositivo diferente, quais são os benefícios práticos de armazenar parâmetros de modelo?
5. Suponha que desejamos reutilizar apenas partes de uma rede para serem incorporadas a uma rede de arquitetura diferente. Como você usaria, digamos, as duas primeiras camadas de uma rede anterior em uma nova rede?
6. Como você salvaria a arquitetura e os parâmetros da rede? Que restrições você imporia à arquitetura?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/327)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTg5ODAwMDg2OF19
-->
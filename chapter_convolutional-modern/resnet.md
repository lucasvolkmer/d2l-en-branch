# Redes Residuais (ResNet)
:label:`sec_resnet`

À medida que projetamos redes cada vez mais profundas, torna-se imperativo entender como a adição de camadas pode aumentar a complexidade e a expressividade da rede.
Ainda mais importante é a capacidade de projetar redes onde adicionar camadas torna as redes estritamente mais expressivas, em vez de apenas diferentes.
Para fazer algum progresso, precisamos de um pouco de matemática.


## Classes Função

Considere $\mathcal{F}$, a classe de funções que uma arquitetura de rede específica (junto com as taxas de aprendizado e outras configurações de hiperparâmetros) pode alcançar.
Ou seja, para todos os $f \in \mathcal{F}$ existe algum conjunto de parâmetros (por exemplo, pesos e vieses) que podem ser obtidos através do treinamento em um conjunto de dados adequado.
Vamos supor que $f^*$ seja a função "verdade" que realmente gostaríamos de encontrar.
Se estiver em $\mathcal{F}$, estamos em boa forma, mas normalmente não teremos tanta sorte.
Em vez disso, tentaremos encontrar $f^*_\mathcal{F}$, que é nossa melhor aposta em $\mathcal{F}$.
Por exemplo,
dado um conjunto de dados com recursos $\mathbf{X}$
e rótulos $\mathbf{y}$,
podemos tentar encontrá-lo resolvendo o seguinte problema de otimização:

$$f^*_\mathcal{F} \stackrel{\mathrm{def}}{=} \mathop{\mathrm{argmin}}_f L(\mathbf{X}, \mathbf{y}, f) \text{ subject to } f \in \mathcal{F}.$$

É razoável supor que, se projetarmos uma arquitetura diferente e mais poderosa $\mathcal{F}'$, chegaremos a um resultado melhor. Em outras palavras, esperaríamos que $f^*_{\mathcal{F}'}$ seja "melhor" do que $f^*_{\mathcal{F}}$. No entanto, se $\mathcal{F} \not\subseteq \mathcal{F}'$ não há garantia de que isso acontecerá. Na verdade, $f^*_{\mathcal{F}'}$ pode muito bem ser pior.
Conforme ilustrado por :numref:`fig_functionclasses`,
para classes de função não aninhadas, uma classe de função maior nem sempre se aproxima da função "verdade" $f^*$. Por exemplo,
à esquerda de: numref: `fig_functionclasses`,
embora$\mathcal{F}_3$ esteja mais perto de  $f^*$ do que $\mathcal{F}_1$, $\mathcal{F}_6$ se afasta e não há garantia de que aumentar ainda mais a complexidade pode reduzir o distância de $f^*$.
Com classes de função aninhadas
onde $\mathcal{F}_1 \subseteq \ldots \subseteq \mathcal{F}_6$
à direita de :numref:`fig_functionclasses`,
podemos evitar o problema mencionado nas classes de função não aninhadas.


![Para classes de função não aninhadas, uma classe de função maior (indicada por área) não garante a aproximação da função "verdade" ($f^*$). Isso não acontece em classes de funções aninhadas.](../img/functionclasses.svg)
:label:`fig_functionclasses`

Por isso,
somente se as classes de função maiores contiverem as menores teremos a garantia de que aumentá-las aumenta estritamente o poder expressivo da rede.
Para redes neurais profundas,
se pudermos
treinar a camada recém-adicionada em uma função de identidade $f(\mathbf{x}) = \mathbf{x}$, o novo modelo será tão eficaz quanto o modelo original. Como o novo modelo pode obter uma solução melhor para se ajustar ao conjunto de dados de treinamento, a camada adicionada pode facilitar a redução de erros de treinamento.

Essa é a pergunta que He et al. considerado quando se trabalha em modelos de visão computacional muito profundos :cite:`He.Zhang.Ren.ea.2016`.
No cerne de sua proposta de *rede residual* (*ResNet*) está a ideia de que cada camada adicional deve
mais facilmente
conter a função de identidade como um de seus elementos.
Essas considerações são bastante profundas, mas levaram a uma soluçao surpreendentemente simples, um *bloco residual*.
Com ele, a ResNet venceu o Desafio de Reconhecimento Visual em Grande Escala da ImageNet em 2015. O design teve uma profunda influência em como
construir redes neurais profundas.



## Blocos Residuais

Let us focus on a local part of a neural network, as depicted in :numref:`fig_residual_block`. Denote the input by $\mathbf{x}$.
We assume that the desired underlying mapping we want to obtain by learning is $f(\mathbf{x})$, to be used as the input to the activation function on the top.
On the left of :numref:`fig_residual_block`,
the portion within the dotted-line box 
must directly learn the mapping $f(\mathbf{x})$.
On the right,
the portion within the dotted-line box
needs to
learn the *residual mapping* $f(\mathbf{x}) - \mathbf{x}$,
which is how the residual block derives its name.
If the identity mapping $f(\mathbf{x}) = \mathbf{x}$ is the desired underlying mapping,
the residual mapping is easier to learn:
we only need to push the weights and biases
of the
upper weight layer (e.g., fully-connected layer and convolutional layer)
within the dotted-line box
to zero.
The right figure in :numref:`fig_residual_block` illustrates the  *residual block* of ResNet,
where the solid line carrying the layer input 
$\mathbf{x}$ to the addition operator
is called a *residual connection* (or *shortcut connection*).
With residual blocks, inputs can 
forward propagate faster through the residual connections across layers.

Vamos nos concentrar em uma parte local de uma rede neural, conforme descrito em :numref:`fig_residual_block`. Denote a entrada por $\mathbf{x}$.
Assumimos que o mapeamento subjacente desejado que queremos obter aprendendo é $f(\mathbf{x})$, a ser usado como entrada para a função de ativação no topo.
À esquerda de :numref:`fig_residual_block`,
a parte dentro da caixa de linha pontilhada
deve aprender diretamente o mapeamento $f(\mathbf{x})$.
A direita,
a parte dentro da caixa de linha pontilhada
precisa de
aprenda o *mapeamento residual* $f(\mathbf{x}) - \mathbf{x}$,
que é como o bloco residual deriva seu nome.
Se o mapeamento de identidade $f(\mathbf{x}) = \mathbf{x}$ for o mapeamento subjacente desejado,
o mapeamento residual é mais fácil de aprender:
nós só precisamos empurrar os pesos e preconceitos
do
camada de peso superior (por exemplo, camada totalmente conectada e camada convolucional)
dentro da caixa de linha pontilhada
a zero.
A figura certa em :numref:`fig_residual_block` ilustra o *bloco residual* do ResNet,
onde a linha sólida carregando a entrada da camada
$\mathbf{x}$ para o operador de adição
é chamada de *conexão residual* (ou *conexão de atalho*).
Com blocos residuais, as entradas podem
para a frente se propagam mais rápido através das conexões residuais entre as camadas.

![A regular block (left) and a residual block (right).](../img/residual-block.svg)
:label:`fig_residual_block`


ResNet follows VGG's full $3\times 3$ convolutional layer design. The residual block has two $3\times 3$ convolutional layers with the same number of output channels. Each convolutional layer is followed by a batch normalization layer and a ReLU activation function. Then, we skip these two convolution operations and add the input directly before the final ReLU activation function.
This kind of design requires that the output of the two convolutional layers has to be of the same shape as the input, so that they can be added together. If we want to change the number of channels, we need to introduce an additional $1\times 1$ convolutional layer to transform the input into the desired shape for the addition operation. Let us have a look at the code below.

ResNet segue o design de camada convolucional $ 3 \ times 3 $ completo do VGG. O bloco residual tem duas camadas convolucionais $ 3 \ vezes 3 $ com o mesmo número de canais de saída. Cada camada convolucional é seguida por uma camada de normalização em lote e uma função de ativação ReLU. Em seguida, pulamos essas duas operações de convolução e adicionamos a entrada diretamente antes da função de ativação final do ReLU.
Esse tipo de projeto requer que a saída das duas camadas convolucionais tenha o mesmo formato da entrada, para que possam ser somadas. Se quisermos mudar o número de canais, precisamos introduzir uma camada convolucional adicional $ 1 \ vezes 1 $ para transformar a entrada na forma desejada para a operação de adição. Vamos dar uma olhada no código abaixo.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class Residual(nn.Block):  #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class Residual(tf.keras.Model):  #@save
    """The Residual block of ResNet."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)
```

This code generates two types of networks: one where we add the input to the output before applying the ReLU nonlinearity whenever `use_1x1conv=False`, and one where we adjust channels and resolution by means of a $1 \times 1$ convolution before adding. :numref:`fig_resnet_block` illustrates this:

![ResNet block with and without $1 \times 1$ convolution.](../img/resnet-block.svg)
:label:`fig_resnet_block`

Now let us look at a situation where the input and output are of the same shape.

Este código gera dois tipos de redes: uma onde adicionamos a entrada à saída antes de aplicar a não linearidade ReLU sempre que `use_1x1conv = False`, e outra onde ajustamos os canais e a resolução por meio de uma convolução $ 1 \ vezes 1 $ antes de adicionar. : numref: `fig_resnet_block` ilustra isso:

! [Bloco ResNet com e sem $ 1 \ vezes 1 $ convolução.] (../ img / resnet-block.svg)
: label: `fig_resnet_block`

Agora, vejamos uma situação em que a entrada e a saída têm a mesma forma.

```{.python .input}
blk = Residual(3)
blk.initialize()
X = np.random.uniform(size=(4, 3, 6, 6))
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(3)
X = tf.random.uniform((4, 6, 6, 3))
Y = blk(X)
Y.shape
```

We also have the option to halve the output height and width while increasing the number of output channels.

Também temos a opção de reduzir pela metade a altura e largura de saída, aumentando o número de canais de saída.

```{.python .input}
blk = Residual(6, use_1x1conv=True, strides=2)
blk.initialize()
blk(X).shape
```

```{.python .input}
#@tab pytorch
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape
```

```{.python .input}
#@tab tensorflow
blk = Residual(6, use_1x1conv=True, strides=2)
blk(X).shape
```

## ResNet Model

The first two layers of ResNet are the same as those of the GoogLeNet we described before: the $7\times 7$ convolutional layer with 64 output channels and a stride of 2 is followed by the $3\times 3$ maximum pooling layer with a stride of 2. The difference is the batch normalization layer added after each convolutional layer in ResNet.

As duas primeiras camadas do ResNet são iguais às do GoogLeNet que descrevemos antes: a camada convolucional $ 7 \ times 7 $ com 64 canais de saída e uma passada de 2 é seguida pela camada de pooling máxima $ 3 \ times 3 $ com uma passada de 2. A diferença é a camada de normalização de lote adicionada após cada camada convolucional no ResNet.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
b1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

GoogLeNet uses four modules made up of Inception blocks.
However, ResNet uses four modules made up of residual blocks, each of which uses several residual blocks with the same number of output channels. 
The number of channels in the first module is the same as the number of input channels. Since a maximum pooling layer with a stride of 2 has already been used, it is not necessary to reduce the height and width. In the first residual block for each of the subsequent modules, the number of channels is doubled compared with that of the previous module, and the height and width are halved.

Now, we implement this module. Note that special processing has been performed on the first module.

GoogLeNet usa quatro módulos compostos de blocos de iniciação.
No entanto, o ResNet usa quatro módulos compostos de blocos residuais, cada um dos quais usa vários blocos residuais com o mesmo número de canais de saída.
O número de canais no primeiro módulo é igual ao número de canais de entrada. Como uma camada de pooling máxima com uma passada de 2 já foi usada, não é necessário reduzir a altura e a largura. No primeiro bloco residual para cada um dos módulos subsequentes, o número de canais é duplicado em comparação com o do módulo anterior e a altura e a largura são reduzidas à metade.

Agora, implementamos este módulo. Observe que o processamento especial foi executado no primeiro módulo.

```{.python .input}
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk
```

```{.python .input}
#@tab pytorch
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
```

```{.python .input}
#@tab tensorflow
class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False,
                 **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X
```

Then, we add all the modules to ResNet. Here, two residual blocks are used for each module.

Em seguida, adicionamos todos os módulos ao ResNet. Aqui, dois blocos residuais são usados para cada módulo.

```{.python .input}
net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
```

```{.python .input}
#@tab pytorch
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
```

```{.python .input}
#@tab tensorflow
b2 = ResnetBlock(64, 2, first_block=True)
b3 = ResnetBlock(128, 2)
b4 = ResnetBlock(256, 2)
b5 = ResnetBlock(512, 2)
```

Finally, just like GoogLeNet, we add a global average pooling layer, followed by the fully-connected layer output.

Finalmente, assim como GoogLeNet, adicionamos uma camada de pooling global média, seguida pela saída da camada totalmente conectada.

```{.python .input}
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
```

```{.python .input}
#@tab tensorflow
# Recall that we define this as a function so we can reuse later and run it
# within `tf.distribute.MirroredStrategy`'s scope to utilize various
# computational resources, e.g. GPUs. Also note that even though we have
# created b1, b2, b3, b4, b5 but we will recreate them inside this function's
# scope instead
def net():
    return tf.keras.Sequential([
        # The following layers are the same as b1 that we created earlier
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # The following layers are the same as b2, b3, b4, and b5 that we
        # created earlier
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=10)])
```

There are 4 convolutional layers in each module (excluding the $1\times 1$ convolutional layer). Together with the first $7\times 7$ convolutional layer and the final fully-connected layer, there are 18 layers in total. Therefore, this model is commonly known as ResNet-18.
By configuring different numbers of channels and residual blocks in the module, we can create different ResNet models, such as the deeper 152-layer ResNet-152. Although the main architecture of ResNet is similar to that of GoogLeNet, ResNet's structure is simpler and easier to modify. All these factors have resulted in the rapid and widespread use of ResNet. :numref:`fig_resnet18` depicts the full ResNet-18.

Existem 4 camadas convolucionais em cada módulo (excluindo a camada convolucional $ 1 \ vezes 1 $). Junto com a primeira camada convolucional $ 7 vezes 7 $ e a camada final totalmente conectada, há 18 camadas no total. Portanto, esse modelo é comumente conhecido como ResNet-18.
Configurando diferentes números de canais e blocos residuais no módulo, podemos criar diferentes modelos de ResNet, como o ResNet-152 de 152 camadas mais profundo. Embora a arquitetura principal do ResNet seja semelhante à do GoogLeNet, a estrutura do ResNet é mais simples e fácil de modificar. Todos esses fatores resultaram no uso rápido e generalizado da ResNet. : numref: `fig_resnet18` representa o ResNet-18 completo.

![The ResNet-18 architecture.](../img/resnet18.svg)
:label:`fig_resnet18`

Before training ResNet, let us observe how the input shape changes across different modules in ResNet. As in all the previous architectures, the resolution decreases while the number of channels increases up until the point where a global average pooling layer aggregates all features.

Antes de treinar o ResNet, vamos observar como a forma da entrada muda nos diferentes módulos do ResNet. Como em todas as arquiteturas anteriores, a resolução diminui enquanto o número de canais aumenta até o ponto em que uma camada de pooling média global agrega todos os recursos.

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform(shape=(1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

## Training

We train ResNet on the Fashion-MNIST dataset, just like before.

Treinamos ResNet no conjunto de dados Fashion-MNIST, assim como antes.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Summary

* Nested function classes are desirable. Learning an additional layer in deep neural networks as an identity function (though this is an extreme case) should be made easy.
* The residual mapping can learn the identity function more easily, such as pushing parameters in the weight layer to zero.
* We can train an effective deep neural network by having residual blocks. Inputs can forward propagate faster through the residual connections across layers.
* ResNet had a major influence on the design of subsequent deep neural networks, both for convolutional and sequential nature.

* As classes de funções aninhadas são desejáveis. Aprender uma camada adicional em redes neurais profundas como uma função de identidade (embora este seja um caso extremo) deve ser facilitado.
* O mapeamento residual pode aprender a função de identidade mais facilmente, como empurrar parâmetros na camada de peso para zero.
* Podemos treinar uma rede neural profunda eficaz tendo blocos residuais. As entradas podem se propagar para frente mais rápido através das conexões residuais entre as camadas.
* O ResNet teve uma grande influência no projeto de redes neurais profundas subsequentes, tanto de natureza convolucional quanto sequencial.


## Exercises

1. What are the major differences between the Inception block in :numref:`fig_inception` and the residual block? After removing some paths in the Inception block, how are they related to each other?
1. Refer to Table 1 in the ResNet paper :cite:`He.Zhang.Ren.ea.2016` to
   implement different variants.
1. For deeper networks, ResNet introduces a "bottleneck" architecture to reduce
   model complexity. Try to implement it.
1. In subsequent versions of ResNet, the authors changed the "convolution, batch
   normalization, and activation" structure to the "batch normalization,
   activation, and convolution" structure. Make this improvement
   yourself. See Figure 1 in :cite:`He.Zhang.Ren.ea.2016*1`
   for details.
1. Why can't we just increase the complexity of functions without bound, even if the function classes are nested?

1. Quais são as principais diferenças entre o bloco de iniciação em: numref: `fig_inception` e o bloco residual? Depois de remover alguns caminhos no bloco de Iniciação, como eles se relacionam?
1. Consulte a Tabela 1 no artigo ResNet: cite: `He.Zhang.Ren.ea.2016` para
    implementar variantes diferentes.
1. Para redes mais profundas, a ResNet apresenta uma arquitetura de "gargalo" para reduzir
    complexidade do modelo. Tente implementá-lo.
1. Nas versões subsequentes do ResNet, os autores alteraram a configuração "convolução, lote
    normalização e ativação "estrutura para a" normalização em lote,
    estrutura de ativação e convolução ". Faça esta melhoria
    você mesmo. Veja a Figura 1 em: cite: `He.Zhang.Ren.ea.2016 * 1`
    para detalhes.
1. Por que não podemos simplesmente aumentar a complexidade das funções sem limites, mesmo se as classes de função estiverem aninhadas?
:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/85)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/86)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/333)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTczNjAyMzg0NCwxNDM0MDMxMTI1LDM3Nz
U4NTU0M119
-->
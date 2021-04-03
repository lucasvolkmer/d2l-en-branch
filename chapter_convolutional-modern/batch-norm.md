# Normalização de Lotes
:label:`sec_batch_norm`


Treinar redes neurais profundas é difícil.
E fazer com que eles convirjam em um período de tempo razoável pode ser complicado.
Nesta seção, descrevemos a *normalização em lote*, uma técnica popular e eficaz
que acelera consistentemente a convergência de redes profundas :cite:`Ioffe.Szegedy.2015`.
Juntamente com os blocos residuais --- cobertos posteriormente em :numref:`sec_resnet` --- normalização em lote
tornou possível para os praticantes
para treinar rotineiramente redes com mais de 100 camadas.



## Treinando Redes Profundas

Para motivar a normalização do lote, vamos revisar
alguns desafios práticos que surgem
ao treinar modelos de aprendizado de máquina e redes neurais em particular.

Em primeiro lugar, as escolhas relativas ao pré-processamento de dados costumam fazer uma enorme diferença nos resultados finais.
Lembre-se de nossa aplicação de MLPs para prever preços de casas (:numref:`sec_kaggle_house`).
Nosso primeiro passo ao trabalhar com dados reais
era padronizar nossos recursos de entrada
para cada um tem uma média de zero e variância de um.
Intuitivamente, essa padronização funciona bem com nossos otimizadores
porque coloca os parâmetros *a priori* em uma escala semelhante.

Em segundo lugar, para um MLP ou CNN típico, enquanto treinamos,
as variáveis (por exemplo, saídas de transformação afim em MLP)
em camadas intermediárias
podem assumir valores com magnitudes amplamente variáveis:
tanto ao longo das camadas da entrada até a saída, através das unidades na mesma camada,
e ao longo do tempo devido às nossas atualizações nos parâmetros do modelo.
Os inventores da normalização de lote postularam informalmente
que esse desvio na distribuição de tais variáveis poderia dificultar a convergência da rede.
Intuitivamente, podemos conjeturar que se um
camada tem valores variáveis que são 100 vezes maiores que os de outra camada,
isso pode exigir ajustes compensatórios nas taxas de aprendizagem.
   
Terceiro, redes mais profundas são complexas e facilmente capazes de overfitting.
Isso significa que a regularização se torna mais crítica.

A normalização em lote é aplicada a camadas individuais
(opcionalmente, para todos eles) e funciona da seguinte forma:
Em cada iteração de treinamento,
primeiro normalizamos as entradas (de normalização em lote)
subtraindo sua média e
dividindo por seu desvio padrão,
onde ambos são estimados com base nas estatísticas do minibatch atual.
Em seguida, aplicamos um coeficiente de escala e um deslocamento de escala.
É precisamente devido a esta *normalização* baseada em estatísticas *batch*
que *normalização de lote* deriva seu nome.

Observe que se tentamos aplicar a normalização de lote com minibatches de tamanho 1,
não seríamos capazes de aprender nada.
Isso porque depois de subtrair as médias,
cada unidade oculta teria valor 0!
Como você pode imaginar, uma vez que estamos dedicando uma seção inteira à normalização em lote,
com minibatches grandes o suficiente, a abordagem se mostra eficaz e estável.
Uma lição aqui é que, ao aplicar a normalização em lote,
a escolha do tamanho do lote pode ser
ainda mais significativo do que sem normalização em lote.

Formalmente, denotando por $\mathbf{x} \in \mathcal{B}$ uma entrada para normalização em lote ($\mathrm{BN}$)
que é de um minibatch $\mathcal{B}$,
a normalização em lote transforma $\mathbf{x}$
de acordo com a seguinte expressão:

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

Em :eqref:`eq_batchnorm`, 
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ é a média da amostra
e $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ é o desvio padrão da amostra do minibatch $\mathcal{B}$.
Depois de aplicar a padronização,
o minibatch resultante
tem média zero e variância unitária.
Porque a escolha da variação da unidade
(vs. algum outro número mágico) é uma escolha arbitrária,
normalmente incluímos elemento a elemento
*parâmetro de escala* $\boldsymbol{\gamma}$ e *parâmetro de deslocamento* $\boldsymbol{\beta}$
que tem a mesma forma de $\mathbf{x}$.
Observe que $\boldsymbol{\gamma}$ e$\boldsymbol{\beta}$ são
  parâmetros que precisam ser aprendidos em conjunto com os outros parâmetros do modelo.

Consequentemente, as magnitudes variáveis
para camadas intermediárias não pode divergir durante o treinamento
porque a normalização em lote centra ativamente e os redimensiona de volta
para uma dada média e tamanho (via $\hat{\boldsymbol{\mu}}_\mathcal{B}$ e ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$)
Um pedaço da intuição ou sabedoria do praticante
é que a normalização em lote parece permitir taxas de aprendizagem mais agressivas.

Formalmente, calculamos $\hat{\boldsymbol{\mu}}_\mathcal{B}$ e${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ em :eqref:`eq_batchnorm` como a seguir:

$$\begin{aligned} \hat{\boldsymbol{\mu}}_\mathcal{B} &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x},\\
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 &= \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.\end{aligned}$$

Observe que adicionamos uma pequena constante $\epsilon > 0$
para a estimativa de variância
para garantir que nunca tentemos a divisão por zero,
mesmo nos casos em que a estimativa de variância empírica pode desaparecer.
As estimativas $\hat{\boldsymbol{\mu}}_\mathcal{B}$ e ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ neutralizam o problema de escala
usando estimativas ruidosas de média e variância.
Você pode pensar que esse barulho deve ser um problema.
Acontece que isso é realmente benéfico.

This turns out to be a recurring theme in deep learning.
For reasons that are not yet well-characterized theoretically,
various sources of noise in optimization
often lead to faster training and less overfitting:
this variation appears to act as a form of regularization.
In some preliminary research,
:cite:`Teye.Azizpour.Smith.2018` and :cite:`Luo.Wang.Shao.ea.2018`
relate the properties of batch normalization to Bayesian priors and penalties respectively.
In particular, this sheds some light on the puzzle
of why batch normalization works best for moderate minibatches sizes in the $50 \sim 100$ range.

Esse é um tema recorrente no aprendizado profundo.
Por razões que ainda não são bem caracterizadas teoricamente,
várias fontes de ruído na otimização
muitas vezes levam a um treinamento mais rápido e menor sobreajuste:
essa variação parece atuar como uma forma de regularização.
Em algumas pesquisas preliminares, :cite:`Teye.Azizpour.Smith.2018` e :cite:`Luo.Wang.Shao.ea.2018`
relacionam as propriedades de normalização de lote aos antecedentes e penalidades Bayesianas, respectivamente.
Em particular, isso lança alguma luz sobre o quebra-cabeça
de por que a normalização em lote funciona melhor para tamanhos moderados de minibatches na faixa de $50 \sim 100$.

Fixing a trained model, you might think
that we would prefer using the entire dataset
to estimate the mean and variance.
Once training is complete, why would we want
the same image to be classified differently,
depending on the batch in which it happens to reside?
During training, such exact calculation is infeasible
because the intermediate variables
for all data examples
change every time we update our model.
However, once the model is trained,
we can calculate the means and variances
of each layer's variables based on the entire dataset.
Indeed this is standard practice for
models employing batch normalization
and thus batch normalization layers function differently
in *training mode* (normalizing by minibatch statistics)
and in *prediction mode* (normalizing by dataset statistics).

We are now ready to take a look at how batch normalization works in practice.


Consertando um modelo treinado, você pode pensar
que preferiríamos usar todo o conjunto de dados
para estimar a média e a variância.
Depois que o treinamento for concluído, por que desejaríamos
a mesma imagem a ser classificada de forma diferente,
dependendo do lote em que reside?
Durante o treinamento, esse cálculo exato é inviável
porque as variáveis intermediárias
para todos os exemplos de dados
mudar toda vez que atualizamos nosso modelo.
No entanto, uma vez que o modelo é treinado,
podemos calcular as médias e variações
das variáveis de cada camada com base em todo o conjunto de dados.
Na verdade, esta é uma prática padrão para
modelos que empregam normalização em lote
e, portanto, as camadas de normalização em lote funcionam de maneira diferente
no * modo de treinamento * (normalizando por estatísticas de minibatch)
e no * modo de previsão * (normalizando por estatísticas do conjunto de dados).

Agora estamos prontos para dar uma olhada em como a normalização em lote funciona na prática.

## Batch Normalization Layers

Batch normalization implementations for fully-connected layers
and convolutional layers are slightly different.
We discuss both cases below.
Recall that one key differences between batch normalization and other layers
is that because batch normalization operates on a full minibatch at a time,
we cannot just ignore the batch dimension
as we did before when introducing other layers.

Implementações de normalização em lote para camadas totalmente conectadas
e as camadas convolucionais são ligeiramente diferentes.
Discutimos ambos os casos abaixo.
Lembre-se de que uma diferença fundamental entre a normalização em lote e outras camadas
é que porque a normalização de lote opera em um minibatch completo por vez,
não podemos simplesmente ignorar a dimensão do lote
como fizemos antes ao introduzir outras camadas.

### Fully-Connected Layers

When applying batch normalization to fully-connected layers,
the original paper inserts batch normalization after the affine transformation
and before the nonlinear activation function (later applications may insert batch normalization right after activation functions) :cite:`Ioffe.Szegedy.2015`.
Denoting the input to the fully-connected layer by $\mathbf{x}$,
the affine transformation
by $\mathbf{W}\mathbf{x} + \mathbf{b}$ (with the weight parameter $\mathbf{W}$ and the bias parameter $\mathbf{b}$),
and the activation function by $\phi$,
we can express the computation of a batch-normalization-enabled,
fully-connected layer output $\mathbf{h}$ as follows:

Ao aplicar a normalização de lote a camadas totalmente conectadas,
o papel original insere a normalização do lote após a transformação afim
e antes da função de ativação não linear (aplicativos posteriores podem inserir a normalização em lote logo após as funções de ativação): cite: `Ioffe.Szegedy.2015`.
Denotando a entrada para a camada totalmente conectada por $ \ mathbf {x} $,
a transformação afim
por $ \ mathbf {W} \ mathbf {x} + \ mathbf {b} $ (com o parâmetro de peso $ \ mathbf {W} $ e o parâmetro de polarização $ \ mathbf {b} $),
e a função de ativação por $ \ phi $,
podemos expressar o cálculo de uma normalização de lote habilitada,
saída de camada totalmente conectada $ \ mathbf {h} $ como segue:

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Recall that mean and variance are computed
on the *same* minibatch 
on which the transformation is applied.

Lembre-se de que a média e a variância são calculadas
no * mesmo * minibatch
no qual a transformação é aplicada.

### Convolutional Layers

Similarly, with convolutional layers,
we can apply batch normalization after the convolution
and before the nonlinear activation function.
When the convolution has multiple output channels,
we need to carry out batch normalization
for *each* of the outputs of these channels,
and each channel has its own scale and shift parameters,
both of which are scalars.
Assume that our minibatches contain $m$ examples
and that for each channel,
the output of the convolution has height $p$ and width $q$.
For convolutional layers, we carry out each batch normalization
over the $m \cdot p \cdot q$ elements per output channel simultaneously.
Thus, we collect the values over all spatial locations
when computing the mean and variance
and consequently 
apply the same mean and variance
within a given channel
to normalize the value at each spatial location.

Da mesma forma, com camadas convolucionais,
podemos aplicar a normalização de lote após a convolução
e antes da função de ativação não linear.
Quando a convolução tem vários canais de saída,
precisamos realizar a normalização do lote
para * cada * uma das saídas desses canais,
e cada canal tem sua própria escala e parâmetros de deslocamento,
ambos são escalares.
Suponha que nossos minibatches contenham $ m $ exemplos
e isso para cada canal,
a saída da convolução tem altura $ p $ e largura $ q $.
Para camadas convolucionais, realizamos a normalização de cada lote
nos elementos $ m \ cdot p \ cdot q $ por canal de saída simultaneamente.
Assim, coletamos os valores de todas as localizações espaciais
ao calcular a média e a variância
e consequentemente
aplique a mesma média e variância
dentro de um determinado canal
para normalizar o valor em cada localização espacial.


### Batch Normalization During Prediction

As we mentioned earlier, batch normalization typically behaves differently
in training mode and prediction mode.
First, the noise in the sample mean and the sample variance
arising from estimating each on minibatches
are no longer desirable once we have trained the model.
Second, we might not have the luxury
of computing per-batch normalization statistics.
For example,
we might need to apply our model to make one prediction at a time.

Como mencionamos anteriormente, a normalização em lote normalmente se comporta de maneira diferente
no modo de treinamento e no modo de previsão.
Primeiro, o ruído na média da amostra e a variância da amostra
decorrentes da estimativa de cada um em minibatches
não são mais desejáveis, uma vez que treinamos o modelo.
Em segundo lugar, podemos não ter o luxo
de computação de estatísticas de normalização por lote.
Por exemplo,
podemos precisar aplicar nosso modelo para fazer uma previsão de cada vez.

Typically, after training, we use the entire dataset
to compute stable estimates of the variable statistics
and then fix them at prediction time.
Consequently, batch normalization behaves differently during training and at test time.
Recall that dropout also exhibits this characteristic.

Normalmente, após o treinamento, usamos todo o conjunto de dados
para calcular estimativas estáveis das estatísticas variáveis
e, em seguida, corrigi-los no momento da previsão.
Conseqüentemente, a normalização do lote se comporta de maneira diferente durante o treinamento e no momento do teste.
Lembre-se de que o abandono também exibe essa característica.

## Implementation from Scratch

Below, we implement a batch normalization layer with tensors from scratch.

Abaixo, implementamos uma camada de normalização em lote com tensores do zero.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `autograd` to determine whether the current mode is training mode or
    # prediction mode
    if not autograd.is_training():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully-connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

We can now create a proper `BatchNorm` layer.
Our layer will maintain proper parameters
for scale `gamma` and shift `beta`,
both of which will be updated in the course of training.
Additionally, our layer will maintain
moving averages of the means and variances
for subsequent use during model prediction.

Agora podemos criar uma camada `BatchNorm` adequada.
Nossa camada manterá os parâmetros adequados
para escala `gama` e deslocamento` beta`,
ambos serão atualizados no decorrer do treinamento.
Além disso, nossa camada manterá
médias móveis das médias e variâncias
para uso subsequente durante a previsão do modelo.

Putting aside the algorithmic details,
note the design pattern underlying our implementation of the layer.
Typically, we define the mathematics in a separate function, say `batch_norm`.
We then integrate this functionality into a custom layer,
whose code mostly addresses bookkeeping matters,
such as moving data to the right device context,
allocating and initializing any required variables,
keeping track of moving averages (here for mean and variance), and so on.
This pattern enables a clean separation of mathematics from boilerplate code.
Also note that for the sake of convenience
we did not worry about automatically inferring the input shape here,
thus we need to specify the number of features throughout.
Do not worry, the high-level batch normalization APIs in the deep learning framework will care of this for us and we will demonstrate that later.

Deixando de lado os detalhes algorítmicos,
observe o padrão de design subjacente à nossa implementação da camada.
Normalmente, definimos a matemática em uma função separada, digamos `batch_norm`.
Em seguida, integramos essa funcionalidade em uma camada personalizada,
cujo código aborda principalmente questões de contabilidade,
como mover dados para o contexto certo do dispositivo,
alocar e inicializar quaisquer variáveis necessárias,
acompanhar as médias móveis (aqui para média e variância) e assim por diante.
Esse padrão permite uma separação clara da matemática do código clichê.
Observe também que por uma questão de conveniência
não nos preocupamos em inferir automaticamente a forma de entrada aqui,
portanto, precisamos especificar o número de recursos por toda parte.
Não se preocupe, as APIs de normalização de lote de alto nível na estrutura de aprendizado profundo cuidarão disso para nós e iremos demonstrar isso mais tarde.

```{.python .input}
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.python .input}
#@tab pytorch
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

```{.python .input}
#@tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

## Applying Batch Normalization in LeNet

To see how to apply `BatchNorm` in context,
below we apply it to a traditional LeNet model (:numref:`sec_lenet`).
Recall that batch normalization is applied
after the convolutional layers or fully-connected layers
but before the corresponding activation functions.

Para ver como aplicar `BatchNorm` no contexto,
abaixo nós o aplicamos a um modelo LeNet tradicional (: numref: `sec_lenet`).
Lembre-se de que a normalização de lote é aplicada
após as camadas convolucionais ou camadas totalmente conectadas
mas antes das funções de ativação correspondentes.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
# Recall that this has to be a function that will be passed to `d2l.train_ch6`
# so that model building or compiling need to be within `strategy.scope()` in
# order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )
```

As before, we will train our network on the Fashion-MNIST dataset.
This code is virtually identical to that when we first trained LeNet (:numref:`sec_lenet`).
The main difference is the considerably larger learning rate.

Como antes, treinaremos nossa rede no conjunto de dados Fashion-MNIST.
Este código é virtualmente idêntico àquele quando treinamos o LeNet pela primeira vez (: numref: `sec_lenet`).
A principal diferença é a taxa de aprendizagem consideravelmente maior.

```{.python .input}
#@tab mxnet, pytorch
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.python .input}
#@tab tensorflow
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

Let us have a look at the scale parameter `gamma`
and the shift parameter `beta` learned
from the first batch normalization layer.

Vamos dar uma olhada no parâmetro de escala `gamma`
e o parâmetro shift `beta` aprendeu
da primeira camada de normalização de lote.

```{.python .input}
net[1].gamma.data().reshape(-1,), net[1].beta.data().reshape(-1,)
```

```{.python .input}
#@tab pytorch
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

```{.python .input}
#@tab tensorflow
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
```

## Concise Implementation

Compared with the `BatchNorm` class,
which we just defined ourselves,
we can use the `BatchNorm` class defined in high-level APIs from the deep learning framework directly.
The code looks virtually identical
to the application our implementation above.

Comparado com a classe `BatchNorm`,
que acabamos de definir a nós mesmos,
podemos usar a classe `BatchNorm` definida em APIs de alto nível diretamente do framework de aprendizado profundo.
O código parece virtualmente idêntico
para a aplicação nossa implementação acima.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
```

Below, we use the same hyperparameters to train our model.
Note that as usual, the high-level API variant runs much faster
because its code has been compiled to C++ or CUDA
while our custom implementation must be interpreted by Python.

Abaixo, usamos os mesmos hiperparâmetros para treinar nosso modelo.
Observe que, como de costume, a variante de API de alto nível é executada muito mais rápido
porque seu código foi compilado para C ++ ou CUDA
enquanto nossa implementação customizada deve ser interpretada por Python.

```{.python .input}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Controversy

Intuitively, batch normalization is thought
to make the optimization landscape smoother.
However, we must be careful to distinguish between
speculative intuitions and true explanations
for the phenomena that we observe when training deep models.
Recall that we do not even know why simpler
deep neural networks (MLPs and conventional CNNs)
generalize well in the first place.
Even with dropout and weight decay,
they remain so flexible that their ability to generalize to unseen data
cannot be explained via conventional learning-theoretic generalization guarantees.

Intuitivamente, a normalização de lote é pensada
para tornar o cenário de otimização mais suave.
No entanto, devemos ter o cuidado de distinguir entre
intuições especulativas e explicações verdadeiras
para os fenômenos que observamos ao treinar modelos profundos.
Lembre-se que não sabemos nem porque é mais simples
redes neurais profundas (MLPs e CNNs convencionais)
generalize bem em primeiro lugar.
Mesmo com abandono e queda de peso,
eles permanecem tão flexíveis que sua capacidade de generalizar para dados invisíveis
não pode ser explicado por meio de garantias convencionais de generalização da teoria da aprendizagem.

In the original paper proposing batch normalization,
the authors, in addition to introducing a powerful and useful tool,
offered an explanation for why it works:
by reducing *internal covariate shift*.
Presumably by *internal covariate shift* the authors
meant something like the intuition expressed above---the
notion that the distribution of variable values changes
over the course of training.
However, there were two problems with this explanation:
i) This drift is very different from *covariate shift*,
rendering the name a misnomer.
ii) The explanation offers an under-specified intuition
but leaves the question of *why precisely this technique works*
an open question wanting for a rigorous explanation.
Throughout this book, we aim to convey the intuitions that practitioners
use to guide their development of deep neural networks.
However, we believe that it is important
to separate these guiding intuitions
from established scientific fact.
Eventually, when you master this material
and start writing your own research papers
you will want to be clear to delineate
between technical claims and hunches.

No artigo original, propondo a normalização do lote,
os autores, além de apresentar uma ferramenta poderosa e útil,
ofereceu uma explicação de por que funciona:
reduzindo * deslocamento interno da covariável *.
Presumivelmente por * mudança interna de covariável * os autores
significava algo como a intuição expressa acima --- o
noção de que a distribuição dos valores das variáveis ​​muda
ao longo do treinamento.
No entanto, houve dois problemas com esta explicação:
i) Esta deriva é muito diferente de * mudança de covariável *,
tornando o nome um nome impróprio.
ii) A explicação oferece uma intuição subespecificada
mas deixa a questão de * por que exatamente essa técnica funciona *
uma questão aberta que necessita de uma explicação rigorosa.
Ao longo deste livro, pretendemos transmitir as intuições de que os praticantes
usar para orientar o desenvolvimento de redes neurais profundas.
No entanto, acreditamos que é importante
para separar essas intuições orientadoras
a partir de fato científico estabelecido.
Eventualmente, quando você dominar este material
e comece a escrever seus próprios artigos de pesquisa
você vai querer ser claro para delinear
entre afirmações técnicas e palpites.

Following the success of batch normalization,
its explanation in terms of *internal covariate shift*
has repeatedly surfaced in debates in the technical literature
and broader discourse about how to present machine learning research.
In a memorable speech given while accepting a Test of Time Award
at the 2017 NeurIPS conference,
Ali Rahimi used *internal covariate shift*
as a focal point in an argument likening
the modern practice of deep learning to alchemy.
Subsequently, the example was revisited in detail
in a position paper outlining
troubling trends in machine learning :cite:`Lipton.Steinhardt.2018`.
Other authors
have proposed alternative explanations for the success of batch normalization,
some claiming that batch normalization's success comes despite exhibiting behavior
that is in some ways opposite to those claimed in the original paper :cite:`Santurkar.Tsipras.Ilyas.ea.2018`.

Após o sucesso da normalização em lote,
sua explicação em termos de * mudança interna de covariável *
tem aparecido repetidamente em debates na literatura técnica
e um discurso mais amplo sobre como apresentar a pesquisa de aprendizado de máquina.
Em um discurso memorável ao aceitar o Prêmio Teste do Tempo
na conferência NeurIPS de 2017,
Ali Rahimi usou * mudança de covariável interna *
como um ponto focal em um argumento comparando
a prática moderna de aprendizado profundo para alquimia.
Posteriormente, o exemplo foi revisitado em detalhes
em uma posição delineando papel
tendências preocupantes em aprendizado de máquina: cite: `Lipton.Steinhardt.2018`.
Outros autores
propuseram explicações alternativas para o sucesso da normalização em lote,
alguns alegando que o sucesso da normalização em lote vem apesar de exibir comportamento
isso é, de certa forma, oposto ao afirmado no artigo original: cite: `Santurkar.Tsipras.Ilyas.ea.2018`.

We note that the *internal covariate shift*
is no more worthy of criticism than any of
thousands of similarly vague claims
made every year in the technical machine learning literature.
Likely, its resonance as a focal point of these debates
owes to its broad recognizability to the target audience.
Batch normalization has proven an indispensable method,
applied in nearly all deployed image classifiers,
earning the paper that introduced the technique
tens of thousands of citations.

Notamos que a * mudança interna da covariável *
não é mais digno de crítica do que qualquer um dos
milhares de afirmações igualmente vagas
feito todos os anos na literatura técnica de aprendizado de máquina.
Provavelmente, sua ressonância como ponto focal desses debates
deve-se ao seu amplo reconhecimento pelo público-alvo.
A normalização em lote provou ser um método indispensável,
aplicado em quase todos os classificadores de imagem implantados,
ganhando o papel que introduziu a técnica
dezenas de milhares de citações.


## Summary

* During model training, batch normalization continuously adjusts the intermediate output of the neural network by utilizing the mean and standard deviation of the minibatch, so that the values of the intermediate output in each layer throughout the neural network are more stable.
* The batch normalization methods for fully-connected layers and convolutional layers are slightly different.
* Like a dropout layer, batch normalization layers have different computation results in training mode and prediction mode.
* Batch normalization has many beneficial side effects, primarily that of regularization. On the other hand, the original motivation of reducing internal covariate shift seems not to be a valid explanation.

* Durante o treinamento do modelo, a normalização em lote ajusta continuamente a saída intermediária da rede neural, utilizando a média e o desvio padrão do minibatch, de modo que os valores da saída intermediária em cada camada em toda a rede neural sejam mais estáveis.
* Os métodos de normalização de lote para camadas totalmente conectadas e camadas convolucionais são ligeiramente diferentes.
* Como uma camada de eliminação, as camadas de normalização em lote têm resultados de computação diferentes no modo de treinamento e no modo de previsão.
* A normalização em lote tem muitos efeitos colaterais benéficos, principalmente o da regularização. Por outro lado, a motivação original de reduzir a mudança interna da covariável parece não ser uma explicação válida.

## Exercises

1. Can we remove the bias parameter from the fully-connected layer or the convolutional layer before the batch normalization? Why?
1. Compare the learning rates for LeNet with and without batch normalization.
    1. Plot the increase in training and test accuracy.
    1. How large can you make the learning rate?
1. Do we need batch normalization in every layer? Experiment with it?
1. Can you replace dropout by batch normalization? How does the behavior change?
1. Fix the parameters `beta` and `gamma`, and observe and analyze the results.
1. Review the online documentation for `BatchNorm` from the high-level APIs to see the other applications for batch normalization.
1. Research ideas: think of other normalization transforms that you can apply? Can you apply the probability integral transform? How about a full rank covariance estimate?

1. Podemos remover o parâmetro de polarização da camada totalmente conectada ou da camada convolucional antes da normalização do lote? Porque?
1. Compare as taxas de aprendizagem para LeNet com e sem normalização de lote.
     1. Plote o aumento na precisão do treinamento e do teste.
     1. Quão grande você pode aumentar a taxa de aprendizado?
1. Precisamos de normalização de lote em cada camada? Experimentar?
1. Você pode substituir o abandono pela normalização em lote? Como o comportamento muda?
1. Fixe os parâmetros `beta` e` gamma`, observe e analise os resultados.
1. Revise a documentação online para `BatchNorm` das APIs de alto nível para ver os outros aplicativos para normalização de lote.
1. Ideias de pesquisa: pense em outras transformações de normalização que você pode aplicar? Você pode aplicar a transformação integral de probabilidade? Que tal uma estimativa de covariância de classificação completa?
:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTIwNDA2NDQ0NywtMTc3MTg1OTg1NiwtNz
QxNzQ0NjI2LC03NDkxNDUwNjQsLTEzMjE5ODI1MjYsLTE5NTc3
OTQzMzYsOTc5NjE3MzE4XX0=
-->
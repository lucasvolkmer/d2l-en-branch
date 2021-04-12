# *Pooling* de Atenção: Regressão de Kernel de Nadaraya-Watson 
:label:`sec_nadaraya-waston`

Agora você conhece os principais componentes dos mecanismos de atenção sob a estrutura em :numref:`fig_qkv`.
Para recapitular,
as interações entre
consultas (dicas volitivas) e chaves (dicas não volitivas)
resultam em *concentração de atenção*.
O *pooling* de atenção agrega valores seletivamente (entradas sensoriais) para produzir a saída.
Nesta secção,
vamos descrever o agrupamento de atenção em mais detalhes
para lhe dar uma visão de alto nível
como os mecanismos de atenção funcionam na prática.
Especificamente,
o modelo de regressão do kernel de Nadaraya-Watson
proposto em 1964
é um exemplo simples, mas completo
para demonstrar o aprendizado de máquina com mecanismos de atenção.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## Gerando o Dataset


Para manter as coisas simples,
vamos considerar o seguinte problema de regressão:
dado um conjunto de dados de pares de entrada-saída $\{(x_1, y_1), \ldots, (x_n, y_n)\}$,
como aprender $f$ para prever a saída $\hat{y} = f(x)$ para qualquer nova entrada $x$?

Aqui, geramos um conjunto de dados artificial de acordo com a seguinte função não linear com o termo de ruído $\epsilon$:

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$

onde $\epsilon$ obedece a uma distribuição normal com média zero e desvio padrão 0,5.
Ambos, 50 exemplos de treinamento e 50 exemplos de teste
são gerados.
Para visualizar melhor o padrão de atenção posteriormente, as entradas de treinamento são classificadas.

```{.python .input}
n_train = 50  # No. of training examples
x_train = np.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab pytorch
n_train = 50  # No. of training examples
x_train, _ = torch.sort(d2l.rand(n_train) * 5)   # Training inputs
```

```{.python .input}
#@tab all
def f(x):
    return 2 * d2l.sin(x) + x**0.8

y_train = f(x_train) + d2l.normal(0.0, 0.5, (n_train,))  # Training outputs
x_test = d2l.arange(0, 5, 0.1)  # Testing examples
y_truth = f(x_test)  # Ground-truth outputs for the testing examples
n_test = len(x_test)  # No. of testing examples
n_test
```

A função a seguir plota todos os exemplos de treinamento (representados por círculos),
a função de geração de dados de verdade básica `f` sem o termo de ruído (rotulado como "Truth"), e a função de predição aprendida (rotulado como "Pred").

```{.python .input}
#@tab all
def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## *Pooling* Médio

Começamos com talvez o estimador "mais idiota" do mundo para este problema de regressão:
usando o *pooling* médio para calcular a média de todos os resultados do treinamento:

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,$$
:eqlabel:`eq_avg-pooling`

que é plotado abaixo. Como podemos ver, este estimador não é tão inteligente.

```{.python .input}
y_hat = y_train.mean().repeat(n_test)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
plot_kernel_reg(y_hat)
```

## *Pooling* de Atenção não-Paramétrico

Obviamente,
o agrupamento médio omite as entradas $x_i$.
Uma ideia melhor foi proposta
por Nadaraya :cite:`Nadaraya.1964`
e Waston :cite:`Watson.1964`
para pesar as saídas $y_i$ de acordo com seus locais de entrada:

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$
:eqlabel:`eq_nadaraya-waston`

onde $K$ é um *kernel*.
O estimador em :eqref:`eq_nadaraya-waston`
é chamado de *regressão do kernel Nadaraya-Watson*.
Aqui não entraremos em detalhes sobre os grãos.
Lembre-se da estrutura dos mecanismos de atenção em :numref:`fig_qkv`.
Do ponto de vista da atenção,
podemos reescrever :eqref:`eq_nadaraya-waston`
em uma forma mais generalizada de *concentração de atenção*:

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,$$
:eqlabel:`eq_attn-pooling`



onde $x$ é a consulta e $(x_i, y_i)$ é o par de valores-chave.
Comparando :eqref:`eq_attn-pooling` e :eqref:`eq_avg-pooling`,
a atenção concentrada aqui
é uma média ponderada de valores $y_i$.
O *peso de atenção* $\alpha(x, x_i)$
em :eqref:`eq_attn-pooling`
é atribuído ao valor correspondente $y_i$
baseado na interação
entre a consulta $x$ e a chave $x_i$
modelado por $\alpha$.
Para qualquer consulta, seus pesos de atenção sobre todos os pares de valores-chave são uma distribuição de probabilidade válida: eles não são negativos e somam um.

Para obter intuições de concentração de atenção,
apenas considere um *kernel gaussiano* definido como

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).
$$


Conectando o kernel gaussiano em
:eqref:`eq_attn-pooling` e
:eqref:`eq_nadaraya-waston` dá

$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$$
:eqlabel:`eq_nadaraya-waston-gaussian`


In :eqref:`eq_nadaraya-waston-gaussian`,
uma chave $x_i$ que está mais próxima da consulta dada $x$ obterá
*mais atenção * por meio de um *peso de atenção maior* atribuído ao valor correspondente da chave $y_i$.

Notavelmente, a regressão do kernel Nadaraya-Watson é um modelo não paramétrico;
assim :eqref:`eq_nadaraya-waston-gaussian`
é um exemplo de *agrupamento de atenção não paramétrica*.
A seguir, traçamos a previsão com base neste
modelo de atenção não paramétrica.
A linha prevista é suave e mais próxima da verdade fundamental do que a produzida pelo agrupamento médio.

```{.python .input}
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = npx.softmax(-(X_repeat - x_train)**2 / 2)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `X_repeat`: (`n_test`, `n_train`), where each row contains the
# same testing inputs (i.e., same queries)
X_repeat = d2l.reshape(x_test.repeat_interleave(n_train), (-1, n_train))
# Note that `x_train` contains the keys. Shape of `attention_weights`:
# (`n_test`, `n_train`), where each row contains attention weights to be
# assigned among the values (`y_train`) given each query
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# Each element of `y_hat` is weighted average of values, where weights are
# attention weights
y_hat = d2l.matmul(attention_weights, y_train)
plot_kernel_reg(y_hat)
```

Agora, vamos dar uma olhada nos pesos de atenção.
Aqui, as entradas de teste são consultas, enquanto as entradas de treinamento são essenciais.
Uma vez que ambas as entradas são classificadas,
podemos ver que quanto mais próximo o par de chave de consulta está,
o maior peso de atenção está no *pooling* de atenção.

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## *Pooling* de Atenção Paramétrica


A regressão de kernel não paramétrica de Nadaraya-Watson
desfruta do benefício de *consistência*:
com dados suficientes, esse modelo converge para a solução ótima.
Não obstante,
podemos facilmente integrar parâmetros aprendíveis no *pooling* de atenção.

Por exemplo, um pouco diferente de :eqref:`eq_nadaraya-waston-gaussian`, na sequência
a distância entre a consulta $x$ e a chave $x_i$
é multiplicado por um parâmetro aprendível $w$:


$$\begin{aligned}f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_i)w)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.\end{aligned}$$
:eqlabel:`eq_nadaraya-waston-gaussian-para`

No resto da seção,
vamos treinar este modelo aprendendo o parâmetro de
a concentração de atenção em :eqref:`eq_nadaraya-waston-gaussian-para`.

### Multiplicação de matriz de lote
:label:`subsec_batch_dot`

To more efficiently compute attention
for minibatches,
we can leverage batch matrix multiplication utilities
provided by deep learning frameworks.


Suppose that the first minibatch contains $n$ matrices $\mathbf{X}_1, \ldots, \mathbf{X}_n$ of shape $a\times b$, and the second minibatch contains $n$ matrices $\mathbf{Y}_1, \ldots, \mathbf{Y}_n$ of shape $b\times c$. Their batch matrix multiplication
results in
$n$ matrices $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$ of shape $a\times c$. Therefore, given two tensors of shape ($n$, $a$, $b$) and ($n$, $b$, $c$), the shape of their batch matrix multiplication output is ($n$, $a$, $c$).

```{.python .input}
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
npx.batch_dot(X, Y).shape
```

```{.python .input}
#@tab pytorch
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
torch.bmm(X, Y).shape
```

In the context of attention mechanisms, we can use minibatch matrix multiplication to compute weighted averages of values in a minibatch.

```{.python .input}
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20), (2, 10))
npx.batch_dot(np.expand_dims(weights, 1), np.expand_dims(values, -1))
```

```{.python .input}
#@tab pytorch
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20.0), (2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

### Defining the Model

Using minibatch matrix multiplication,
below we define the parametric version
of Nadaraya-Watson kernel regression
based on the parametric attention pooling in
:eqref:`eq_nadaraya-waston-gaussian-para`.

```{.python .input}
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = npx.softmax(
            -((queries - keys) * self.w.data())**2 / 2)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return npx.batch_dot(np.expand_dims(self.attention_weights, 1),
                             np.expand_dims(values, -1)).reshape(-1)
```

```{.python .input}
#@tab pytorch
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # Shape of the output `queries` and `attention_weights`:
        # (no. of queries, no. of key-value pairs)
        queries = d2l.reshape(
            queries.repeat_interleave(keys.shape[1]), (-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # Shape of `values`: (no. of queries, no. of key-value pairs)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)
```

### Training

In the following, we transform the training dataset
to keys and values to train the attention model.
In the parametric attention pooling,
any training input takes key-value pairs from all the training examples except for itself to predict its output.

```{.python .input}
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = np.tile(x_train, (n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = np.tile(y_train, (n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).astype('bool')],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).astype('bool')],
                     (n_train, -1))
```

```{.python .input}
#@tab pytorch
# Shape of `X_tile`: (`n_train`, `n_train`), where each column contains the
# same training inputs
X_tile = x_train.repeat((n_train, 1))
# Shape of `Y_tile`: (`n_train`, `n_train`), where each column contains the
# same training outputs
Y_tile = y_train.repeat((n_train, 1))
# Shape of `keys`: ('n_train', 'n_train' - 1)
keys = d2l.reshape(X_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                   (n_train, -1))
# Shape of `values`: ('n_train', 'n_train' - 1)
values = d2l.reshape(Y_tile[(1 - d2l.eye(n_train)).type(torch.bool)],
                     (n_train, -1))
```

Using the squared loss and stochastic gradient descent,
we train the parametric attention model.

```{.python .input}
net = NWKernelRegression()
net.initialize()
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    with autograd.record():
        l = loss(net(x_train, keys, values), y_train)
    l.backward()
    trainer.step(1)
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
#@tab pytorch
net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    trainer.zero_grad()
    # Note: L2 Loss = 1/2 * MSE Loss. PyTorch has MSE Loss which is slightly
    # different from MXNet's L2Loss by a factor of 2. Hence we halve the loss
    l = loss(net(x_train, keys, values), y_train) / 2
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

After training the parametric attention model,
we can plot its prediction.
Trying to fit the training dataset with noise,
the predicted line is less smooth
than its nonparametric counterpart that was plotted earlier.

```{.python .input}
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = np.tile(x_train, (n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = np.tile(y_train, (n_test, 1))
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```

```{.python .input}
#@tab pytorch
# Shape of `keys`: (`n_test`, `n_train`), where each column contains the same
# training inputs (i.e., same keys)
keys = x_train.repeat((n_test, 1))
# Shape of `value`: (`n_test`, `n_train`)
values = y_train.repeat((n_test, 1))
y_hat = net(x_test, keys, values).unsqueeze(1).detach()
plot_kernel_reg(y_hat)
```

Comparing with nonparametric attention pooling,
the region with large attention weights becomes sharper
in the learnable and parametric setting.

```{.python .input}
d2l.show_heatmaps(np.expand_dims(np.expand_dims(net.attention_weights, 0), 0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

```{.python .input}
#@tab pytorch
d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                  xlabel='Sorted training inputs',
                  ylabel='Sorted testing inputs')
```

## Summary

* Nadaraya-Watson kernel regression is an example of machine learning with attention mechanisms.
* The attention pooling of Nadaraya-Watson kernel regression is a weighted average of the training outputs. From the attention perspective, the attention weight is assigned to a value based on a function of a query and the key that is paired with the value.
* Attention pooling can be either nonparametric or parametric.


## Exercises

1. Increase the number of training examples. Can you learn  nonparametric Nadaraya-Watson kernel regression better?
1. What is the value of our learned $w$ in the parametric attention pooling experiment? Why does it make the weighted region sharper when visualizing the attention weights?
1. How can we add hyperparameters to nonparametric Nadaraya-Watson kernel regression to predict better?
1. Design another parametric attention pooling for the kernel regression of this section. Train this new model and visualize its attention weights.



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1599)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTM4NjUzODc5NCwtNTIwMDgwODk0LC00Nj
E1NDQ5MTksOTYxMzY5NDE1XX0=
-->
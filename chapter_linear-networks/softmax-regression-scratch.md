# Implementação da Regressão *Softmax*  do Zero
:label:`sec_softmax_scratch`

(**Assim como implementamos a regressão linear do zero, acreditamos que**)
regressão *softmax*
é igualmente fundamental e
(**você deve saber os detalhes sangrentos de**)
(~~*regressão softmax*~~)
como implementá-lo sozinho.
Vamos trabalhar com o *dataset* Fashion-MNIST, recém-introduzido em :numref:`sec_fashion_mnist`,
configurando um iterador de dados com *batch size* 256.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from IPython import display
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Inicializando os Parâmetros do Modelo


Como em nosso exemplo de regressão linear,
cada exemplo aqui será representado por um vetor de comprimento fixo.
Cada exemplo no conjunto de dados bruto é uma imagem $28 \times 28$.
Nesta seção, [**vamos nivelar cada imagem,
tratando-os como vetores de comprimento 784.**]
No futuro, falaremos sobre estratégias mais sofisticadas
para explorar a estrutura espacial em imagens,
mas, por enquanto, tratamos cada localização de pixel como apenas outro recurso.

Lembre-se de que na regressão *softmax*,
temos tantas saídas quanto classes.
(**Como nosso conjunto de dados tem 10 classes,
nossa rede terá uma dimensão de saída de 10.**)
Consequentemente, nossos pesos constituirão uma matriz $784 \times 10$
e os *bias* constituirão um vetor-linha $1 \times 10$.
Tal como acontece com a regressão linear, vamos inicializar nossos pesos `W`
com ruído Gaussiano e nossos *bias* com o valor inicial 0.
```{.python .input}
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                 mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))
```

## Definindo a Operação do *Softmax*

Antes de implementar o modelo de regressão do *softmax*,
vamos revisar brevemente como o operador de soma funciona
ao longo de dimensões específicas em um tensor,
conforme discutido em: numref :numref:`subseq_lin-alg-reduction` e :numref:`subseq_lin-alg-non-reduction`.
[**Dada uma matriz `X`, podemos somar todos os elementos (por padrão) ou apenas
sobre elementos no mesmo eixo,**]
ou seja, a mesma coluna (eixo 0) ou a mesma linha (eixo 1).
Observe que se `X` é um tensor com forma (2, 3)
e somamos as colunas,
o resultado será um vetor com forma (3,).
Ao invocar o operador de soma,
podemos especificar para manter o número de eixos no tensor original,
em vez de reduzir a dimensão que resumimos.
Isso resultará em um tensor bidimensional com forma (1, 3).

```{.python .input}
#@tab pytorch
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdim=True), d2l.reduce_sum(X, 1, keepdim=True)
```

```{.python .input}
#@tab mxnet, tensorflow
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

Agora estamos prontos para (**implementar a operação do *softmax* **).
Lembre-se de que o *softmax* consiste em três etapas:
i) exponenciamos cada termo (usando `exp`);
ii) somamos cada linha (temos uma linha por exemplo no lote)
para obter a constante de normalização para cada exemplo;
iii) dividimos cada linha por sua constante de normalização,
garantindo que o resultado seja 1.
Antes de olhar para o código, vamos lembrar
como isso parece, expresso como uma equação:

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

O denominador, ou constante de normalização,
às vezes também é chamada de *função de partição*
(e seu logaritmo é chamado de função de partição de log).
As origens desse nome estão em [física estatística](https://en.wikipedia.org/wiki/Partition_function_ (estatística_mecânica))
onde uma equação relacionada modela a distribuição
sobre um conjunto de partículas.

```{.python .input}
#@tab mxnet, tensorflow
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

```{.python .input}
#@tab pytorch
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

Como você pode ver, para qualquer entrada aleatória,
[**transformamos cada elemento em um número não negativo.
Além disso, cada linha soma 1,**]
como é necessário para uma probabilidade.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
#@tab tensorflow
X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, 1)
```

Observe que embora pareça correto matematicamente,
fomos um pouco desleixados em nossa implementação
porque falhamos em tomar precauções contra estouro numérico ou estouro negativo
devido a elementos grandes ou muito pequenos da matriz.

## Definindo o Modelo

Agora que definimos a operação do *softmax*,
podemos [**implementar o modelo de regressão softmax.**]
O código a seguir define como a entrada é mapeada para a saída por meio da rede.
Observe que achatamos cada imagem original no lote
em um vetor usando a função `reshape`
antes de passar os dados pelo nosso modelo.

```{.python .input}
#@tab all
def net(X):
    return softmax(d2l.matmul(d2l.reshape(X, (-1, W.shape[0])), W) + b)
```

## Definindo a Função de Perda


Em seguida, precisamos implementar a função de perda de entropia cruzada,
conforme apresentado em :numref:`sec_softmax`.
Esta pode ser a função de perda mais comum
em todo o *deep learning* porque, no momento,
os problemas de classificação superam em muito os problemas de regressão.

Lembre-se de que a entropia cruzada leva a *log-likelihood* negativa
da probabilidade prevista atribuída ao rótulo verdadeiro.
Em vez de iterar as previsões com um *loop for* Python
(que tende a ser ineficiente),
podemos escolher todos os elementos por um único operador.
Abaixo, nós [**criamos dados de amostra `y_hat`
com 2 exemplos de probabilidades previstas em 3 classes e seus rótulos correspondentes `y`.**]
Com `y` sabemos que no primeiro exemplo a primeira classe é a previsão correta e
no segundo exemplo, a terceira classe é a verdade fundamental.
[**Usando `y` como os índices das probabilidades em` y_hat`,**]
escolhemos a probabilidade da primeira classe no primeiro exemplo
e a probabilidade da terceira classe no segundo exemplo.

```{.python .input}
#@tab mxnet, pytorch
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

Agora podemos (**implementar a função de perda de entropia cruzada**) de forma eficiente com apenas uma linha de código.

```{.python .input}
#@tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

## Exatidão da Classificação


Dada a distribuição de probabilidade prevista `y_hat`,
normalmente escolhemos a classe com a maior probabilidade prevista
sempre que a previsão que devemos produzir é difícil.
Na verdade, muitos aplicativos exigem que façamos uma escolha.
O Gmail deve categorizar um e-mail em "Principal", "Social", "Atualizações" ou "Fóruns".
Pode estimar probabilidades internamente,
mas no final do dia ele tem que escolher uma das classes.

Quando as previsões são consistentes com a classe de *label* `y`, elas estão corretas.
A precisão da classificação é a fração de todas as previsões corretas.
Embora possa ser difícil otimizar a precisão diretamente (não é diferenciável),
muitas vezes é a medida de desempenho que mais nos preocupa,
e quase sempre o relataremos ao treinar classificadores.

Para calcular a precisão, fazemos o seguinte.
Primeiro, se `y_hat` é uma matriz,
presumimos que a segunda dimensão armazena pontuações de predição para cada classe.
Usamos `argmax` para obter a classe prevista pelo índice para a maior entrada em cada linha.
Em seguida, [**comparamos a classe prevista com a verdade fundamental `y` elemento a elemento.**]
Uma vez que o operador de igualdade `==` é sensível aos tipos de dados,
convertemos o tipo de dados de `y_hat` para corresponder ao de` y`.
O resultado é um tensor contendo entradas de 0 (falso) e 1 (verdadeiro).
Tirar a soma resulta no número de previsões corretas.

```{.python .input}
#@tab all
def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
```

We will continue to use the variables `y_hat` and `y`
defined before
as the predicted probability distributions and labels, respectively.
We can see that the first example's prediction class is 2
(the largest element of the row is 0.6 with the index 2),
which is inconsistent with the actual label, 0.
The second example's prediction class is 2
(the largest element of the row is 0.5 with the index of 2),
which is consistent with the actual label, 2.
Therefore, the classification accuracy rate for these two examples is 0.5.

```{.python .input}
#@tab all
accuracy(y_hat, y) / len(y)
```

[**Similarly, we can evaluate the accuracy for any model `net` on a dataset**]
that is accessed via the data iterator `data_iter`.

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

Here `Accumulator` is a utility class to accumulate sums over multiple variables.
In the above `evaluate_accuracy` function,
we create 2 variables in the `Accumulator` instance for storing both
the number of correct predictions and the number of predictions, respectively.
Both will be accumulated over time as we iterate over the dataset.

```{.python .input}
#@tab all
class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

[**Because we initialized the `net` model with random weights,
the accuracy of this model should be close to random guessing,**]
i.e., 0.1 for 10 classes.

```{.python .input}
#@tab all
evaluate_accuracy(net, test_iter)
```

## Training

[**The training loop**]
for softmax regression should look strikingly familiar
if you read through our implementation
of linear regression in :numref:`sec_linear_scratch`.
Here we refactor the implementation to make it reusable.
First, we define a function to train for one epoch.
Note that `updater` is a general function to update the model parameters,
which accepts the batch size as an argument.
It can be either a wrapper of the `d2l.sgd` function
or a framework's built-in optimization function.

```{.python .input}
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Train a model within one epoch (defined in Chapter 3)."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab pytorch
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y),
                       y.size().numel())
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab tensorflow
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Keras implementations for loss takes (labels, predictions)
            # instead of (predictions, labels) that users might implement
            # in this book, e.g. `cross_entropy` that we implemented above
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras loss by default returns the average loss in a batch
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

Before showing the implementation of the training function,
we define [**a utility class that plot data in animation.**]
Again, it aims to simplify code in the rest of the book.

```{.python .input}
#@tab all
class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

[~~The training function~~]
The following training function then
trains a model `net` on a training dataset accessed via `train_iter`
for multiple epochs, which is specified by `num_epochs`.
At the end of each epoch,
the model is evaluated on a testing dataset accessed via `test_iter`.
We will leverage the `Animator` class to visualize
the training progress.

```{.python .input}
#@tab all
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

As an implementation from scratch,
we [**use the minibatch stochastic gradient descent**] defined in :numref:`sec_linear_scratch`
to optimize the loss function of the model with a learning rate 0.1.

```{.python .input}
#@tab mxnet, pytorch
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

```{.python .input}
#@tab tensorflow
class Updater():  #@save
    """For updating parameters using minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

Now we [**train the model with 10 epochs.**]
Note that both the number of epochs (`num_epochs`),
and learning rate (`lr`) are adjustable hyperparameters.
By changing their values, we may be able
to increase the classification accuracy of the model.

```{.python .input}
#@tab all
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## Prediction

Now that training is complete,
our model is ready to [**classify some images.**]
Given a series of images,
we will compare their actual labels
(first line of text output)
and the predictions from the model
(second line of text output).

```{.python .input}
#@tab all
def predict_ch3(net, test_iter, n=6):  #@save
    """Predict labels (defined in Chapter 3)."""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## Summary

* With softmax regression, we can train models for multiclass classification.
* The training loop of softmax regression is very similar to that in linear regression: retrieve and read data, define models and loss functions, then train models using optimization algorithms. As you will soon find out, most common deep learning models have similar training procedures.

## Exercises

1. In this section, we directly implemented the softmax function based on the mathematical definition of the softmax operation. What problems might this cause? Hint: try to calculate the size of $\exp(50)$.
1. The function `cross_entropy` in this section was implemented according to the definition of the cross-entropy loss function.  What could be the problem with this implementation? Hint: consider the domain of the logarithm.
1. What solutions you can think of to fix the two problems above?
1. Is it always a good idea to return the most likely label? For example, would you do this for medical diagnosis?
1. Assume that we want to use softmax regression to predict the next word based on some features. What are some problems that might arise from a large vocabulary?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5NTMzMzIwMTksMTE1MzA4MTM1LDE2MD
A5ODQ0NTYsNzQyNTE3NDg3LC0yMTMxNTA1NDkzLC05OTkwNzg3
NjcsLTE5ODQxODk3MjVdfQ==
-->
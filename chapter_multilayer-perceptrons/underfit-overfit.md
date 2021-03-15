# Seleção do Modelo, *Underfitting*, e *Overfitting*
:label:`sec_model_selection`


Como cientistas de *machine learning*,
nosso objetivo é descobrir *padrões*.
Mas como podemos ter certeza de que
realmente descobrimos um padrão *geral*
e não simplesmente memorizamos nossos dados?
Por exemplo, imagine que queremos caçar
 padrões entre marcadores genéticos
ligando os pacientes ao seu estado de demência,
onde os rótulos são retirados do conjunto
$\{\text{dementia}, \text{mild cognitive impairment}, \text{healthy}\}$
Como os genes de cada pessoa os identificam de forma única
(ignorando irmãos idênticos),
é possível memorizar todo o conjunto de dados.

Não queremos que nosso modelo diga
*"É o Bob! Lembro-me dele! Ele tem demência!"*
O motivo é simples.
Quando implantamos o modelo no futuro,
nós encontraremos pacientes
que o modelo nunca viu antes.
Nossas previsões só serão úteis
se nosso modelo realmente descobriu um padrão *geral*.

Para recapitular mais formalmente,
nosso objetivo é descobrir padrões
que capturam regularidades na população subjacente
da qual nosso conjunto de treinamento foi extraído.
Se tivermos sucesso neste empreendimento,
então poderíamos avaliar com sucesso o risco
mesmo para indivíduos que nunca encontramos antes.
Este problema --- como descobrir padrões que *generalizam* --- é
o problema fundamental do *machine learning*.

O perigo é que, quando treinamos modelos,
acessamos apenas uma pequena amostra de dados.
Os maiores conjuntos de dados de imagens públicas contêm
cerca de um milhão de imagens.
Mais frequentemente, devemos aprender com apenas milhares
ou dezenas de milhares de exemplos de dados.
Em um grande sistema hospitalar, podemos acessar
centenas de milhares de registros médicos.
Ao trabalhar com amostras finitas, corremos o risco
de poder descobrir associações aparentes
que acabam não se sustentando quando coletamos mais dados.

O fenômeno de ajustar nossos dados de treinamento
mais precisamente do que ajustamos, a distribuição subjacente é chamada de *overfitting*, e as técnicas usadas para combater o *overfitting* são chamadas de *regularização*.
Nas seções anteriores, você deve ter observado
esse efeito durante a experiência com o conjunto de dados *Fashion-MNIST*.
Se você alterou a estrutura do modelo ou os hiperparâmetros durante o experimento, deve ter notado que, com neurônios, camadas e períodos de treinamento suficientes, o modelo pode eventualmente atingir uma precisão perfeita no conjunto de treinamento, mesmo quando a precisão dos dados de teste se deteriora.


## Erro de Treinamento e Erro de Generalização


Para discutir este fenômeno de forma mais formal,
precisamos diferenciar entre erro de treinamento e erro de generalização.
O *erro de treinamento* é o erro do nosso modelo
conforme calculado no conjunto de dados de treinamento,
enquanto *erro de generalização* é a expectativa do erro do nosso modelo
deveríamos aplicá-lo a um fluxo infinito de exemplos de dados adicionais
extraído da mesma distribuição de dados subjacente que nossa amostra original.

De forma problemática, nunca podemos calcular o erro de generalização com exatidão.
Isso ocorre porque o fluxo de dados infinitos é um objeto imaginário.
Na prática, devemos *estimar* o erro de generalização
aplicando nosso modelo a um conjunto de teste independente
constituído de uma seleção aleatória de exemplos de dados
que foram retirados de nosso conjunto de treinamento.


Os três experimentos mentais a seguir
ajudarão a ilustrar melhor esta situação.
Considere um estudante universitário tentando se preparar para o exame final.
Um aluno diligente se esforçará para praticar bem
e testar suas habilidades usando exames de anos anteriores.
No entanto, um bom desempenho em exames anteriores não é garantia
que ele se sobressairá quando for importante.
Por exemplo, o aluno pode tentar se preparar
aprendendo de cor as respostas às questões do exame.
Isso requer que o aluno memorize muitas coisas.
Ela pode até se lembrar das respostas de exames anteriores perfeitamente.
Outro aluno pode se preparar tentando entender
as razões para dar certas respostas.
Na maioria dos casos, o último aluno se sairá muito melhor.

Da mesma forma, considere um modelo que simplesmente usa uma tabela de pesquisa para responder às perguntas. Se o conjunto de entradas permitidas for discreto e razoavelmente pequeno, talvez depois de ver *muitos* exemplos de treinamento, essa abordagem teria um bom desempenho. Ainda assim, esse modelo não tem capacidade de fazer melhor do que adivinhação aleatória quando confrontado com exemplos que nunca viu antes.
Na realidade, os espaços de entrada são muito grandes para memorizar as respostas correspondentes a cada entrada concebível. Por exemplo, considere as imagens $28\times28$ em preto e branco. Se cada pixel pode ter um entre $256$ valores de tons de cinza, então há $256^{784}$ imagens possíveis. Isso significa que há muito mais imagens em miniatura em escala de cinza de baixa resolução do que átomos no universo. Mesmo se pudéssemos encontrar esses dados, nunca poderíamos nos dar ao luxo de armazenar a tabela de pesquisa.

Por último, considere o problema de tentar classificar os resultados dos lançamentos de moeda (classe 0: cara, classe 1: coroa)
com base em alguns recursos contextuais que podem estar disponíveis.
Suponha que a moeda seja justa.
Não importa o algoritmo que criamos,
o erro de generalização sempre será $\frac{1}{2}$.
No entanto, para a maioria dos algoritmos,
devemos esperar que nosso erro de treinamento seja consideravelmente menor,
dependendo da sorte do sorteio,
mesmo se não tivéssemos nenhuma *feature*!
Considere o conjunto de dados {0, 1, 1, 1, 0, 1}.
Nosso algoritmo sem recursos teria que recorrer sempre à previsão
da *classe majoritária*, que parece ser *1* em nossa amostra limitada.
Neste caso, o modelo que sempre prevê a classe 1
incorrerá em um erro de $\frac{1}{3}$,
consideravelmente melhor do que nosso erro de generalização.
Conforme aumentamos a quantidade de dados,
a probabilidade de que a fração de caras
irá se desviar significativamente de $\frac{1}{2}$ diminui,
e nosso erro de treinamento viria a corresponder ao erro de generalização.

### Teoria de Aprendizagem Estatística


Como a generalização é o problema fundamental no *machine learning*,
você pode não se surpreender ao aprender
que muitos matemáticos e teóricos dedicaram suas vidas
para desenvolver teorias formais para descrever este fenômeno.
Em seu [teorema de mesmo nome](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem), Glivenko e Cantelli
derivaram a taxa na qual o erro de treinamento
converge para o erro de generalização.
Em uma série de artigos seminais, [Vapnik e Chervonenkis](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory)
estenderam esta teoria a classes de funções mais gerais.
Este trabalho lançou as bases da teoria da aprendizagem estatística.


No ambiente de aprendizagem supervisionada padrão, que abordamos até agora e manteremos ao longo da maior parte deste livro,
presumimos que tanto os dados de treinamento quanto os dados de teste
são desenhados *independentemente* de distribuições *idênticas*.
Isso é comumente chamado de *suposição i.i.d.*,
o que significa que o processo que faz a amostragem de nossos dados não tem memória.
Em outras palavras,
o segundo exemplo desenhado e o terceiro desenhado
não são mais correlacionados do que a segunda e a segunda milionésima amostra extraída.

Ser um bom cientista de *machine learning* exige pensar criticamente,
e você já deve estar cutucando buracos nessa suposição,
surgindo com casos comuns em que a suposição falha.
E se treinarmos um preditor de risco de mortalidade
em dados coletados de pacientes no UCSF Medical Center,
e aplicá-lo em pacientes no *Massachusetts General Hospital*?
Essas distribuições simplesmente não são idênticas.
Além disso, os empates podem ser correlacionados no tempo.
E se estivermos classificando os tópicos dos Tweets?
O ciclo de notícias criaria dependências temporais
nos tópicos em discussão, violando quaisquer pressupostos de independência.


Às vezes, podemos escapar impunes de violações menores da suposição i.i.d. 
e nossos modelos continuarão a funcionar muito bem.
Afinal, quase todos os aplicativos do mundo real
envolvem pelo menos alguma violação menor da suposição i.i.d.,
e ainda temos muitas ferramentas úteis para
várias aplicações, como
reconhecimento de rosto,
reconhecimento de voz e tradução de idiomas.

Outras violações certamente causarão problemas.
Imagine, por exemplo, se tentarmos treinar
um sistema de reconhecimento de rosto treinando-o
exclusivamente em estudantes universitários
e então tentar implantá-lo como uma ferramenta
para monitorar a geriatria em uma população de lares de idosos.
É improvável que funcione bem, uma vez que estudantes universitários
tendem a parecer consideravelmente diferentes dos idosos.

Nos capítulos subsequentes, discutiremos problemas
decorrentes de violações da suposição i.i.d..
Por enquanto, mesmo tomando a suposição i.i.d. como certa,
compreender a generalização é um problema formidável.
Além disso, elucidando os fundamentos teóricos precisos
isso que podem explicar por que redes neurais profundas generalizam tão bem como o fazem,
continua a irritar as maiores mentes da teoria do aprendizado.

Quando treinamos nossos modelos, tentamos pesquisar uma função
que se ajusta aos dados de treinamento da melhor maneira possível.
Se a função é tão flexível que pode pegar padrões falsos
tão facilmente quanto às associações verdadeiras,
então ele pode funcionar *muito bem* sem produzir um modelo
que generaliza bem para dados invisíveis.
Isso é exatamente o que queremos evitar ou pelo menos controlar.
Muitas das técnicas de aprendizado profundo são heurísticas e truques
visando a proteção contra *overfitting*.

### Complexidade do Modelo

When we have simple models and abundant data,
we expect the generalization error to resemble the training error.
When we work with more complex models and fewer examples,
we expect the training error to go down but the generalization gap to grow.
What precisely constitutes model complexity is a complex matter.
Many factors govern whether a model will generalize well.
For example a model with more parameters might be considered more complex.
A model whose parameters can take a wider range of values
might be more complex.
Often with neural networks, we think of a model
that takes more training iterations as more complex,
and one subject to *early stopping* (fewer training iterations) as less complex.

It can be difficult to compare the complexity among members
of substantially different model classes
(say, decision trees vs. neural networks).
For now, a simple rule of thumb is quite useful:
a model that can readily explain arbitrary facts
is what statisticians view as complex,
whereas one that has only a limited expressive power
but still manages to explain the data well
is probably closer to the truth.
In philosophy, this is closely related to Popper's
criterion of falsifiability
of a scientific theory: a theory is good if it fits data
and if there are specific tests that can be used to disprove it.
This is important since all statistical estimation is
*post hoc*,
i.e., we estimate after we observe the facts,
hence vulnerable to the associated fallacy.
For now, we will put the philosophy aside and stick to more tangible issues.

In this section, to give you some intuition,
we will focus on a few factors that tend
to influence the generalizability of a model class:

1. The number of tunable parameters. When the number of tunable parameters, sometimes called the *degrees of freedom*, is large, models tend to be more susceptible to overfitting.
1. The values taken by the parameters. When weights can take a wider range of values, models can be more susceptible to overfitting.
1. The number of training examples. It is trivially easy to overfit a dataset containing only one or two examples even if your model is simple. But overfitting a dataset with millions of examples requires an extremely flexible model.

## Model Selection

In machine learning, we usually select our final model
after evaluating several candidate models.
This process is called *model selection*.
Sometimes the models subject to comparison
are fundamentally different in nature
(say, decision trees vs. linear models).
At other times, we are comparing
members of the same class of models
that have been trained with different hyperparameter settings.

With MLPs, for example,
we may wish to compare models with
different numbers of hidden layers,
different numbers of hidden units,
and various choices of the activation functions
applied to each hidden layer.
In order to determine the best among our candidate models,
we will typically employ a validation dataset.


### Validation Dataset

In principle we should not touch our test set
until after we have chosen all our hyperparameters.
Were we to use the test data in the model selection process,
there is a risk that we might overfit the test data.
Then we would be in serious trouble.
If we overfit our training data,
there is always the evaluation on test data to keep us honest.
But if we overfit the test data, how would we ever know?


Thus, we should never rely on the test data for model selection.
And yet we cannot rely solely on the training data
for model selection either because
we cannot estimate the generalization error
on the very data that we use to train the model.


In practical applications, the picture gets muddier.
While ideally we would only touch the test data once,
to assess the very best model or to compare
a small number of models to each other,
real-world test data is seldom discarded after just one use.
We can seldom afford a new test set for each round of experiments.

The common practice to address this problem
is to split our data three ways,
incorporating a *validation dataset* (or *validation set*)
in addition to the training and test datasets.
The result is a murky practice where the boundaries
between validation and test data are worryingly ambiguous.
Unless explicitly stated otherwise, in the experiments in this book
we are really working with what should rightly be called
training data and validation data, with no true test sets.
Therefore, the accuracy reported in each experiment of the book is really the validation accuracy and not a true test set accuracy.

### $K$-Fold Cross-Validation

When training data is scarce,
we might not even be able to afford to hold out
enough data to constitute a proper validation set.
One popular solution to this problem is to employ
$K$*-fold cross-validation*.
Here, the original training data is split into $K$ non-overlapping subsets.
Then model training and validation are executed $K$ times,
each time training on $K-1$ subsets and validating
on a different subset (the one not used for training in that round).
Finally, the training and validation errors are estimated
by averaging over the results from the $K$ experiments.

## Underfitting or Overfitting?

When we compare the training and validation errors,
we want to be mindful of two common situations.
First, we want to watch out for cases
when our training error and validation error are both substantial
but there is a little gap between them.
If the model is unable to reduce the training error,
that could mean that our model is too simple
(i.e., insufficiently expressive)
to capture the pattern that we are trying to model.
Moreover, since the *generalization gap*
between our training and validation errors is small,
we have reason to believe that we could get away with a more complex model.
This phenomenon is known as *underfitting*.

On the other hand, as we discussed above,
we want to watch out for the cases
when our training error is significantly lower
than our validation error, indicating severe *overfitting*.
Note that overfitting is not always a bad thing.
With deep learning especially, it is well known
that the best predictive models often perform
far better on training data than on holdout data.
Ultimately, we usually care more about the validation error
than about the gap between the training and validation errors.

Whether we overfit or underfit can depend
both on the complexity of our model
and the size of the available training datasets,
two topics that we discuss below.

### Model Complexity

To illustrate some classical intuition
about overfitting and model complexity,
we give an example using polynomials.
Given training data consisting of a single feature $x$
and a corresponding real-valued label $y$,
we try to find the polynomial of degree $d$

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

to estimate the labels $y$.
This is just a linear regression problem
where our features are given by the powers of $x$,
the model's weights are given by $w_i$,
and the bias is given by $w_0$ since $x^0 = 1$ for all $x$.
Since this is just a linear regression problem,
we can use the squared error as our loss function.


A higher-order polynomial function is more complex
than a lower-order polynomial function,
since the higher-order polynomial has more parameters
and the model function's selection range is wider.
Fixing the training dataset,
higher-order polynomial functions should always
achieve lower (at worst, equal) training error
relative to lower degree polynomials.
In fact, whenever the data examples each have a distinct value of $x$,
a polynomial function with degree equal to the number of data examples
can fit the training set perfectly.
We visualize the relationship between polynomial degree
and underfitting vs. overfitting in :numref:`fig_capacity_vs_error`.

![Influence of model complexity on underfitting and overfitting](../img/capacity-vs-error.svg)
:label:`fig_capacity_vs_error`

### Dataset Size

The other big consideration to bear in mind is the dataset size.
Fixing our model, the fewer samples we have in the training dataset,
the more likely (and more severely) we are to encounter overfitting.
As we increase the amount of training data,
the generalization error typically decreases.
Moreover, in general, more data never hurt.
For a fixed task and data distribution,
there is typically a relationship between model complexity and dataset size.
Given more data, we might profitably attempt to fit a more complex model.
Absent sufficient data, simpler models may be more difficult to beat.
For many tasks, deep learning only outperforms linear models
when many thousands of training examples are available.
In part, the current success of deep learning
owes to the current abundance of massive datasets
due to Internet companies, cheap storage, connected devices,
and the broad digitization of the economy.

## Polynomial Regression

We can now (**explore these concepts interactively
by fitting polynomials to data.**)

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import math
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np
import math
```

### Generating the Dataset

First we need data. Given $x$, we will [**use the following cubic polynomial to generate the labels**] on training and test data:

(**$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2).$$**)

The noise term $\epsilon$ obeys a normal distribution
with a mean of 0 and a standard deviation of 0.1.
For optimization, we typically want to avoid
very large values of gradients or losses.
This is why the *features*
are rescaled from $x^i$ to $\frac{x^i}{i!}$.
It allows us to avoid very large values for large exponents $i$.
We will synthesize 100 samples each for the training set and test set.

```{.python .input}
#@tab all
max_degree = 20  # Maximum degree of the polynomial
n_train, n_test = 100, 100  # Training and test dataset sizes
true_w = np.zeros(max_degree)  # Allocate lots of empty space
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!
# Shape of `labels`: (`n_train` + `n_test`,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
```

Again, monomials stored in `poly_features`
are rescaled by the gamma function,
where $\Gamma(n)=(n-1)!$.
[**Take a look at the first 2 samples**] from the generated dataset.
The value 1 is technically a feature,
namely the constant feature corresponding to the bias.

```{.python .input}
#@tab pytorch, tensorflow
# Convert from NumPy ndarrays to tensors
true_w, features, poly_features, labels = [d2l.tensor(x, dtype=
    d2l.float32) for x in [true_w, features, poly_features, labels]]
```

```{.python .input}
#@tab all
features[:2], poly_features[:2, :], labels[:2]
```

### Training and Testing the Model

Let us first [**implement a function to evaluate the loss on a given dataset**].

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_loss(net, data_iter, loss):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_loss(net, data_iter, loss):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

Now [**define the training function**].

```{.python .input}
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = gluon.loss.L2Loss()
    net = nn.Sequential()
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net.add(nn.Dense(1, use_bias=False))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data().asnumpy())
```

```{.python .input}
#@tab pytorch
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
```

```{.python .input}
#@tab tensorflow
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = tf.losses.MeanSquaredError()
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, use_bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = tf.keras.optimizers.SGD(learning_rate=.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net.get_weights()[0].T)
```

### [**Third-Order Polynomial Function Fitting (Normal)**]

We will begin by first using a third-order polynomial function, which is the same order as that of the data generation function.
The results show that this model's training and test losses can be both effectively reduced.
The learned model parameters are also close
to the true values $w = [5, 1.2, -3.4, 5.6]$.

```{.python .input}
#@tab all
# Pick the first four dimensions, i.e., 1, x, x^2/2!, x^3/3! from the
# polynomial features
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
```

### [**Linear Function Fitting (Underfitting)**]

Let us take another look at linear function fitting.
After the decline in early epochs,
it becomes difficult to further decrease
this model's training loss.
After the last epoch iteration has been completed,
the training loss is still high.
When used to fit nonlinear patterns
(like the third-order polynomial function here)
linear models are liable to underfit.

```{.python .input}
#@tab all
# Pick the first two dimensions, i.e., 1, x, from the polynomial features
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
```

### [**Higher-Order Polynomial Function Fitting  (Overfitting)**]

Now let us try to train the model
using a polynomial of too high degree.
Here, there are insufficient data to learn that
the higher-degree coefficients should have values close to zero.
As a result, our overly-complex model
is so susceptible that it is being influenced
by noise in the training data.
Though the training loss can be effectively reduced,
the test loss is still much higher.
It shows that
the complex model overfits the data.

```{.python .input}
#@tab all
# Pick all the dimensions from the polynomial features
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
```

In the subsequent sections, we will continue
to discuss overfitting problems
and methods for dealing with them,
such as weight decay and dropout.


## Summary

* Since the generalization error cannot be estimated based on the training error, simply minimizing the training error will not necessarily mean a reduction in the generalization error. Machine learning models need to be careful to safeguard against overfitting so as to minimize the generalization error.
* A validation set can be used for model selection, provided that it is not used too liberally.
* Underfitting means that a model is not able to reduce the training error. When training error is much lower than validation error, there is overfitting.
* We should choose an appropriately complex model and avoid using insufficient training samples.


## Exercises

1. Can you solve the polynomial regression problem exactly? Hint: use linear algebra.
1. Consider model selection for polynomials:
    1. Plot the training loss vs. model complexity (degree of the polynomial). What do you observe? What degree of polynomial do you need to reduce the training loss to 0?
    1. Plot the test loss in this case.
    1. Generate the same plot as a function of the amount of data.
1. What happens if you drop the normalization ($1/i!$) of the polynomial features $x^i$? Can you fix this in some other way?
1. Can you ever expect to see zero generalization error?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbODU0ODY2MzcyLDE0NDM3NTk5Myw0OTM5OT
Y3NzcsLTEyNTI5MTY5MDgsLTEwNDA5Njk5MjVdfQ==
-->
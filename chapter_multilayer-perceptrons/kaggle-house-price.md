# Previsão de Preços de Imóveis no Kaggle
:label:`sec_kaggle_house`


Agora que apresentamos algumas ferramentas básicas
para construir e treinar redes profundas
e regularizá-las com técnicas, incluindo
queda de peso e abandono,
estamos prontos para colocar todo esse conhecimento em prática
participando de uma competição *Kaggle*.
A competição de previsão de preços de casas
é um ótimo lugar para começar.
Os dados são bastante genéricos e não exibem uma estrutura exótica
que podem exigir modelos especializados (como áudio ou vídeo).
Este conjunto de dados, coletado por Bart de Cock em 2011 :cite:`De-Cock.2011`,
cobre os preços da habitação em Ames, IA do período de 2006-2010.
É consideravelmente maior do que o famoso [conjunto de dados de habitação de Boston](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) de Harrison e Rubinfeld (1978),
ostentando mais exemplos e mais recursos.


Nesta seção, iremos orientá-lo nos detalhes de
pré-processamento de dados, design de modelo e seleção de hiperparâmetros.
Esperamos que, por meio de uma abordagem prática,
você ganhe algumas intuições que irão guiá-lo
em sua carreira como cientista de dados.


## *Download* e *Cache* de *datasets*
Ao longo do livro, treinaremos e testaremos modelos
em vários conjuntos de dados baixados.
Aqui, implementamos várias funções utilitárias
para facilitar o *download* de dados.
Primeiro, mantemos um dicionário `DATA_HUB`
que mapeia uma string (o *nome* do conjunto de dados)
a uma tupla contendo o URL para localizar o conjunto de dados
e a chave SHA-1 que verifica a integridade do arquivo.
Todos esses conjuntos de dados são hospedados no site
cujo endereço é `DATA_URL`.

```{.python .input}
#@tab all
import os
import requests
import zipfile
import tarfile
import hashlib

#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```

A seguinte função `download` baixa um conjunto de dados,
armazenando em cache em um diretório local (`../data` por padrão)
e retorna o nome do arquivo baixado.
Se um arquivo correspondente a este conjunto de dados
já existe no diretório de cache
e seu SHA-1 corresponde ao armazenado em `DATA_HUB`,
nosso código usará o arquivo em cache para evitar
obstruir sua internet com *downloads* redundantes.

```{.python .input}
#@tab all
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # Hit cache
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
```

Também implementamos duas funções utilitárias adicionais:
uma é baixar e extrair um arquivo zip ou tar
e o outro para baixar todos os conjuntos de dados usados neste livro de `DATA_HUB` para o diretório de cache.

```{.python .input}
#@tab all
def download_extract(name, folder=None):  #@save
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """Download all files in the DATA_HUB."""
    for name in DATA_HUB:
        download(name)
```

## *Kaggle*

[Kaggle](https://www.kaggle.com) é uma plataforma popular
que hospeda competições de *machine learning*.
Cada competição se concentra em um conjunto de dados e muitos
são patrocinados por interessados que oferecem prêmios
para as soluções vencedoras.
A plataforma ajuda os usuários a interagir
por meio de fóruns e código compartilhado,
fomentando a colaboração e a competição.
Embora a perseguição ao placar muitas vezes saia do controle,
com pesquisadores focando miopicamente nas etapas de pré-processamento
em vez de fazer perguntas fundamentais,
também há um enorme valor na objetividade de uma plataforma
que facilita comparações quantitativas diretas
entre abordagens concorrentes, bem como compartilhamento de código
para que todos possam aprender o que funcionou e o que não funcionou.
Se você quiser participar de uma competição *Kaggle*,
primeiro você precisa se registrar para uma conta
(veja :numref:`fig_kaggle`).

![Site do Kaggle.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

Na página de competição de previsão de preços de casas, conforme ilustrado
em :numref:`fig_house_pricing`,
você pode encontrar o conjunto de dados (na guia "Dados"),
enviar previsões e ver sua classificação,
A URL está bem aqui:

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![A página da competição de previsão de preços de casas.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## Acessando e Lendo o Conjunto de Dados


Observe que os dados da competição são separados
em conjuntos de treinamento e teste.
Cada registro inclui o valor da propriedade da casa
e atributos como tipo de rua, ano de construção,
tipo de telhado, condição do porão, etc.
Os recursos consistem em vários tipos de dados.
Por exemplo, o ano de construção
é representado por um número inteiro,
o tipo de telhado por atribuições categóricas discretas,
e outros recursos por números de ponto flutuante.
E é aqui que a realidade complica as coisas:
para alguns exemplos, alguns dados estão ausentes
com o valor ausente marcado simplesmente como "na".
O preço de cada casa está incluído
para o conjunto de treinamento apenas
(afinal, é uma competição).
Queremos particionar o conjunto de treinamento
para criar um conjunto de validação,
mas só podemos avaliar nossos modelos no conjunto de teste oficial
depois de enviar previsões para Kaggle.
A guia "Dados" na guia da competição
em :numref:`fig_house_pricing`
tem links para baixar os dados.


Para começar, vamos ler e processar os dados
usando `pandas`, que introduzimos em :numref:`sec_pandas`.
Então, você vai querer ter certeza de que instalou o `pandas`
antes de prosseguir.
Felizmente, se você estiver no Jupyter,
podemos instalar pandas sem nem mesmo sair do notebook.

```{.python .input}
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
#@tab pytorch
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
import numpy as np
```

```{.python .input}
#@tab tensorflow
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
import numpy as np
```

Por conveniência, podemos baixar e armazenar em cache
o conjunto de dados de habitação *Kaggle*
usando o *script* que definimos acima.

```{.python .input}
#@tab all
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

Usamos `pandas` para carregar os dois arquivos csv contendo dados de treinamento e teste, respectivamente.

```{.python .input}
#@tab all
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

O conjunto de dados de treinamento inclui 1460 exemplos,
80 características e 1 rótulo, enquanto os dados de teste
contém 1459 exemplos e 80 características.

```{.python .input}
#@tab all
print(train_data.shape)
print(test_data.shape)
```

Vamos dar uma olhada nos primeiros quatro e nos dois últimos recursos
bem como o rótulo (SalePrice) dos primeiros quatro exemplos.

```{.python .input}
#@tab all
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

Podemos ver que em cada exemplo, a primeira característica é o ID.
Isso ajuda o modelo a identificar cada exemplo de treinamento.
Embora seja conveniente, ele não carrega
qualquer informação para fins de previsão.
Portanto, nós o removemos do conjunto de dados
antes de alimentar os dados no modelo.

```{.python .input}
#@tab all
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## Pré-processamento de Dados

Conforme declarado acima, temos uma grande variedade de tipos de dados.
Precisaremos pré-processar os dados antes de começarmos a modelagem.
Vamos começar com as *features* numéricas.
Primeiro, aplicamos uma heurística,
substituindo todos os valores ausentes
pela média da feature correspondente.
Então, para colocar todos os recursos em uma escala comum,
nós *padronizamos* os dados
redimensionando recursos para média zero e variancia unitária:

$$x \leftarrow \frac{x - \mu}{\sigma},$$

onde $\mu$ e $\sigma$ denotam média e desvio padrão, respectivamente.
Para verificar se isso realmente transforma
nossa *feature* (variável) de modo que tenha média zero e variância unitária,
observe que $E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$
e que $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$.
Intuitivamente, padronizamos os dados
por duas razões.
Primeiro, se mostra conveniente para otimização.
Em segundo lugar, porque não sabemos *a priori*
quais *features* serão relevantes,
não queremos penalizar coeficientes
atribuídos a uma *feature* mais do que a qualquer outra.

```{.python .input}
#@tab all
# If test data were inaccessible, mean and standard deviation could be 
# calculated from training data
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing
# values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

Next we deal with discrete values.
This includes features such as "MSZoning".
We replace them by a one-hot encoding
in the same way that we previously transformed
multiclass labels into vectors (see :numref:`subsec_classification-problem`).
For instance, "MSZoning" assumes the values "RL" and "RM".
Dropping the "MSZoning" feature,
two new indicator features
"MSZoning_RL" and "MSZoning_RM" are created with values being either 0 or 1.
According to one-hot encoding,
if the original value of "MSZoning" is "RL",
then "MSZoning_RL" is 1 and "MSZoning_RM" is 0.
The `pandas` package does this automatically for us.

```{.python .input}
#@tab all
# `Dummy_na=True` considers "na" (missing value) as a valid feature value, and
# creates an indicator feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

You can see that this conversion increases
the number of features from 79 to 331.
Finally, via the `values` attribute,
we can extract the NumPy format from the `pandas` format
and convert it into the tensor
representation for training.

```{.python .input}
#@tab all
n_train = train_data.shape[0]
train_features = d2l.tensor(all_features[:n_train].values, dtype=d2l.float32)
test_features = d2l.tensor(all_features[n_train:].values, dtype=d2l.float32)
train_labels = d2l.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=d2l.float32)
```

## Training

To get started we train a linear model with squared loss.
Not surprisingly, our linear model will not lead
to a competition-winning submission
but it provides a sanity check to see whether
there is meaningful information in the data.
If we cannot do better than random guessing here,
then there might be a good chance
that we have a data processing bug.
And if things work, the linear model will serve as a baseline
giving us some intuition about how close the simple model
gets to the best reported models, giving us a sense
of how much gain we should expect from fancier models.

```{.python .input}
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()

def get_net():
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(
        1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    return net
```

With house prices, as with stock prices,
we care about relative quantities
more than absolute quantities.
Thus we tend to care more about
the relative error $\frac{y - \hat{y}}{y}$
than about the absolute error $y - \hat{y}$.
For instance, if our prediction is off by USD 100,000
when estimating the price of a house in Rural Ohio,
where the value of a typical house is 125,000 USD,
then we are probably doing a horrible job.
On the other hand, if we err by this amount
in Los Altos Hills, California,
this might represent a stunningly accurate prediction
(there, the median house price exceeds 4 million USD).

One way to address this problem is to
measure the discrepancy in the logarithm of the price estimates.
In fact, this is also the official error measure
used by the competition to evaluate the quality of submissions.
After all, a small value $\delta$ for $|\log y - \log \hat{y}| \leq \delta$
translates into $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.
This leads to the following root-mean-squared-error between the logarithm of the predicted price and the logarithm of the label price:

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input}
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

```{.python .input}
#@tab pytorch
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()
```

```{.python .input}
#@tab tensorflow
def log_rmse(y_true, y_pred):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = tf.clip_by_value(y_pred, 1, float('inf'))
    return tf.sqrt(tf.reduce_mean(loss(
        tf.math.log(y_true), tf.math.log(clipped_preds))))
```

Unlike in previous sections, our training functions
will rely on the Adam optimizer
(we will describe it in greater detail later).
The main appeal of this optimizer is that,
despite doing no better (and sometimes worse)
given unlimited resources for hyperparameter optimization,
people tend to find that it is significantly less sensitive
to the initial learning rate.

```{.python .input}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab pytorch
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

```{.python .input}
#@tab tensorflow
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    net.compile(loss=loss, optimizer=optimizer)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = loss(y, y_hat)
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
        train_ls.append(log_rmse(train_labels, net(train_features)))
        if test_labels is not None:
            test_ls.append(log_rmse(test_labels, net(test_features)))
    return train_ls, test_ls
```

## $K$-Fold Cross-Validation

You might recall that we introduced $K$-fold cross-validation
in the section where we discussed how to deal
with model selection (:numref:`sec_model_selection`).
We will put this to good use to select the model design
and to adjust the hyperparameters.
We first need a function that returns
the $i^\mathrm{th}$ fold of the data
in a $K$-fold cross-validation procedure.
It proceeds by slicing out the $i^\mathrm{th}$ segment
as validation data and returning the rest as training data.
Note that this is not the most efficient way of handling data
and we would definitely do something much smarter
if our dataset was considerably larger.
But this added complexity might obfuscate our code unnecessarily
so we can safely omit it here owing to the simplicity of our problem.

```{.python .input}
#@tab all
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = d2l.concat([X_train, X_part], 0)
            y_train = d2l.concat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

The training and verification error averages are returned
when we train $K$ times in the $K$-fold cross-validation.

```{.python .input}
#@tab all
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'fold {i + 1}, train log rmse {float(train_ls[-1]):f}, '
              f'valid log rmse {float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

## Model Selection

In this example, we pick an untuned set of hyperparameters
and leave it up to the reader to improve the model.
Finding a good choice can take time,
depending on how many variables one optimizes over.
With a large enough dataset,
and the normal sorts of hyperparameters,
$K$-fold cross-validation tends to be
reasonably resilient against multiple testing.
However, if we try an unreasonably large number of options
we might just get lucky and find that our validation
performance is no longer representative of the true error.

```{.python .input}
#@tab all
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-fold validation: avg train log rmse: {float(train_l):f}, '
      f'avg valid log rmse: {float(valid_l):f}')
```

Notice that sometimes the number of training errors
for a set of hyperparameters can be very low,
even as the number of errors on $K$-fold cross-validation
is considerably higher.
This indicates that we are overfitting.
Throughout training you will want to monitor both numbers.
Less overfitting might indicate that our data can support a more powerful model.
Massive overfitting might suggest that we can gain
by incorporating regularization techniques.

##  Submitting Predictions on Kaggle

Now that we know what a good choice of hyperparameters should be,
we might as well use all the data to train on it
(rather than just $1-1/K$ of the data
that are used in the cross-validation slices).
The model that we obtain in this way
can then be applied to the test set.
Saving the predictions in a csv file
will simplify uploading the results to Kaggle.

```{.python .input}
#@tab all
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'train log rmse {float(train_ls[-1]):f}')
    # Apply the network to the test set
    preds = d2l.numpy(net(test_features))
    # Reformat it to export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

One nice sanity check is to see
whether the predictions on the test set
resemble those of the $K$-fold cross-validation process.
If they do, it is time to upload them to Kaggle.
The following code will generate a file called `submission.csv`.

```{.python .input}
#@tab all
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

Next, as demonstrated in :numref:`fig_kaggle_submit2`,
we can submit our predictions on Kaggle
and see how they compare with the actual house prices (labels)
on the test set.
The steps are quite simple:

* Log in to the Kaggle website and visit the house price prediction competition page.
* Click the “Submit Predictions” or “Late Submission” button (as of this writing, the button is located on the right).
* Click the “Upload Submission File” button in the dashed box at the bottom of the page and select the prediction file you wish to upload.
* Click the “Make Submission” button at the bottom of the page to view your results.

![Submitting data to Kaggle](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## Summary

* Real data often contain a mix of different data types and need to be preprocessed.
* Rescaling real-valued data to zero mean and unit variance is a good default. So is replacing missing values with their mean.
* Transforming categorical features into indicator features allows us to treat them like one-hot vectors.
* We can use $K$-fold cross-validation to select the model and adjust the hyperparameters.
* Logarithms are useful for relative errors.


## Exercises

1. Submit your predictions for this section to Kaggle. How good are your predictions?
1. Can you improve your model by minimizing the logarithm of prices directly? What happens if you try to predict the logarithm of the price rather than the price?
1. Is it always a good idea to replace missing values by their mean? Hint: can you construct a situation where the values are not missing at random?
1. Improve the score on Kaggle by tuning the hyperparameters through $K$-fold cross-validation.
1. Improve the score by improving the model (e.g., layers, weight decay, and dropout).
1. What happens if we do not standardize the continuous numerical features like what we have done in this section?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTczMTgzMzA4LC01NDMxMDYxODcsMjEzOT
c5ODY1MCwyMjExNjM4ODVdfQ==
-->
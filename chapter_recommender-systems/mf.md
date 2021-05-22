# Fatoração de Matriz
 :cite:`Koren.Bell.Volinsky.2009` 
 
Fatoração de matrizes é um algoritmo bem estabelecido na literatura de sistemas de recomendação. A primeira versão do modelo de fatoração de matrizes é proposta por Simon Funk em um famoso [blog post](https://sifter.org/~simon/journal/20061211.html) em que descreveu a ideia de fatorar a matriz de interação. Em seguida, tornou-se amplamente conhecido devido ao concurso Netflix realizado em 2006. Naquela época, a Netflix, uma empresa de streaming de mídia e locação de vídeo, anunciou um concurso para melhorar o desempenho de seu sistema de recomendação. A melhor equipe que puder melhorar na linha de base do Netflix, ou seja, Cinematch), em 10 por cento ganharia um prêmio de um milhão de dólares. Como tal, este concurso atraiu
muita atenção ao campo de pesquisa de sistemas de recomendação. Posteriormente, o grande prêmio foi ganho pela equipe Pragmatic Chaos da BellKor, uma equipe combinada de BellKor, Pragmatic Theory e BigChaos (você não precisa se preocupar com esses algoritmos agora). Embora a pontuação final tenha sido o resultado de uma solução de conjunto (ou seja, uma combinação de muitos algoritmos), o algoritmo de fatoração de matriz desempenhou um papel crítico na mistura final. O relatório técnico da solução do Grande Prêmio Netflix :cite:`Toscher.Jahrer.Bell.2009` fornece uma introdução detalhada ao modelo adotado. Nesta seção, vamos mergulhar nos detalhes do modelo de fatoração de matriz e sua implementação.


## O modelo de fatoração de matriz

A fatoração de matriz é uma classe de modelos de filtragem colaborativa. Especificamente, o modelo fatoriza a matriz de interação usuário-item (por exemplo, matriz de classificação) no produto de duas matrizes de classificação inferior, capturando a estrutura de classificação inferior das interações usuário-item.

Let $\mathbf{R} \in \mathbb{R}^{m \times n}$ denote the interaction matrix with $m$ users and $n$ items, and the values of $\mathbf{R}$ represent explicit ratings. The user-item interaction will be factorized into a user latent matrix $\mathbf{P} \in \mathbb{R}^{m \times k}$ and an item latent matrix $\mathbf{Q} \in \mathbb{R}^{n \times k}$, where $k \ll m, n$, is the latent factor size. Let $\mathbf{p}_u$ denote the $u^\mathrm{th}$ row of $\mathbf{P}$ and $\mathbf{q}_i$ denote the $i^\mathrm{th}$ row of $\mathbf{Q}$.  For a given item $i$, the elements of $\mathbf{q}_i$ measure the extent to which the item possesses those characteristics such as the genres and languages of a movie. For a given user $u$, the elements of $\mathbf{p}_u$ measure the extent of interest the user has in items' corresponding characteristics. These latent factors might measure obvious dimensions as mentioned in those examples or are completely uninterpretable. The predicted ratings can be estimated by

Seja $\mathbf{R} \in \mathbb{R}^{m \times n}$ a matriz de interação com $m$ usuários e $n$ itens, e os valores de $\mathbf{R}$ representam explícito avaliações. A interação usuário-item será fatorada em uma matriz latente de usuário $\mathbf{P} \in \mathbb{R}^{m \times k}$ e uma matriz latente de item $\mathbf{Q} \in \mathbb{R}^{n \times k}$, onde $k \ll m, n$, é o tamanho do fator latente. Deixe $\mathbf{p}_u$ denotar a linha $u^\mathrm{th}$ linha de $\mathbf{P}$ denotar a linha $ i ^ \ mathrm {th} $ de $ \ mathbf {Q} $. Para um determinado item $i$, os elementos de $ \ mathbf {q} _i $ medem até que ponto o item possui essas características, como gêneros e idiomas de um filme. Para um dado usuário $ u $, os elementos de $ \ mathbf {p} _u $ medem a extensão do interesse que o usuário tem nas características correspondentes dos itens. Esses fatores latentes podem medir dimensões óbvias conforme mencionado nesses exemplos ou são completamente ininterpretáveis. As classificações previstas podem ser estimadas por

$$\hat{\mathbf{R}} = \mathbf{PQ}^\top$$

where $\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$ is the predicted rating matrix which has the same shape as $\mathbf{R}$. One major problem of this prediction rule is that users/items biases can not be modeled. For example, some users tend to give higher ratings or some items always get lower ratings due to poorer quality. These biases are commonplace in real-world applications. To capture these biases, user specific and item specific bias terms are introduced. Specifically, the predicted rating user $u$ gives to item $i$ is calculated by

$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
$$

Then, we train the matrix factorization model by minimizing the mean squared error between predicted rating scores and real rating scores.  The objective function is defined as follows:

$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}} \sum_{(u, i) \in \mathcal{K}} \| \mathbf{R}_{ui} -
\hat{\mathbf{R}}_{ui} \|^2 + \lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )
$$

where $\lambda$ denotes the regularization rate. The regularizing term $\lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )$ is used to avoid over-fitting by penalizing the magnitude of the parameters. The $(u, i)$ pairs for which $\mathbf{R}_{ui}$ is known are stored in the set
$\mathcal{K}=\{(u, i) \mid \mathbf{R}_{ui} \text{ is known}\}$. The model parameters can be learned with an optimization algorithm, such as Stochastic Gradient Descent and Adam.

An intuitive illustration of the matrix factorization model is shown below:

![Illustration of matrix factorization model](../img/rec-mf.svg)

In the rest of this section, we will explain the implementation of matrix factorization and train the model on the MovieLens dataset.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```

## Model Implementation

First, we implement the matrix factorization model described above. The user and item latent factors can be created with the `nn.Embedding`. The `input_dim` is the number of items/users and the (`output_dim`) is the dimension of the latent factors ($k$).  We can also use `nn.Embedding` to create the user/item biases by setting the `output_dim` to one. In the `forward` function, user and item ids are used to look up the embeddings.

```{.python .input  n=4}
class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_items, output_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
```

## Evaluation Measures

We then implement the RMSE (root-mean-square error) measure, which is commonly used to measure the differences between rating scores predicted by the model and the actually observed ratings (ground truth) :cite:`Gunawardana.Shani.2015`. RMSE is defined as:

$$
\mathrm{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u, i) \in \mathcal{T}}(\mathbf{R}_{ui} -\hat{\mathbf{R}}_{ui})^2}
$$

where $\mathcal{T}$ is the set consisting of pairs of users and items that you want to evaluate on. $|\mathcal{T}|$ is the size of this set. We can use the RMSE function provided by `mx.metric`.

```{.python .input  n=3}
def evaluator(net, test_iter, devices):
    rmse = mx.metric.RMSE()  # Get the RMSE
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, devices, even_split=False)
        i = gluon.utils.split_and_load(items, devices, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, devices, even_split=False)
        r_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    return float(np.mean(np.array(rmse_list)))
```

## Training and Evaluating the Model


In the training function, we adopt the $L_2$ loss with weight decay. The weight decay mechanism has the same effect as the $L_2$ regularization.

```{.python .input  n=4}
#@save
def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=d2l.try_all_gpus(), evaluator=None,
                        **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            train_feat = input_data[0:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'],
                                  devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

Finally, let us put all things together and train the model. Here, we set the latent factor dimension to 30.

```{.python .input  n=5}
devices = d2l.try_all_gpus()
num_users, num_items, train_iter, test_iter = d2l.split_and_load_ml100k(
    test_ratio=0.1, batch_size=512)
net = MF(30, num_users, num_items)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                    devices, evaluator)
```

Below, we use the trained model to predict the rating that a user (ID 20) might give to an item (ID 30).

```{.python .input  n=6}
scores = net(np.array([20], dtype='int', ctx=devices[0]),
             np.array([30], dtype='int', ctx=devices[0]))
scores
```

## Summary

* The matrix factorization model is widely used in recommender systems.  It can be used to predict ratings that a user might give to an item.
* We can implement and train matrix factorization for recommender systems.


## Exercises

* Vary the size of latent factors. How does the size of latent factors influence the model performance?
* Try different optimizers, learning rates, and weight decay rates.
* Check the predicted rating scores of other users for a specific movie.


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/400)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTc3MjE5MTA0NCwtMTg1OTYwMzAyM119
-->
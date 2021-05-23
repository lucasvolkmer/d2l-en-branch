# Classificação personalizada para sistemas de recomendação

Nas seções anteriores, apenas o feedback explícito foi considerado e os modelos foram treinados e testados nas classificações observadas. Existem dois pontos negativos de tais métodos: primeiro, a maior parte do feedback não é explícito, mas implícito em cenários do mundo real, e o feedback explícito pode ser mais caro de coletar. Em segundo lugar, pares de itens de usuário não observados que podem ser preditivos para os interesses dos usuários são totalmente ignorados, tornando esses métodos inadequados para os casos em que as classificações não estão faltando aleatoriamente, mas devido às preferências dos usuários. Os pares de itens de usuário não observados são uma mistura de feedback negativo real (os usuários não estão interessados nos itens) e valores ausentes (o usuário pode interagir com os itens no futuro). Simplesmente ignoramos os pares não observados na fatoração da matriz e no AutoRec. Claramente, esses modelos são incapazes de distinguir entre pares observados e não observados e geralmente não são adequados para tarefas de classificação personalizada.

Para esse fim, uma classe de modelos de recomendação com o objetivo de gerar listas de recomendações classificadas a partir de feedback implícito ganhou popularidade. Em geral, os modelos de classificação personalizados podem ser otimizados com abordagens pontuais, de pares ou de lista. As abordagens pontuais consideram uma única interação por vez e treinam um classificador ou regressor para prever preferências individuais. A fatoração de matriz e o AutoRec são otimizados com objetivos pontuais. As abordagens de pares consideram um par de itens para cada usuário e visam aproximar a ordenação ideal para esse par. Normalmente, as abordagens de pares são mais adequadas para a tarefa de classificação porque a previsão da ordem relativa é uma reminiscência da natureza da classificação. Abordagens listwise aproximam a ordem de toda a lista de itens, por exemplo, otimização direta das medidas de classificação como ganho cumulativo com desconto normalizado ([NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)). No entanto, as abordagens listwise são mais complexas e intensivas em computação do que as abordagens pontuais ou de pares. Nesta seção, apresentaremos dois objetivos / perdas de pares, perda de classificação personalizada Bayesiana e perda de dobradiça, e suas respectivas implementações.

## Perda de classificação personalizada bayesiana e sua implementação

Bayesian personalized ranking (BPR) :cite:`Rendle.Freudenthaler.Gantner.ea.2009` is a pairwise personalized ranking loss that is derived from the maximum posterior estimator. It has been widely used in many existing recommendation models. The training data of BPR consists of both positive and negative pairs (missing values). It assumes that the user prefers the positive item over all other non-observed items.

In formal, the training data is constructed by tuples in the form of $(u, i, j)$, which represents that the user $u$ prefers the item $i$ over the item $j$. The Bayesian formulation of BPR which aims to maximize the posterior probability is given below:

A classificação personalizada bayesiana (BPR): cite: `Rendle.Freudenthaler.Gantner.ea.2009` é uma perda de classificação personalizada aos pares que é derivada do estimador posterior máximo. Ele tem sido amplamente utilizado em muitos modelos de recomendação existentes. Os dados de treinamento do BPR consistem em pares positivos e negativos (valores ausentes). Ele assume que o usuário prefere o item positivo a todos os outros itens não observados.

Formalmente, os dados de treinamento são construídos por tuplas na forma de $ (u, i, j) $, que representa que o usuário $ u $ prefere o item $ i $ em vez do item $ j $. A formulação bayesiana do BPR que visa maximizar a probabilidade posterior é dada a seguir:

$$
p(\Theta \mid >_u )  \propto  p(>_u \mid \Theta) p(\Theta)
$$

Where $\Theta$ represents the parameters of an arbitrary recommendation model, $>_u$ represents the desired personalized total ranking of all items for user $u$. We can formulate the maximum posterior estimator to derive the generic optimization criterion for the personalized ranking task.

$$
\begin{aligned}
\text{BPR-OPT} : &= \ln p(\Theta \mid >_u) \\
         & \propto \ln p(>_u \mid \Theta) p(\Theta) \\
         &= \ln \prod_{(u, i, j \in D)} \sigma(\hat{y}_{ui} - \hat{y}_{uj}) p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \ln p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta \|\Theta \|^2
\end{aligned}
$$


where $D := \{(u, i, j) \mid i \in I^+_u \wedge j \in I \backslash I^+_u \}$ is the training set, with $I^+_u$ denoting the items the user $u$ liked, $I$ denoting all items, and $I \backslash I^+_u$ indicating all other items excluding items the user liked. $\hat{y}_{ui}$ and $\hat{y}_{uj}$ are the predicted scores of the user $u$ to item $i$ and $j$, respectively. The prior $p(\Theta)$ is a normal distribution with zero mean and variance-covariance matrix $\Sigma_\Theta$. Here, we let $\Sigma_\Theta = \lambda_\Theta I$.

![Illustration of Bayesian Personalized Ranking](../img/rec-ranking.svg)
We will implement the base class  `mxnet.gluon.loss.Loss` and override the `forward` method to construct the Bayesian personalized ranking loss. We begin by importing the Loss class and the np module.

```{.python .input  n=5}
from mxnet import gluon, np, npx
npx.set_np()
```

The implementation of BPR loss is as follows.

```{.python .input  n=2}
#@save
class BPRLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - np.sum(np.log(npx.sigmoid(distances)), 0, keepdims=True)
        return loss
```

## Hinge Loss and its Implementation

The Hinge loss for ranking has different form to the [hinge loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss) provided within the gluon library that is often used in classifiers such as SVMs.  The loss used for ranking in recommender systems has the following form.

$$
 \sum_{(u, i, j \in D)} \max( m - \hat{y}_{ui} + \hat{y}_{uj}, 0)
$$

where $m$ is the safety margin size. It aims to push negative items away from positive items. Similar to BPR, it aims to optimize for relevant distance between positive and negative samples instead of absolute outputs, making it well suited to recommender systems.

```{.python .input  n=3}
#@save
class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(- distances + margin, 0))
        return loss
```

These two losses are interchangeable for personalized ranking in recommendation.

## Summary

- There are three types of ranking losses available for the personalized ranking task in recommender systems, namely, pointwise, pairwise and listwise methods.
- The two pairwise loses, Bayesian personalized ranking loss and hinge loss, can be used interchangeably.

## Exercises

- Are there any variants of BPR and hinge loss available?
- Can you find any recommendation models that use BPR or hinge loss?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/402)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNTU0Nzg3NjIwLDYzMTY0NTEyN119
-->
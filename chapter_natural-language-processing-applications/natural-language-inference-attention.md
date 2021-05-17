# Inferência de Linguagem Natural: Usando a Atenção
:label:`sec_natural-language-inference-attention`

Introduzimos a tarefa de inferência em linguagem natural e o conjunto de dados SNLI em :numref:`sec_natural-language-inference-and-dataset`. Em vista de muitos modelos baseados em arquiteturas complexas e profundas, Parikh et al. proposto para abordar a inferência de linguagem natural com mecanismos de atenção e chamou-o de "modelo de atenção decomposto" :cite:`Parikh.Tackstrom.Das.ea.2016`.
Isso resulta em um modelo sem camadas recorrentes ou convolucionais, alcançando o melhor resultado no momento no conjunto de dados SNLI com muito menos parâmetros.
Nesta seção, iremos descrever e implementar este método baseado em atenção (com MLPs) para inferência de linguagem natural, conforme descrito em :numref:`fig_nlp-map-nli -ention`.

![Esta seção alimenta o GloVe pré-treinado para uma arquitetura baseada em atenção e MLPs para inferência de linguagem natural.](../img/nlp-map-nli-attention.svg)
:label:`fig_nlp-map-nli-attention`


## O Modelo

Mais simples do que preservar a ordem das palavras em premissas e hipóteses,
podemos apenas alinhar as palavras em uma sequência de texto com todas as palavras na outra e vice-versa,
em seguida, compare e agregue essas informações para prever as relações lógicas
entre premissas e hipóteses.
Semelhante ao alinhamento de palavras entre as frases fonte e alvo na tradução automática,
o alinhamento de palavras entre premissas e hipóteses
pode ser perfeitamente realizado por mecanismos de atenção.

![Inferência de linguagem natural usando mecanismos de atenção. ](../img/nli-attention.svg)
:label:`fig_nli_attention`

:numref:`fig_nli_attention` descreve o método de inferência de linguagem natural usando mecanismos de atenção.
Em um nível superior, consiste em três etapas treinadas em conjunto: alinhar, comparar e agregar.
Iremos ilustrá-los passo a passo a seguir.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

### Alinhar

A primeira etapa é alinhar as palavras em uma sequência de texto a cada palavra na outra sequência.
Suponha que a premissa seja "preciso dormir" e a hipótese "estou cansado".
Devido à semelhança semântica,
podemos desejar alinhar "i" na hipótese com "i" na premissa,
e alinhe "cansado" na hipótese com "sono" na premissa.
Da mesma forma, podemos desejar alinhar "i" na premissa com "i" na hipótese,
e alinhar "necessidade" e "sono" na premissa com "cansado" na hipótese.
Observe que esse alinhamento é *suave* usando a média ponderada,
onde, idealmente, grandes pesos estão associados às palavras a serem alinhadas.
Para facilitar a demonstração, :numref:`fig_nli_attention` mostra tal alinhamento de uma maneira *dura*.

Agora descrevemos o alinhamento suave usando mecanismos de atenção com mais detalhes.
Denotamos por  $\mathbf{A} = (\mathbf{a}_1, \ldots, \mathbf{a}_m)$
e  $\mathbf{B} = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$ a premissa e hipótese,
cujo número de palavras são $m$ e $n$, respectivamente,
onde  $\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$ ($i = 1, \ldots, m, j = 1, \ldots, n$) é um vetor de incorporação de palavras $d$-dimensional.
Para o alinhamento suave, calculamos os pesos de atenção $e_{ij} \in \mathbb{R}$ como

$$e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),$$
:eqlabel:`eq_nli_e`

onde a função $f$ é um MLP definido na seguinte função `mlp`.
A dimensão de saída de $f$ é especificada pelo argumento `num_hiddens` de` mlp`.

```{.python .input}
def mlp(num_hiddens, flatten):
    net = nn.Sequential()
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    return net
```

```{.python .input}
#@tab pytorch
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
```


Deve-se destacar que, em :eqref:`eq_nli_e`
$f$ pega as entradas $\mathbf{a}_i$ and $\mathbf{b}_j$ separadamente em vez de pegar um par delas juntas como entrada.
Este truque de *decomposição* leva a apenas aplicações $m + n$ (complexidade linear) de $f$ em vez de $mn$ aplicativos
(complexidade quadrática).


Normalizando os pesos de atenção em :eqref:`eq_nli_e`,
calculamos a média ponderada de todas as palavras incluídas na hipótese
para obter a representação da hipótese que está suavemente alinhada com a palavra indexada por $i$ na premissa:

$$
\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.
$$

Da mesma forma, calculamos o alinhamento suave de palavras da premissa para cada palavra indexada por $j$ na hipótese:

$$
\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.
$$

Abaixo, definimos a classe `Attend` para calcular o alinhamento suave das hipóteses (`beta`) com as premissas de entrada `A` e o alinhamento suave das premissas (`alfa`) com as hipóteses de entrada `B`.

```{.python .input}
class Attend(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (b`atch_size`, no. of words in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of words in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of words in sequence A,
        # no. of words in sequence B)
        e = npx.batch_dot(f_A, f_B, transpose_b=True)
        # Shape of `beta`: (`batch_size`, no. of words in sequence A,
        # `embed_size`), where sequence B is softly aligned with each word
        # (axis 1 of `beta`) in sequence A
        beta = npx.batch_dot(npx.softmax(e), B)
        # Shape of `alpha`: (`batch_size`, no. of words in sequence B,
        # `embed_size`), where sequence A is softly aligned with each word
        # (axis 1 of `alpha`) in sequence B
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), A)
        return beta, alpha
```

```{.python .input}
#@tab pytorch
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (`batch_size`, no. of words in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of words in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of words in sequence A,
        # no. of words in sequence B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of words in sequence A,
        # `embed_size`), where sequence B is softly aligned with each word
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of words in sequence B,
        # `embed_size`), where sequence A is softly aligned with each word
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
```

### Comparando


Na próxima etapa, comparamos uma palavra em uma sequência com a outra sequência que está suavemente alinhada com essa palavra.
Observe que no alinhamento suave, todas as palavras de uma sequência, embora provavelmente com pesos de atenção diferentes, serão comparadas com uma palavra na outra sequência.
Para facilitar a demonstração, :numref:`fig_nli_attention` emparelha palavras com palavras alinhadas de uma forma *dura*.
Por exemplo, suponha que a etapa de atendimento determina que "necessidade" e "sono" na premissa estão ambos alinhados com "cansado" na hipótese, o par "cansado - preciso dormir" será comparado.

Na etapa de comparação, alimentamos a concatenação (operador $[\cdot, \cdot]$) de palavras de uma sequência e palavras alinhadas de outra sequência em uma função $g$ (um MLP):

$$\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.$$

:eqlabel:`eq_nli_v_ab`


Em:eqref:`eq_nli_v_ab`, $\mathbf{v}_{A,i}$ é a comparação entre a palavra $i$ na premissa e todas as palavras da hipótese que estão suavemente alinhadas com a palavra $i$;
enquanto $\mathbf{v}_{B,j}$ é a comparação entre a palavra $j$ na hipótese e todas as palavras da premissa que estão suavemente alinhadas com a palavra $j$.
A seguinte classe `Compare` define como a etapa de comparação.

```{.python .input}
class Compare(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(np.concatenate([A, beta], axis=2))
        V_B = self.g(np.concatenate([B, alpha], axis=2))
        return V_A, V_B
```

```{.python .input}
#@tab pytorch
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
```

### Agregando

Com dois conjuntos de vetores de comparação $\mathbf{v}_{A,i}$ ($i = 1, \ldots, m$) e $\mathbf{v}_{B,j}$ ($j = 1, \ldots, n$) disponível,
na última etapa, agregaremos essas informações para inferir a relação lógica.
Começamos resumindo os dois conjuntos:

$$
\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.
$$

Em seguida, alimentamos a concatenação de ambos os resultados do resumo na função $h$ (um MLP) para obter o resultado da classificação do relacionamento lógico:

$$
\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

A etapa de agregação é definida na seguinte classe `Aggregate`.

```{.python .input}
class Aggregate(nn.Block):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.h.add(nn.Dense(num_outputs))

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.h(np.concatenate([V_A, V_B], axis=1))
        return Y_hat
```

```{.python .input}
#@tab pytorch
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
```

### Juntando Tudo

Ao reunir as etapas de atendimento, comparação e agregação,
definimos o modelo de atenção decomposto para treinar conjuntamente essas três etapas.

```{.python .input}
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab pytorch
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

## Treinamento e Avaliação do Modelo

Agora vamos treinar e avaliar o modelo de atenção decomposto definido no conjunto de dados SNLI.
Começamos lendo o *dataset*.


### Lendo o *Dataset*

Baixamos e lemos o conjunto de dados SNLI usando a função definida em :numref:`sec_natural-language-inference-and-dataset`. O tamanho do lote e o comprimento da sequência são definidos em $256$ e $50$, respectivamente.

```{.python .input}
#@tab all
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

### Criando o Modelo

We use the pretrained $100$-dimensional GloVe embedding to represent the input tokens.
Thus, we predefine the dimension of vectors $\mathbf{a}_i$ and $\mathbf{b}_j$ in :eqref:`eq_nli_e` as $100$.
The output dimension of functions $f$ in :eqref:`eq_nli_e` and $g$ in :eqref:`eq_nli_v_ab` is set to $200$.
Then we create a model instance, initialize its parameters,
and load the GloVe embedding to initialize vectors of input tokens.

```{.python .input}
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
net.initialize(init.Xavier(), ctx=devices)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
```

```{.python .input}
#@tab pytorch
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);
```

### Training and Evaluating the Model

In contrast to the `split_batch` function in :numref:`sec_multi_gpu` that takes single inputs such as text sequences (or images),
we define a `split_batch_multi_inputs` function to take multiple inputs such as premises and hypotheses in minibatches.

```{.python .input}
#@save
def split_batch_multi_inputs(X, y, devices):
    """Split multi-input `X` and `y` into multiple devices."""
    X = list(zip(*[gluon.utils.split_and_load(
        feature, devices, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, devices, even_split=False))
```

Now we can train and evaluate the model on the SNLI dataset.

```{.python .input}
lr, num_epochs = 0.001, 4
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

### Using the Model

Finally, define the prediction function to output the logical relationship between a pair of premise and hypothesis.

```{.python .input}
#@save
def predict_snli(net, vocab, premise, hypothesis):
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab pytorch
#@save
def predict_snli(net, vocab, premise, hypothesis):
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

We can use the trained model to obtain the natural language inference result for a sample pair of sentences.

```{.python .input}
#@tab all
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

## Summary

* The decomposable attention model consists of three steps for predicting the logical relationships between premises and hypotheses: attending, comparing, and aggregating.
* With attention mechanisms, we can align words in one text sequence to every word in the other, and vice versa. Such alignment is soft using weighted average, where ideally large weights are associated with the words to be aligned.
* The decomposition trick leads to a more desirable linear complexity than quadratic complexity when computing attention weights.
* We can use pretrained word embedding as the input representation for downstream natural language processing task such as natural language inference.


## Exercises

1. Train the model with other combinations of hyperparameters. Can you get better accuracy on the test set?
1. What are major drawbacks of the decomposable attention model for natural language inference?
1. Suppose that we want to get the level of semantical similarity (e.g., a continuous value between $0$ and $1$) for any pair of sentences. How shall we collect and label the dataset? Can you design a model with attention mechanisms?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/395)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1530)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjg3NzU3ODYyLC0xMTI5MjA0Njk5LC0yNT
Y0MjAyNTYsNTcyMTcyNjY2LC05MzQ1OTY5MjVdfQ==
-->
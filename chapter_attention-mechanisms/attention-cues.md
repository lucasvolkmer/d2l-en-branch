# Dicas para atenção
:label:`sec_attention-cues`


Obrigado pela sua atenção
a este livro.
Atenção é um recurso escasso:
no momento
você está lendo este livro
e ignorando o resto.
Assim, semelhante ao dinheiro,
sua atenção está sendo paga com um custo de oportunidade.
Para garantir que seu investimento de atenção
agora vale a pena,
estamos altamente motivados a prestar nossa atenção com cuidado
para produzir um bom livro.
Atenção
é a pedra angular do arco da vida e
detém a chave para o excepcionalismo de qualquer trabalho.


Já que a economia estuda a alocação de recursos escassos,
Nós estamos
na era da economia da atenção,
onde a atenção humana é tratada como uma mercadoria limitada, valiosa e escassa
que pode ser trocada.
Numerosos modelos de negócios foram
desenvolvido para capitalizar sobre ele.
Em serviços de streaming de música ou vídeo,
ou prestamos atenção aos seus anúncios
ou pagar para escondê-los.
Para crescer no mundo dos jogos online,
nós ou prestamos atenção a
participar de batalhas, que atraem novos jogadores,
ou pagamos dinheiro para nos tornarmos poderosos instantaneamente.
Nada vem de graça.

Contudo,
as informações em nosso ambiente não são escassas,
atenção é.
Ao inspecionar uma cena visual,
nosso nervo óptico recebe informações
na ordem de $10^8$ bits por segundo,
excedendo em muito o que nosso cérebro pode processar totalmente.
Felizmente,
nossos ancestrais aprenderam com a experiência (também conhecido como dados)
que *nem todas as entradas sensoriais são criadas iguais*.
Ao longo da história humana,
a capacidade de direcionar a atenção
para apenas uma fração da informação de interesse
habilitou nosso cérebro
para alocar recursos de forma mais inteligente
para sobreviver, crescer e se socializar,
como a detecção de predadores, presas e companheiros.



## Dicas de Atenção em Biologia


Para explicar como nossa atenção é implantada no mundo visual,
uma estrutura de dois componentes surgiu
e tem sido generalizado.
Essa ideia remonta a William James na década de 1890,
que é considerado o "pai da psicologia americana" :cite:`James.2007`.
Nesta estrutura,
assuntos direcionam seletivamente o holofote da atenção
usando a *dica não-voluntária* e a *dica volitiva*.

A sugestão não-voluntária é baseada em
a saliência e conspicuidade de objetos no ambiente.
Imagine que há cinco objetos à sua frente:
um jornal, um artigo de pesquisa, uma xícara de café, um caderno e um livro como em :numref:`fig_eye-coffee`.
Embora todos os produtos de papel sejam impressos em preto e branco,
a xícara de café é vermelha.
Em outras palavras,
este café é intrinsecamente saliente e conspícuo neste ambiente visual,
chamando a atenção automática e involuntariamente.
Então você traz a fóvea (o centro da mácula onde a acuidade visual é mais alta) para o café como mostrado em :numref:`fig_eye-coffee`.

![Usando a sugestão não-voluntária baseada na saliência (xícara vermelha, não papel), a atenção é involuntariamente voltada para o café.](../img/eye-coffee.svg)
:width:`400px`
:label:`fig_eye-coffee`

After drinking coffee,
you become caffeinated and
want to read a book.
So you turn your head, refocus your eyes,
and look at the book as depicted in :numref:`fig_eye-book`.
Different from
the case in :numref:`fig_eye-coffee`
where the coffee biases you towards
selecting based on saliency,
in this task-dependent case you select the book under
cognitive and volitional control.
Using the volitional cue based on variable selection criteria,
this form of attention is more deliberate.
It is also more powerful with the subject's voluntary effort.

![Using the volitional cue (want to read a book) that is task-dependent, attention is directed to the book under volitional control.](../img/eye-book.svg)
:width:`400px`
:label:`fig_eye-book`


## Queries, Keys, and Values

Inspired by the nonvolitional and volitional attention cues that explain the attentional deployment,
in the following we will
describe a framework for
designing attention mechanisms
by incorporating these two attention cues.

To begin with, consider the simpler case where only
nonvolitional cues are available.
To bias selection over sensory inputs,
we can simply use
a parameterized fully-connected layer
or even non-parameterized
max or average pooling.

Therefore,
what sets attention mechanisms
apart from those fully-connected layers
or pooling layers
is the inclusion of the volitional cues.
In the context of attention mechanisms,
we refer to volitional cues as *queries*.
Given any query,
attention mechanisms
bias selection over sensory inputs (e.g., intermediate feature representations)
via *attention pooling*.
These sensory inputs are called *values* in the context of attention mechanisms.
More generally,
every value is paired with a *key*,
which can be thought of the nonvolitional cue of that sensory input.
As shown in :numref:`fig_qkv`,
we can design attention pooling
so that the given query (volitional cue) can interact with keys (nonvolitional cues),
which guides bias selection over values (sensory inputs).

![Attention mechanisms bias selection over values (sensory inputs) via attention pooling, which incorporates queries (volitional cues) and keys (nonvolitional cues).](../img/qkv.svg)
:label:`fig_qkv`

Note that there are many alternatives for the design of attention mechanisms.
For instance,
we can design a non-differentiable attention model
that can be trained using reinforcement learning methods :cite:`Mnih.Heess.Graves.ea.2014`.
Given the dominance of the framework in :numref:`fig_qkv`,
models under this framework
will be the center of our attention in this chapter.


## Visualization of Attention

Average pooling
can be treated as a weighted average of inputs,
where weights are uniform.
In practice,
attention pooling aggregates values using weighted average, where weights are computed between the given query and different keys.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```
To visualize attention weights,
we define the `show_heatmaps` function.
Its input `matrices` has the shape (number of rows for display, number of columns for display, number of queries, number of keys).

```{.python .input}
#@tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

For demonstration,
we consider a simple case where
the attention weight is one only when the query and the key are the same; otherwise it is zero.

```{.python .input}
#@tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

In the subsequent sections,
we will often invoke this function to visualize attention weights.

## Summary

* Human attention is a limited, valuable, and scarce resource.
* Subjects selectively direct attention using both the nonvolitional and volitional cues. The former is based on saliency and the latter is task-dependent.
* Attention mechanisms are different from fully-connected layers or pooling layers due to inclusion of the volitional cues.
* Attention mechanisms bias selection over values (sensory inputs) via attention pooling, which incorporates queries (volitional cues) and keys (nonvolitional cues). Keys and values are paired.
* We can visualize attention weights between queries and keys.

## Exercises

1. What can be the volitional cue when decoding a sequence token by token in machine translation? What are the nonvolitional cues and the sensory inputs?
1. Randomly generate a $10 \times 10$ matrix and use the softmax operation to ensure each row is a valid probability distribution. Visualize the output attention weights.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1710)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDI3NzM3NTg0LDMwNDA2ODQ0NSwtMTg1NT
I3NTYzMCwtODQ3OTkyMDA3XX0=
-->
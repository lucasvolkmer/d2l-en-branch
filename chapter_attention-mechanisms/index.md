# Mecanismos de Atenção
:label:`chap_attention`


O nervo óptico do sistema visual de um primata
recebe entrada sensorial massiva,
excedendo em muito o que o cérebro pode processar totalmente.
Felizmente,
nem todos os estímulos são criados iguais.
Focalização e concentração de consciência
permitiram que os primatas direcionassem a atenção
para objetos de interesse,
como presas e predadores,
no ambiente visual complexo.
A capacidade de prestar atenção a
apenas uma pequena fração das informações
tem significado evolutivo,
permitindo seres humanos
para viver e ter sucesso.

Os cientistas têm estudado a atenção
no campo da neurociência cognitiva
desde o século XIX.
Neste capítulo,
começaremos revisando uma estrutura popular
explicando como a atenção é implantada em uma cena visual.
Inspirado pelas dicas de atenção neste quadro,
nós iremos projetar modelos
que alavancam tais dicas de atenção.
Notavelmente, a regressão do kernel Nadaraya-Waston
em 1964 é uma demonstração simples de aprendizado de máquina com *mecanismos de atenção*.

Next, we will go on to introduce attention functions
that have been extensively used in
the design of attention models in deep learning.
Specifically,
we will show how to use these functions
to design the *Bahdanau attention*,
a groundbreaking attention model in deep learning
that can align bidirectionally and is differentiable.

In the end,
equipped with
the more recent
*multi-head attention*
and *self-attention* designs,
we will describe the *transformer* architecture
based solely on attention mechanisms.
Since their proposal in 2017,
transformers
have been pervasive in modern
deep learning applications,
such as in areas of
language,
vision, speech,
and reinforcement learning.

```toc
:maxdepth: 2

attention-cues
nadaraya-waston
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTM2MTUxNTI0NywtMTMxNDMzNDU4Ml19
-->
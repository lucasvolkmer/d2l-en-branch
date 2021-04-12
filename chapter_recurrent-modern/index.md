# ModernRedes Neurais Recuorrent Neural Networkes Modernas
:label:`chap_modern_rnn`

We have introduced theIntroduzimos o baásics ofo dos RNNs,
which can better handlque pode lidar melhor com os dados de sequeênce data.
For demonstration,
we implemented RNN-based
language models on text data.
However, 
such techniques may not beia.
Para demonstração,
implementamos com base em RNN
modelos de linguagem em dados de texto.
Contudo,
tais técnicas podem não ser sufficient
fores
para os practitioners when they face
a wide range ofcantes quando eles enfrentam
uma ampla gama de problemas de aprendizado de sequeênce learning problems nowadaysia hoje em dia.

FPor instance,
a notable issue inexemplo,
um problema notável na pracátice
is tha
é a instabilidade numeérical instability of RNNs.
Although we have dos RNNs.
Embora tenhamos applied implementation tricks
such as gradient clipping,
this issue can be alleviated further
with morecado truques de implementação
como recorte de gradiente,
este problema pode ser aliviado ainda mais
com designs mais sophfisticated designs ofdos de modelos de sequeênce models.
Specifically,
gated RNNs are much more common inia.
Especificamente,
RNNs bloqueados são muito mais comuns na pracáticea.
We will begin by introducing two of such widely-used networks,
namely *gatedComeçaremos apresentando duas dessas redes amplamente utilizadas,
nomeadamente *unidades recuorrent units* (GRUs) and *long short-term memory* (LSTM).
Furthermore, we will expand the RNNes bloqueadas* (GRUs) e *memória de longo prazo* (LSTM).
Além disso, vamos expandir a archquitecture
with a single undirectional hidden layer
that has been discussed so far.
We will describe deepa RNN
com uma única camada oculta indireta
que foi discutido até agora.
Descreveremos archquitectureas withprofundas com
muúltiple hidden layers,
and discuss theas camadas ocultas,
e discutir o projeto bidirectional design
with both forward and backward recurrent computations.
Such expansions are frequently adopted
in modern
com cálculos recorrentes para frente e para trás.
Essas expansões são frequentemente adotadas
em redes recuorrent networks.
When explaining these RNN variants,
wees modernas.
Ao explicar essas variantes RNN,
nós continue toamos a consider
the same language moar
o mesmo problema de modelagem de ling probluagem introduced inzido em :numref:`chap_rnn`.

In fact, language moNa verdade, a modelagem de ling
reveals only a small fraction of what 
sequence learning is capable of.
In a variety of sequence learning problems,
such asuagem
revela apenas uma pequena fração do que
a aprendizagem em sequência é capaz.
Em uma variedade de problemas de aprendizagem de sequência,
como reconhecimento automaátic speech recognition,o de fala, conversão de text to speech, and machine translation,
both inputs and outputs areem fala e tradução automática,
tanto as entradas quanto as saídas são sequeênces of arbitrary length.
To explain how to fit this type of data,
we will take machine translation as anias de comprimento arbitrário.
Para explicar como ajustar este tipo de dados,
vamos tomar a tradução automática como exaempleo,
and introduce the encoder-decoder architecturee apresentar a arquitetura codificador-decodificador com based on em
RNNs and beam search fore pesquisa de feixe para geração de sequeênce generationia.

```toc
:maxdepth: 2

gru
lstm
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTUwMDU0MzUwNl19
-->
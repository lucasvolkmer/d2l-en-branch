# Redes Neurais Recorrentes
:label:`chap_rnn`


Até agora encontramos dois tipos de dados: dados tabulares e dados de imagem.
Para o último, projetamos camadas especializadas para aproveitar a regularidade delas.
Em outras palavras, se permutássemos os pixels em uma imagem, seria muito mais difícil raciocinar sobre seu conteúdo de algo que se pareceria muito com o fundo de um padrão de teste na época da TV analógica.

O mais importante é que, até agora, assumimos tacitamente que todos os nossos dados são retirados de alguma distribuição,
e todos os exemplos são distribuídos de forma independente e idêntica (d.i.i.).
Infelizmente, isso não é verdade para a maioria dos dados. Por exemplo, as palavras neste parágrafo são escritas em sequência e seria muito difícil decifrar seu significado se fossem permutadas aleatoriamente.
Da mesma forma, os quadros de imagem em um vídeo, o sinal de áudio em uma conversa e o comportamento de navegação em um site seguem uma ordem sequencial.
Portanto, é razoável supor que modelos especializados para esses dados terão um desempenho melhor em descrevê-los.

Outro problema surge do fato de que podemos não apenas receber uma sequência como uma entrada, mas esperar que continuemos a sequência.
Por exemplo, a tarefa poderia ser continuar a série $2, 4, 6, 8, 10, \ldots$ Isso é bastante comum na análise de série temporal, para prever o mercado de ações, a curva de febre de um paciente ou a aceleração necessária para um carro de corrida. Novamente, queremos ter modelos que possam lidar com esses dados.

In short, while CNNs can efficiently process spatial information, *recurrent neural networks* (RNNs) are designed to better handle sequential information.
RNNs introduce state variables to store past information, together with the current inputs, to determine the current outputs.

Many of the examples for using recurrent networks are based on text data. Hence, we will emphasize language models in this chapter. After a more formal review of sequence data we introduce practical techniques for preprocessing text data.
Next, we discuss basic concepts of a language model and use this discussion as the inspiration for the design of RNNs.
In the end, we describe the gradient calculation method for RNNs to explore problems that may be encountered when training such networks.

```toc
:maxdepth: 2

sequence
text-preprocessing
language-models-and-dataset
rnn
rnn-scratch
rnn-concise
bptt
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTAxNTU0NTg4NCwtMzUwMjc0MTNdfQ==
-->
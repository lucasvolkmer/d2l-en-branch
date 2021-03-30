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


Em suma, embora as CNNs possam processar informações espaciais com eficiência, *as redes neurais recorrentes* (RNNs)[^1] são projetadas para lidar melhor com as informações sequenciais.
As RNNs introduzem variáveis de estado para armazenar informações anteriores, junto com as entradas atuais, para determinar as saídas atuais.

[^1]:*recurrent neural networks*

Muitos dos exemplos de uso de redes recorrentes são baseados em dados de texto. Portanto, enfatizaremos os modelos de linguagem neste capítulo. Após uma revisão mais formal dos dados de sequência, apresentamos técnicas práticas para o pré-processamento de dados de texto.
A seguir, discutimos os conceitos básicos de um modelo de linguagem e usamos essa discussão como inspiração para o design de RNNs.
No final, descrevemos o método de cálculo de gradiente para RNNs para explorar problemas que podem ser encontrados durante o treinamento de tais redes.

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
eyJoaXN0b3J5IjpbMTkzNjQ5MzMwOCwxMDE1NTQ1ODg0LC0zNT
AyNzQxM119
-->
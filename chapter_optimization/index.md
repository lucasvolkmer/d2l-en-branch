# Optimization AlgorithmsAlgoritmos de Otimização
:label:`chap_optimization`

If you read the book inSe você leu o livro em sequeênce up to this point you already used a number of optimization algorithms toia até agora, já usou vários algoritmos de otimização para traein deep learning models.
They were the tools that allowed us to continue updating modelar modelos de aprendizado profundo.
Foram as ferramentas que nos permitiram continuar atualizando os paraâmeters and to minimize the value of the loss function, as evaluated on thros do modelo e minimizar o valor da função perda, conforme avaliado no conjunto de traeining set. Indeed, anyonamento. Na verdade, qualquer pessoa que se content withar em treatingar a optimization as a black box device to minimizeção como um dispositivo de caixa preta para minimizar as funções objective functions in a simple setting might well content oneself with the knowledge that there exists an array of incantations of such a procedure (withas em um ambiente simples pode muito bem se contentar com o conhecimento de que existe uma série de encantamentos de tal procedimento (com naomes such ascomo "SGD" ande "Adam").

To do well, however, some deeper knowledge is required.
Optimization algorithms are important for deep learning.
On one hand )

Para se sair bem, entretanto, é necessário algum conhecimento mais profundo.
Os algoritmos de otimização são importantes para o aprendizado profundo.
Por um lado, traeining aar um modelo complexo deep learning model can take aprendizado profundo pode levar houras, days, or even weeks.
The performance of thias ou até semanas.
O desempenho do algoritmo de optimization algorithm directly affects the model'sção afeta diretamente a eficiência de traeining efficiency.
On the other hand, understanding the principles ofamento do modelo.
Por outro lado, compreender os princípios de different optimization algorithms and the role of theires algoritmos de otimização e a função de seus hyiperparaâmeters
will enable us to tune theros
nos permitirá ajustar os hyiperparaâmeters in a targeted manner to improve the performanceros de maneira direcionada para melhorar of deep learning models.

In thissempenho dos modelos de aprendizado profundo.

Neste chapter, we explore common deep learningítulo, exploramos algoritmos comuns de optimization algorithms in depth.
Almost allção de aprendizagem profunda em profundidade.
Quase todos os problemas de optimization problems arising in deep learning are *nonção que surgem no aprendizado profundo são *não convexos*.
Nonetheless, the design and analysis of algorithms in the context of *convex* entanto, o projeto e a análise de algoritmos no contexto de problemas have proven to be very*convexos* provaram ser muito instructive.
It is for that reason that thisos.
É por essa razão que este chapterítulo includes a primer on convexi uma cartilha sobre optimization and the proof for a very simple stochastic gradient descent algorithm on a convexção convexa e a prova para um algoritmo de descida gradiente estocástico muito simples em uma função objective functiono convexa.

```toc
:maxdepth: 2

optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwMjc5NTY2MzhdfQ==
-->
# Modern Convolutional Neural Networks
:label:`chap_modern_cnn`

Agora que entendemos o básico de conectar CNNs,
vamos levá-lo por um tour pelas arquiteturas modernas da CNN.
Neste capítulo, cada seção corresponde
a uma arquitetura significativa da CNN que foi
em algum ponto (ou atualmente) o modelo básico
sobre o qual muitos projetos de pesquisa e sistemas implantados foram construídos.
Cada uma dessas redes foi brevemente uma arquitetura dominante
e muitos foram vencedores ou segundos classificados na competição ImageNet,
que tem servido como um barômetro do progresso
na aprendizagem supervisionada em visão computacional desde 2010.

Esses modelos incluem AlexNet, a primeira rede em grande escala implantada a
vencer os métodos convencionais de visão por computador em um desafio de visão em grande escala;
a rede VGG, que faz uso de vários blocos repetidos de elementos; a rede na rede (NiN) que convolve
redes neurais inteiras com patch por meio de entradas;
GoogLeNet, que usa redes com concatenações paralelas;
redes residuais (ResNet), que continuam sendo as mais populares
arquitetura pronta para uso em visão computacional;
e redes densamente conectadas (DenseNet),
que são caros para calcular, mas estabeleceram alguns benchmarks recentes.

While the idea of *deep* neural networks is quite simple
(stack together a bunch of layers),
performance can vary wildly across architectures and hyperparameter choices.
The neural networks described in this chapter
are the product of intuition, a few mathematical insights,
and a whole lot of trial and error. 
We present these models in chronological order,
partly to convey a sense of the history
so that you can form your own intuitions 
about where the field is heading 
and perhaps develop your own architectures.
For instance,
batch normalization and residual connections described in this chapter have offered two popular ideas for training and designing deep models.

Embora a ideia de redes neurais *profundas* seja bastante simples
(empilhar um monte de camadas),
o desempenho pode variar muito entre as arquiteturas e as opções de hiperparâmetros.
As redes neurais descritas neste capítulo
são o produto da intuição, alguns insights matemáticos,
e muita tentativa e erro.
Apresentamos esses modelos em ordem cronológica,
em parte para transmitir um sentido da história
para que você possa formar suas próprias intuições
sobre para onde o campo está indo
e talvez desenvolva suas próprias arquiteturas.
Por exemplo,
a normalização em lote e as conexões residuais descritas neste capítulo ofereceram duas ideias populares para treinar e projetar modelos profundos.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwMjg5MzY3NDcsLTEwOTA3NDQ3MTFdfQ
==
-->
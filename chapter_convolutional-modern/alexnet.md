# Deep Convolutional Neural Networks (AlexNet)
:label:`sec_alexnet`


Although CNNs were well known
in the computer vision and machine learning communities
following the introduction of LeNet,
they did not immediately dominate the field.
Although LeNet achieved good results on early small datasets,
the performance and feasibility of training CNNs
on larger, more realistic datasets had yet to be established.
In fact, for much of the intervening time between the early 1990s
and the watershed results of 2012,
neural networks were often surpassed by other machine learning methods,
such as support vector machines.


For computer vision, this comparison is perhaps not fair.
That is although the inputs to convolutional networks
consist of raw or lightly-processed (e.g., by centering) pixel values, practitioners would never feed raw pixels into traditional models.
Instead, typical computer vision pipelines
consisted of manually engineering feature extraction pipelines.
Rather than *learn the features*, the features were *crafted*.
Most of the progress came from having more clever ideas for features,
and the learning algorithm was often relegated to an afterthought.Embora as CNNs fossem bem conhecidas
nas comunidades de visão computacional e aprendizado de máquina
após a introdução do LeNet,
eles não dominaram imediatamente o campo.
Embora LeNet tenha alcançado bons resultados em pequenos conjuntos de dados iniciais,
o desempenho e a viabilidade de treinamento de CNNs
em conjuntos de dados maiores e mais realistas ainda não foram estabelecidos.
Na verdade, durante grande parte do tempo intermediário entre o início da década de 1990
e os resultados do divisor de águas de 2012,
redes neurais muitas vezes eram superadas por outros métodos de aprendizado de máquina,
como máquinas de vetores de suporte.


For computer vision, this comparison is perhaps not fair.
That is although the inputs to convolutional networks
consist of raw or lightly-processed (e.g., by centering) pixel values, practitioners would never feed raw pixels into traditional models.
Instead, typical computer vision pipelines
consisted of manually engineering feature extraction pipelines.
Rather than *learn the features*, the features were *crafted*.
Most of the progress came from having more clever ideas for features,
and the learning algorithm was often relegated to an afterthought.

Para a visão computacional, essa comparação talvez não seja justa.
Isso embora as entradas para redes convolucionais
consistir em valores de pixel brutos ou levemente processados (por exemplo, pela centralização), os profissionais nunca alimentariam pixels brutos em modelos tradicionais.
Em vez disso, pipelines típicos de visão computacional
consistia em pipelines de extração de recursos de engenharia manual.
Em vez de * aprender os recursos *, os recursos foram * criados *.
A maior parte do progresso veio de ter ideias mais inteligentes para recursos,
e o algoritmo de aprendizagem foi frequentemente relegado a uma reflexão tardia.

Although some neural network accelerators were available in the 1990s,
they were not yet sufficiently powerful to make
deep multichannel, multilayer CNNs
with a large number of parameters.
Moreover, datasets were still relatively small.
Added to these obstacles, key tricks for training neural networks
including parameter initialization heuristics,
clever variants of stochastic gradient descent,
non-squashing activation functions,
and effective regularization techniques were still missing.

Thus, rather than training *end-to-end* (pixel to classification) systems,
classical pipelines looked more like this:


Although some neural network accelerators were available in the 1990s,
they were not yet sufficiently powerful to make
deep multichannel, multilayer CNNs
with a large number of parameters.
Moreover, datasets were still relatively small.
Added to these obstacles, key tricks for training neural networks
including parameter initialization heuristics,
clever variants of stochastic gradient descent,
non-squashing activation functions,
and effective regularization techniques were still missing.

Thus, rather than training *end-to-end* (pixel to classification) systems,
classical pipelines looked more like this:

1. Obtain an interesting dataset. In early days, these datasets required expensive sensors (at the time, 1 megapixel images were state-of-the-art).
2. Preprocess the dataset with hand-crafted features based on some knowledge of optics, geometry, other analytic tools, and occasionally on the serendipitous discoveries of lucky graduate students.
3. Feed the data through a standard set of feature extractors such as the SIFT (scale-invariant feature transform) :cite:`Lowe.2004`, the SURF (speeded up robust features) :cite:`Bay.Tuytelaars.Van-Gool.2006`, or any number of other hand-tuned pipelines.
4. Dump the resulting representations into your favorite classifier, likely a linear model or kernel method, to train a classifier.

6. Obtenha um conjunto de dados interessante. No início, esses conjuntos de dados exigiam sensores caros (na época, as imagens de 1 megapixel eram de última geração).
7. Pré-processe o conjunto de dados com recursos feitos à mão com base em algum conhecimento de ótica, geometria, outras ferramentas analíticas e, ocasionalmente, nas descobertas fortuitas de alunos de pós-graduação sortudos.
8. Alimente os dados por meio de um conjunto padrão de extratores de recursos, como o SIFT (transformação de recurso invariante de escala): cite: `Lowe.2004`, o SURF (recursos robustos acelerados): cite:` Bay.Tuytelaars.Van- Gool.2006`, ou qualquer outro duto ajustado manualmente.
9. Despeje as representações resultantes em seu classificador favorito, provavelmente um modelo linear ou método de kernel, para treinar um classificador.

If you spoke to machine learning researchers,
they believed that machine learning was both important and beautiful.
Elegant theories proved the properties of various classifiers.
The field of machine learning was thriving, rigorous, and eminently useful. However, if you spoke to a computer vision researcher,
you would hear a very different story.
The dirty truth of image recognition, they would tell you,
is that features, not learning algorithms, drove progress.
Computer vision researchers justifiably believed
that a slightly bigger or cleaner dataset
or a slightly improved feature-extraction pipeline
mattered far more to the final accuracy than any learning algorithm.

Se você conversou com pesquisadores de aprendizado de máquina,
eles acreditavam que o aprendizado de máquina era importante e bonito.
Teorias elegantes provaram as propriedades de vários classificadores.
O campo do aprendizado de máquina era próspero, rigoroso e eminentemente útil. No entanto, se você falou com um pesquisador de visão computacional,
você ouviria uma história muito diferente.
A verdade suja do reconhecimento de imagem, eles diriam a você,
é que os recursos, e não os algoritmos de aprendizagem, impulsionaram o progresso.
Pesquisadores de visão computacional acreditavam com razão
que um conjunto de dados ligeiramente maior ou mais limpo
ou um pipeline de extração de recursos ligeiramente melhorado
importava muito mais para a precisão final do que qualquer algoritmo de aprendizado.

## Learning Representations

Another way to cast the state of affairs is that
the most important part of the pipeline was the representation.
And up until 2012 the representation was calculated mechanically.
In fact, engineering a new set of feature functions, improving results, and writing up the method was a prominent genre of paper.
SIFT :cite:`Lowe.2004`,
SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`,
HOG (histograms of oriented gradient) :cite:`Dalal.Triggs.2005`,
[bags of visual words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)
and similar feature extractors ruled the roost.

Outra forma de definir o estado de coisas é que
a parte mais importante do pipeline foi a representação.
E até 2012 a representação era calculada mecanicamente.
Na verdade, desenvolver um novo conjunto de funções de recursos, melhorar os resultados e escrever o método era um gênero de papel proeminente.
SIFT: cite: `Lowe.2004`,
SURF: cite: `Bay.Tuytelaars.Van-Gool.2006`,
HOG (histogramas de gradiente orientado): cite: `Dalal.Triggs.2005`,
[pacotes de palavras visuais] (https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)
e extratores de recursos semelhantes governaram o poleiro.

Another group of researchers,
including Yann LeCun, Geoff Hinton, Yoshua Bengio,
Andrew Ng, Shun-ichi Amari, and Juergen Schmidhuber,
had different plans.
They believed that features themselves ought to be learned.
Moreover, they believed that to be reasonably complex,
the features ought to be hierarchically composed
with multiple jointly learned layers, each with learnable parameters.
In the case of an image, the lowest layers might come
to detect edges, colors, and textures.
Indeed,
Alex Krizhevsky, Ilya Sutskever, and Geoff Hinton
proposed a new variant of a CNN,
*AlexNet*,
that achieved excellent performance in the 2012 ImageNet challenge.
AlexNet was named after Alex Krizhevsky,
the first author of the breakthrough ImageNet classification paper :cite:`Krizhevsky.Sutskever.Hinton.2012`.

Outro grupo de pesquisadores,
incluindo Yann LeCun, Geoff Hinton, Yoshua Bengio,
Andrew Ng, Shun-ichi Amari e Juergen Schmidhuber,
tinha planos diferentes.
Eles acreditavam que as próprias características deveriam ser aprendidas.
Além disso, eles acreditavam que era razoavelmente complexo,
os recursos devem ser compostos hierarquicamente
com várias camadas aprendidas em conjunto, cada uma com parâmetros aprendíveis.
No caso de uma imagem, as camadas mais baixas podem vir
para detectar bordas, cores e texturas.
De fato,
Alex Krizhevsky, Ilya Sutskever e Geoff Hinton
propôs uma nova variante de uma CNN,
* AlexNet *,
que obteve excelente desempenho no desafio ImageNet de 2012.
AlexNet foi nomeado após Alex Krizhevsky,
o primeiro autor do inovador artigo de classificação ImageNet: cite: `Krizhevsky.Sutskever.Hinton.2012`.

Interestingly in the lowest layers of the network,
the model learned feature extractors that resembled some traditional filters.
:numref:`fig_filters` is reproduced from the AlexNet paper :cite:`Krizhevsky.Sutskever.Hinton.2012`
and describes lower-level image descriptors.

Curiosamente, nas camadas mais baixas da rede,
o modelo aprendeu extratores de recursos que se assemelhavam a alguns filtros tradicionais.
: numref: `fig_filters` é reproduzido do artigo AlexNet: cite:` Krizhevsky.Sutskever.Hinton.2012`
e descreve descritores de imagem de nível inferior.

![Image filters learned by the first layer of AlexNet.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

Higher layers in the network might build upon these representations
to represent larger structures, like eyes, noses, blades of grass, and so on.
Even higher layers might represent whole objects
like people, airplanes, dogs, or frisbees.
Ultimately, the final hidden state learns a compact representation
of the image that summarizes its contents
such that data belonging to different categories can be easily separated.

As camadas superiores da rede podem se basear nessas representações
para representar estruturas maiores, como olhos, narizes, folhas de grama e assim por diante.
Mesmo camadas mais altas podem representar objetos inteiros
como pessoas, aviões, cães ou frisbees.
Em última análise, o estado oculto final aprende uma representação compacta
da imagem que resume seu conteúdo
de forma que os dados pertencentes a diferentes categorias possam ser facilmente separados.

While the ultimate breakthrough for many-layered CNNs
came in 2012, a core group of researchers had dedicated themselves
to this idea, attempting to learn hierarchical representations of visual data
for many years.
The ultimate breakthrough in 2012 can be attributed to two key factors.

Enquanto a inovação definitiva para CNNs de várias camadas
veio em 2012, um grupo central de pesquisadores se dedicou
a esta ideia, tentando aprender representações hierárquicas de dados visuais
por muitos anos.
O grande avanço em 2012 pode ser atribuído a dois fatores principais.

### Missing Ingredient: Data

Deep models with many layers require large amounts of data
in order to enter the regime
where they significantly outperform traditional methods
based on convex optimizations (e.g., linear and kernel methods).
However, given the limited storage capacity of computers,
the relative expense of sensors,
and the comparatively tighter research budgets in the 1990s,
most research relied on tiny datasets.
Numerous papers addressed the UCI collection of datasets,
many of which contained only hundreds or (a few) thousands of images
captured in unnatural settings with low resolution.

Modelos profundos com muitas camadas requerem grandes quantidades de dados
a fim de entrar no regime
onde superam significativamente os métodos tradicionais
com base em otimizações convexas (por exemplo, métodos lineares e de kernel).
No entanto, dada a capacidade limitada de armazenamento dos computadores,
a despesa relativa de sensores,
e os orçamentos de pesquisa comparativamente mais apertados na década de 1990,
a maioria das pesquisas baseou-se em pequenos conjuntos de dados.
Numerosos artigos abordaram a coleção de conjuntos de dados da UCI,
muitos dos quais continham apenas centenas ou (alguns) milhares de imagens
capturado em configurações não naturais com baixa resolução.

In 2009, the ImageNet dataset was released,
challenging researchers to learn models from 1 million examples,
1000 each from 1000 distinct categories of objects.
The researchers, led by Fei-Fei Li, who introduced this dataset
leveraged Google Image Search to prefilter large candidate sets
for each category and employed
the Amazon Mechanical Turk crowdsourcing pipeline
to confirm for each image whether it belonged to the associated category.
This scale was unprecedented.
The associated competition, dubbed the ImageNet Challenge
pushed computer vision and machine learning research forward,
challenging researchers to identify which models performed best
at a greater scale than academics had previously considered.

Em 2009, o conjunto de dados ImageNet foi lançado,
desafiando os pesquisadores a aprender modelos a partir de 1 milhão de exemplos,
1000 cada de 1000 categorias distintas de objetos.
Os pesquisadores, liderados por Fei-Fei Li, que apresentou este conjunto de dados
aproveitou a Pesquisa de imagens do Google para pré-filtrar grandes conjuntos de candidatos
para cada categoria e empregado
o pipeline de crowdsourcing Amazon Mechanical Turk
para confirmar para cada imagem se pertencia à categoria associada.
Essa escala não tinha precedentes.
A competição associada, apelidada de Desafio ImageNet
impulsionou a pesquisa sobre visão computacional e aprendizado de máquina,
desafiando os pesquisadores a identificar quais modelos tiveram melhor desempenho
em uma escala maior do que os acadêmicos haviam considerado anteriormente.

### Missing Ingredient: Hardware

Deep learning models are voracious consumers of compute cycles.
Training can take hundreds of epochs, and each iteration
requires passing data through many layers of computationally-expensive
linear algebra operations.
This is one of the main reasons why in the 1990s and early 2000s,
simple algorithms based on the more-efficiently optimized
convex objectives were preferred.

Modelos de aprendizado profundo são consumidores vorazes de ciclos de computação.
O treinamento pode levar centenas de épocas, e cada iteração
requer a passagem de dados por muitas camadas de alto custo computacional
operações de álgebra linear.
Esta é uma das principais razões pelas quais, na década de 1990 e no início de 2000,
algoritmos simples baseados em algoritmos otimizados de forma mais eficiente
objetivas convexas foram preferidas. 

*Graphical processing units* (GPUs) proved to be a game changer
in making deep learning feasible.
These chips had long been developed for accelerating
graphics processing to benefit computer games.
In particular, they were optimized for high throughput $4 \times 4$ matrix-vector products, which are needed for many computer graphics tasks.
Fortunately, this math is strikingly similar
to that required to calculate convolutional layers.
Around that time, NVIDIA and ATI had begun optimizing GPUs
for general computing operations,
going as far as to market them as *general-purpose GPUs* (GPGPU).

* Unidades de processamento gráfico * (GPUs) provaram ser uma virada de jogo
para tornar o aprendizado profundo viável.
Esses chips há muito foram desenvolvidos para acelerar
processamento gráfico para beneficiar os jogos de computador.
Em particular, eles foram otimizados para produtos de vetor de matriz de alta capacidade $ 4 \ vezes 4 $, que são necessários para muitas tarefas de computação gráfica.
Felizmente, essa matemática é muito semelhante
ao necessário para calcular camadas convolucionais.
Naquela época, a NVIDIA e a ATI começaram a otimizar GPUs
para operações gerais de computação,
indo tão longe a ponto de comercializá-los como * GPUs de uso geral * (GPGPU).

To provide some intuition, consider the cores of a modern microprocessor
(CPU).
Each of the cores is fairly powerful running at a high clock frequency
and sporting large caches (up to several megabytes of L3).
Each core is well-suited to executing a wide range of instructions,
with branch predictors, a deep pipeline, and other bells and whistles
that enable it to run a large variety of programs.
This apparent strength, however, is also its Achilles heel:
general-purpose cores are very expensive to build.
They require lots of chip area,
a sophisticated support structure
(memory interfaces, caching logic between cores,
high-speed interconnects, and so on),
and they are comparatively bad at any single task.
Modern laptops have up to 4 cores,
and even high-end servers rarely exceed 64 cores,
simply because it is not cost effective.

Para fornecer alguma intuição, considere os núcleos de um microprocessador moderno
(CPU).
Cada um dos núcleos é bastante poderoso rodando em uma alta frequência de clock
e exibindo grandes caches (até vários megabytes de L3).
Cada núcleo é adequado para executar uma ampla gama de instruções,
com preditores de ramificação, um pipeline profundo e outros sinos e assobios
que permitem executar uma grande variedade de programas.
Essa aparente força, no entanto, é também seu calcanhar de Aquiles:
núcleos de uso geral são muito caros para construir.
Eles exigem muita área de chip,
uma estrutura de suporte sofisticada
(interfaces de memória, lógica de cache entre os núcleos,
interconexões de alta velocidade e assim por diante),
e são comparativamente ruins em qualquer tarefa.
Laptops modernos têm até 4 núcleos,
e até mesmo servidores de última geração raramente excedem 64 núcleos,
simplesmente porque não é rentável.

By comparison, GPUs consist of $100 \sim 1000$ small processing elements
(the details differ somewhat between NVIDIA, ATI, ARM and other chip vendors),
often grouped into larger groups (NVIDIA calls them warps).
While each core is relatively weak,
sometimes even running at sub-1GHz clock frequency,
it is the total number of such cores that makes GPUs orders of magnitude faster than CPUs.
For instance, NVIDIA's recent Volta generation offers up to 120 TFlops per chip for specialized instructions
(and up to 24 TFlops for more general-purpose ones),
while floating point performance of CPUs has not exceeded 1 TFlop to date.
The reason for why this is possible is actually quite simple:
first, power consumption tends to grow *quadratically* with clock frequency.
Hence, for the power budget of a CPU core that runs 4 times faster (a typical number),
you can use 16 GPU cores at $1/4$ the speed,
which yields $16 \times 1/4 = 4$ times the performance.
Furthermore, GPU cores are much simpler
(in fact, for a long time they were not even *able*
to execute general-purpose code),
which makes them more energy efficient.
Last, many operations in deep learning require high memory bandwidth.
Again, GPUs shine here with buses that are at least 10 times as wide as many CPUs.

Em comparação, as GPUs consistem em $ 100 \ sim 1000 $ pequenos elementos de processamento
(os detalhes diferem um pouco entre NVIDIA, ATI, ARM e outros fornecedores de chips),
frequentemente agrupados em grupos maiores (a NVIDIA os chama de warps).
Embora cada núcleo seja relativamente fraco,
às vezes até rodando em frequência de clock abaixo de 1 GHz,
é o número total de tais núcleos que torna as GPUs ordens de magnitude mais rápidas do que as CPUs.
Por exemplo, a recente geração Volta da NVIDIA oferece até 120 TFlops por chip para instruções especializadas
(e até 24 TFlops para aqueles de uso geral),
enquanto o desempenho de ponto flutuante de CPUs não excedeu 1 TFlop até o momento.
A razão pela qual isso é possível é bastante simples:
primeiro, o consumo de energia tende a crescer * quadraticamente * com a frequência do clock.
Portanto, para o orçamento de energia de um núcleo da CPU que funciona 4 vezes mais rápido (um número típico),
você pode usar 16 núcleos de GPU por $ 1/4 $ a velocidade,
que rende $ 16 \ vezes 1/4 = 4 $ vezes o desempenho.
Além disso, os núcleos da GPU são muito mais simples
(na verdade, por muito tempo eles nem mesmo foram * capazes *
para executar código de uso geral),
o que os torna mais eficientes em termos de energia.
Por último, muitas operações de aprendizado profundo exigem alta largura de banda de memória.
Novamente, as GPUs brilham aqui com barramentos que são pelo menos 10 vezes mais largos que muitas CPUs.

Back to 2012. A major breakthrough came
when Alex Krizhevsky and Ilya Sutskever
implemented a deep CNN
that could run on GPU hardware.
They realized that the computational bottlenecks in CNNs,
convolutions and matrix multiplications,
are all operations that could be parallelized in hardware.
Using two NVIDIA GTX 580s with 3GB of memory,
they implemented fast convolutions.
The code [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/)
was good enough that for several years
it was the industry standard and powered
the first couple years of the deep learning boom.

De volta a 2012. Um grande avanço veio
quando Alex Krizhevsky e Ilya Sutskever
implementou uma CNN profunda
que pode ser executado em hardware GPU.
Eles perceberam que os gargalos computacionais nas CNNs,
convoluções e multiplicações de matrizes,
são todas as operações que podem ser paralelizadas no hardware.
Usando dois NVIDIA GTX 580s com 3 GB de memória,
eles implementaram convoluções rápidas.
O código [cuda-convnet] (https://code.google.com/archive/p/cuda-convnet/)
foi bom o suficiente por vários anos
era o padrão da indústria e alimentado
os primeiros anos do boom do aprendizado profundo.

## AlexNet

AlexNet, which employed an 8-layer CNN,
won the ImageNet Large Scale Visual Recognition Challenge 2012
by a phenomenally large margin.
This network showed, for the first time,
that the features obtained by learning can transcend manually-designed features, breaking the previous paradigm in computer vision.

The architectures of AlexNet and LeNet are very similar,
as :numref:`fig_alexnet` illustrates.
Note that we provide a slightly streamlined version of AlexNet
removing some of the design quirks that were needed in 2012
to make the model fit on two small GPUs.

AlexNet, que empregava uma CNN de 8 camadas,
venceu o Desafio de Reconhecimento Visual em Grande Escala ImageNet 2012
por uma margem fenomenalmente grande.
Esta rede mostrou, pela primeira vez,
que os recursos obtidos pelo aprendizado podem transcender recursos projetados manualmente, quebrando o paradigma anterior em visão computacional.

As arquiteturas de AlexNet e LeNet são muito semelhantes,
como: numref: `fig_alexnet` ilustra.
Observe que fornecemos uma versão ligeiramente simplificada do AlexNet
removendo algumas das peculiaridades de design que eram necessárias em 2012
para fazer o modelo caber em duas pequenas GPUs.

![From LeNet (left) to AlexNet (right).](../img/alexnet.svg)
:label:`fig_alexnet`

The design philosophies of AlexNet and LeNet are very similar,
but there are also significant differences.
First, AlexNet is much deeper than the comparatively small LeNet5.
AlexNet consists of eight layers: five convolutional layers,
two fully-connected hidden layers, and one fully-connected output layer. Second, AlexNet used the ReLU instead of the sigmoid
as its activation function.
Let us delve into the details below.

As filosofias de design de AlexNet e LeNet são muito semelhantes,
mas também existem diferenças significativas.
Primeiro, o AlexNet é muito mais profundo do que o comparativamente pequeno LeNet5.
AlexNet consiste em oito camadas: cinco camadas convolucionais,
duas camadas ocultas totalmente conectadas e uma camada de saída totalmente conectada. Em segundo lugar, AlexNet usou o ReLU em vez do sigmóide
como sua função de ativação.
Vamos nos aprofundar nos detalhes abaixo.

### Architecture

In AlexNet's first layer, the convolution window shape is $11\times11$.
Since most images in ImageNet are more than ten times higher and wider
than the MNIST images,
objects in ImageNet data tend to occupy more pixels.
Consequently, a larger convolution window is needed to capture the object.
The convolution window shape in the second layer
is reduced to $5\times5$, followed by $3\times3$.
In addition, after the first, second, and fifth convolutional layers,
the network adds maximum pooling layers
with a window shape of $3\times3$ and a stride of 2.
Moreover, AlexNet has ten times more convolution channels than LeNet.

Na primeira camada do AlexNet, a forma da janela de convolução é $ 11 \ times11 $.
Uma vez que a maioria das imagens no ImageNet são mais de dez vezes maiores e mais largas
do que as imagens MNIST,
objetos em dados ImageNet tendem a ocupar mais pixels.
Consequentemente, uma janela de convolução maior é necessária para capturar o objeto.
A forma da janela de convolução na segunda camada
é reduzido para $ 5 \ times5 $, seguido por $ 3 \ times3 $.
Além disso, após a primeira, segunda e quinta camadas convolucionais,
a rede adiciona camadas de pooling máximas
com um formato de janela de $ 3 \ times3 $ e uma distância de 2.
Além disso, o AlexNet tem dez vezes mais canais de convolução do que o LeNet.

After the last convolutional layer there are two fully-connected layers
with 4096 outputs.
These two huge fully-connected layers produce model parameters of nearly 1 GB.
Due to the limited memory in early GPUs,
the original AlexNet used a dual data stream design,
so that each of their two GPUs could be responsible
for storing and computing only its half of the model.
Fortunately, GPU memory is comparatively abundant now,
so we rarely need to break up models across GPUs these days
(our version of the AlexNet model deviates
from the original paper in this aspect).

Após a última camada convolucional, existem duas camadas totalmente conectadas
com 4096 saídas.
Essas duas enormes camadas totalmente conectadas produzem parâmetros de modelo de quase 1 GB.
Devido à memória limitada nas primeiras GPUs,
o AlexNet original usava um design de fluxo de dados duplo,
para que cada uma de suas duas GPUs pudesse ser responsável
para armazenar e computar apenas sua metade do modelo.
Felizmente, a memória da GPU é comparativamente abundante agora,
então raramente precisamos separar os modelos das GPUs hoje em dia
(nossa versão do modelo AlexNet se desvia
do artigo original neste aspecto).

### Activation Functions

Besides, AlexNet changed the sigmoid activation function to a simpler ReLU activation function. On one hand, the computation of the ReLU activation function is simpler. For example, it does not have the exponentiation operation found in the sigmoid activation function.
 On the other hand, the ReLU activation function makes model training easier when using different parameter initialization methods. This is because, when the output of the sigmoid activation function is very close to 0 or 1, the gradient of these regions is almost 0, so that backpropagation cannot continue to update some of the model parameters. In contrast, the gradient of the ReLU activation function in the positive interval is always 1. Therefore, if the model parameters are not properly initialized, the sigmoid function may obtain a gradient of almost 0 in the positive interval, so that the model cannot be effectively trained.
 
Além disso, AlexNet mudou a função de ativação sigmóide para uma função de ativação ReLU mais simples. Por um lado, o cálculo da função de ativação ReLU é mais simples. Por exemplo, ele não tem a operação de exponenciação encontrada na função de ativação sigmóide.
  Por outro lado, a função de ativação ReLU torna o treinamento do modelo mais fácil ao usar diferentes métodos de inicialização de parâmetro. Isso ocorre porque, quando a saída da função de ativação sigmóide está muito próxima de 0 ou 1, o gradiente dessas regiões é quase 0, de modo que a retropropagação não pode continuar a atualizar alguns dos parâmetros do modelo. Em contraste, o gradiente da função de ativação ReLU no intervalo positivo é sempre 1. Portanto, se os parâmetros do modelo não forem inicializados corretamente, a função sigmóide pode obter um gradiente de quase 0 no intervalo positivo, de modo que o modelo não pode ser efetivamente treinados.

### Capacity Control and Preprocessing

AlexNet controls the model complexity of the fully-connected layer
by dropout (:numref:`sec_dropout`),
while LeNet only uses weight decay.
To augment the data even further, the training loop of AlexNet
added a great deal of image augmentation,
such as flipping, clipping, and color changes.
This makes the model more robust and the larger sample size effectively reduces overfitting.
We will discuss data augmentation in greater detail in :numref:`sec_image_augmentation`.

AlexNet controla a complexidade do modelo da camada totalmente conectada
por dropout (: numref: `sec_dropout`),
enquanto o LeNet usa apenas redução de peso.
Para aumentar ainda mais os dados, o loop de treinamento do AlexNet
adicionou uma grande quantidade de aumento de imagem,
como inversão, recorte e alterações de cor.
Isso torna o modelo mais robusto e o tamanho de amostra maior reduz efetivamente o sobreajuste.
Discutiremos o aumento de dados em maiores detalhes em: numref: `sec_image_augmentation`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
# Here, we use a larger 11 x 11 window to capture objects. At the same time,
# we use a stride of 4 to greatly reduce the height and width of the output.
# Here, the number of output channels is much larger than that in LeNet
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    # Here, we use a larger 11 x 11 window to capture objects. At the same
    # time, we use a stride of 4 to greatly reduce the height and width of the
    # output. Here, the number of output channels is much larger than that in
    # LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Make the convolution window smaller, set padding to 2 for consistent
    # height and width across the input and output, and increase the number of
    # output channels
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Use three successive convolutional layers and a smaller convolution
    # window. Except for the final convolutional layer, the number of output
    # channels is further increased. Pooling layers are not used to reduce the
    # height and width of input after the first two convolutional layers
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # Here, the number of outputs of the fully-connected layer is several
    # times larger than that in LeNet. Use the dropout layer to mitigate
    # overfitting
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # Output layer. Since we are using Fashion-MNIST, the number of classes is
    # 10, instead of 1000 as in the paper
    nn.Linear(4096, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        # Here, we use a larger 11 x 11 window to capture objects. At the same
        # time, we use a stride of 4 to greatly reduce the height and width of
        # the output. Here, the number of output channels is much larger than
        # that in LeNet
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        tf.keras.layers.Dense(10)
    ])
```

We construct a single-channel data example with both height and width of 224 to observe the output shape of each layer. It matches the AlexNet architecture in :numref:`fig_alexnet`.

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

## Reading the Dataset

Although AlexNet is trained on ImageNet in the paper, we use Fashion-MNIST here
since training an ImageNet model to convergence could take hours or days
even on a modern GPU.
One of the problems with applying AlexNet directly on Fashion-MNIST
is that its images have lower resolution ($28 \times 28$ pixels)
than ImageNet images.
To make things work, we upsample them to $224 \times 224$
(generally not a smart practice,
but we do it here to be faithful to the AlexNet architecture).
We perform this resizing with the `resize` argument in the `d2l.load_data_fashion_mnist` function.

```{.python .input}
#@tab all
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

## Training

Now, we can start training AlexNet.
Compared with LeNet in :numref:`sec_lenet`,
the main change here is the use of a smaller learning rate
and much slower training due to the deeper and wider network,
the higher image resolution, and the more costly convolutions.

```{.python .input}
#@tab all
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Summary

* AlexNet has a similar structure to that of LeNet, but uses more convolutional layers and a larger parameter space to fit the large-scale ImageNet dataset.
* Today AlexNet has been surpassed by much more effective architectures but it is a key step from shallow to deep networks that are used nowadays.
* Although it seems that there are only a few more lines in AlexNet's implementation than in LeNet, it took the academic community many years to embrace this conceptual change and take advantage of its excellent experimental results. This was also due to the lack of efficient computational tools.
* Dropout, ReLU, and preprocessing were the other key steps in achieving excellent performance in computer vision tasks.

## Exercises

1. Try increasing the number of epochs. Compared with LeNet, how are the results different? Why?
1. AlexNet may be too complex for the Fashion-MNIST dataset.
    1. Try simplifying the model to make the training faster, while ensuring that the accuracy does not drop significantly.
    1. Design a better model that works directly on $28 \times 28$ images.
1. Modify the batch size, and observe the changes in accuracy and GPU memory.
1. Analyze computational performance of AlexNet.
    1. What is the dominant part for the memory footprint of AlexNet?
    1. What is the dominant part for computation in AlexNet?
    1. How about memory bandwidth when computing the results?
1. Apply dropout and ReLU to LeNet-5. Does it improve? How about preprocessing?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/276)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDEwMjA3NzYyXX0=
-->
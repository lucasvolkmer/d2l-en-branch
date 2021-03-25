# De Camadas Totalmente Conectadas às Convoluções
:label:`sec_why-conv`


Até hoje,
os modelos que discutimos até agora
permanecem opções apropriadas
quando estamos lidando com dados tabulares.
Por tabular, queremos dizer que os dados consistem
de linhas correspondentes a exemplos
e colunas correspondentes a *features*.
Com dados tabulares, podemos antecipar
que os padrões que buscamos podem envolver
interações entre as características,
mas não assumimos nenhuma estrutura *a priori*
sobre como as características interagem.

Às vezes, realmente não temos conhecimento para orientar
a construção de arquiteturas mais artesanais.
Nestes casos, um MLP
pode ser o melhor que podemos fazer.
No entanto, para dados perceptivos de alta dimensão,
essas redes sem estrutura podem se tornar difíceis de manejar.

Por exemplo, vamos voltar ao nosso exemplo de execução
de distinguir gatos de cães.
Digamos que fazemos um trabalho completo na coleta de dados,
coletando um conjunto de dados anotado de fotografias de um megapixel.
Isso significa que cada entrada na rede tem um milhão de dimensões.
De acordo com nossas discussões sobre custo de parametrização
de camadas totalmente conectadas em :numref:`subsec_parameterization-cost-fc-layers`,
até mesmo uma redução agressiva para mil dimensões ocultas
exigiria uma camada totalmente conectada
caracterizada por $10^6 \times 10^3 = 10^9$ parâmetros.
A menos que tenhamos muitas GPUs, um talento
para otimização distribuída,
e uma quantidade extraordinária de paciência,
aprender os parâmetros desta rede
pode acabar sendo inviável.

Um leitor cuidadoso pode objetar a este argumento
na base de que a resolução de um megapixel pode não ser necessária.
No entanto, embora possamos ser capazes
de escapar com apenas cem mil pixels,
nossa camada oculta de tamanho 1000 subestima grosseiramente
o número de unidades ocultas que leva
para aprender boas representações de imagens,
portanto, um sistema prático ainda exigirá bilhões de parâmetros.
Além disso, aprender um classificador ajustando tantos parâmetros
pode exigir a coleta de um enorme conjunto de dados.
E ainda hoje tanto os humanos quanto os computadores são capazes
de distinguir gatos de cães muito bem,
aparentemente contradizendo essas intuições.
Isso ocorre porque as imagens exibem uma estrutura rica
que pode ser explorada por humanos
e modelos de aprendizado de máquina semelhantes.
Redes neurais convolucionais (CNNs) são uma forma criativa
que o *machine learning* adotou para explorar
algumas das estruturas conhecidas em imagens naturais.


## Invariância

Imagine que você deseja detectar um objeto em uma imagem.
Parece razoável que qualquer método
que usamos para reconhecer objetos não deveria se preocupar demais
com a localização precisa do objeto na imagem.
Idealmente, nosso sistema deve explorar esse conhecimento.
Os porcos geralmente não voam e os aviões geralmente não nadam.
No entanto, devemos ainda reconhecer
um porco era aquele que aparecia no topo da imagem.
Podemos tirar alguma inspiração aqui
do jogo infantil "Cadê o Wally"
(representado em :numref:`img_waldo`).
O jogo consiste em várias cenas caóticas
repletas de atividades.
Wally aparece em algum lugar em cada uma,
normalmente à espreita em algum local improvável.
O objetivo do leitor é localizá-lo.
Apesar de sua roupa característica,
isso pode ser surpreendentemente difícil,
devido ao grande número de distrações.
No entanto, *a aparência do Wally*
não depende de *onde o Wally está localizado*.
Poderíamos varrer a imagem com um detector Wally
que poderia atribuir uma pontuação a cada *patch*,
indicando a probabilidade de o *patch* conter Wally.
CNNs sistematizam essa ideia de *invariância espacial*,
explorando para aprender representações úteis
com menos parâmetros.

![Uma imagem do jogo "Onde está Wally".](../img/where-wally-walker-books.jpg)
:width:`400px`
:label:`img_waldo`



Agora podemos tornar essas intuições mais concretas
enumerando alguns desideratos para orientar nosso design
de uma arquitetura de rede neural adequada para visão computacional:

1. Nas primeiras camadas, nossa rede
     deve responder de forma semelhante ao mesmo *patch*,
     independentemente de onde aparece na imagem. Este princípio é denominado *invariância da tradução*.
1. As primeiras camadas da rede devem se concentrar nas regiões locais,
    sem levar em conta o conteúdo da imagem em regiões distantes. Este é o princípio de *localidade*.
    Eventualmente, essas representações locais podem ser agregadas
    para fazer previsões em todo o nível da imagem.

Vamos ver como isso se traduz em matemática.



## Restringindo o MLP


Para começar, podemos considerar um MLP
com imagens bidimensionais $\mathbf{X}$ como entradas
e suas representações ocultas imediatas
$\mathbf{H}$ similarmente representadas como matrizes em matemática e como tensores bidimensionais em código, onde $\mathbf{X}$ e $\mathbf{H}$ têm a mesma forma.
Deixe isso penetrar.
Agora concebemos não apenas as entradas, mas
também as representações ocultas como possuidoras de estrutura espacial.

Deixe $[\mathbf{X}]_{i, j}$ e $[\mathbf{H}]_{i, j}$ denotarem o pixel
no local ($i$, $j$)
na imagem de entrada e representação oculta, respectivamente.
Consequentemente, para que cada uma das unidades ocultas
receba entrada de cada um dos pixels de entrada,
nós deixaríamos de usar matrizes de peso
(como fizemos anteriormente em MLPs)
para representar nossos parâmetros
como tensores de peso de quarta ordem $\mathsf{W}$.
Suponha que $\mathbf{U}$ contenha *bias*,
poderíamos expressar formalmente a camada totalmente conectada como

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned},$$

onde a mudança de $\mathsf{W}$ para $\mathsf{V}$ é inteiramente cosmética por enquanto
uma vez que existe uma correspondência um-para-um
entre coeficientes em ambos os tensores de quarta ordem.
Nós simplesmente reindexamos os subscritos $(k, l)$
de modo que $k = i+a$ and $l = j+b$.
Em outras palavras, definimos $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$.
Os índices $a$ e $b$ ultrapassam os deslocamentos positivos e negativos,
cobrindo toda a imagem.
Para qualquer localização dada ($i$, $j$) na representação oculta $[\mathbf{H}]_{i, j}$,
calculamos seu valor somando os pixels em $x$,
centralizado em torno de $(i, j)$ e ponderado por $[\mathsf{V}]_{i, j, a, b}$.

### Invariância de Tradução

Agora vamos invocar o primeiro princípio
estabelecido acima: invariância de tradução.
Isso implica que uma mudança na entrada $\mathbf{X}$
deve simplesmente levar a uma mudança na representação oculta $\mathbf{H}$.
Isso só é possível se $\mathsf{V}$ e $\mathbf{U}$ não dependem realmente de $(i, j)$,
ou seja, temos $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ e $\mathbf{U}$$ é uma constante, digamos $u$.
Como resultado, podemos simplificar a definição de $\mathbf{H}$:

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$


Esta é uma *convolução*!
Estamos efetivamente ponderando pixels em $(i+a, j+b)$
nas proximidades da localização $(i, j)$ com coeficientes$[\mathbf{V}]_{a, b}$
para obter o valor $[\mathbf{H}]_{i, j}$.
Observe que $[\mathbf{V}]_{a, b}$ precisa de muito menos coeficientes do que $[\mathsf{V}]_{i, j, a, b}, pois ele
não depende mais da localização na imagem.
Fizemos um progresso significativo!

###  Localidade

Agora, vamos invocar o segundo princípio: localidade.
Conforme motivado acima, acreditamos que não devemos ter
parecer muito longe do local $(i, j)$
a fim de coletar informações relevantes
para avaliar o que está acontecendo em $[\mathbf{H}]_{i, j}$.
Isso significa que fora de algum intervalo $|a|> \Delta$ or $|b| > \Delta$,
devemos definir $[\mathbf{V}]_{a, b} = 0$.
Equivalentemente, podemos reescrever $[\mathbf{H}]_{i, j}$ como

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_conv-layer`

Observe que :eqref:`eq_conv-layer`, em poucas palavras, é uma *camada convolucional*.
*Redes neurais convolucionais* (CNNs[^1])
são uma família especial de redes neurais que contêm camadas convolucionais.
Na comunidade de pesquisa de *deep learning*,
$\mathbf{V}$ é referido como um *kernel de convolução*,
um *filtro*, ou simplesmente os *pesos* da camada que são parâmetros frequentemente aprendíveis.
Quando a região local é pequena,
a diferença em comparação com uma rede totalmente conectada pode ser dramática.
Embora anteriormente, pudéssemos ter exigido bilhões de parâmetros
para representar apenas uma única camada em uma rede de processamento de imagem,
agora precisamos de apenas algumas centenas, sem
alterar a dimensionalidade de qualquer
as entradas ou as representações ocultas.
O preço pago por esta redução drástica de parâmetros
é que nossos recursos agora são invariantes de tradução
e que nossa camada só pode incorporar informações locais,
ao determinar o valor de cada ativação oculta.
Todo aprendizado depende da imposição de *bias* indutivos.
Quando esses *bias* concordam com a realidade,
obtemos modelos com amostras eficientes
que generalizam bem para dados invisíveis.
Mas é claro, se esses *bias* não concordam com a realidade,
por exemplo, se as imagens acabassem não sendo invariantes à tradução,
nossos modelos podem ter dificuldade até mesmo para se ajustar aos nossos dados de treinamento.

[^1]: *Convolutional Neural Networks.*

## Convoluções


Antes de prosseguir, devemos revisar brevemente
porque a operação acima é chamada de convolução.
Em matemática, a *convolução* entre duas funções,
digamos que $f, g: \mathbb{R}^d \to \mathbb{R}$ é definido como
$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$

That is, we measure the overlap between $f$ and $g$
when one function is "flipped" and shifted by $\mathbf{x}$.
Whenever we have discrete objects, the integral turns into a sum.
For instance, for vectors from
the set of square summable infinite dimensional vectors
with index running over $\mathbb{Z}$ we obtain the following definition:

$$(f * g)(i) = \sum_a f(a) g(i-a).$$

For two-dimensional tensors, we have a corresponding sum
with indices $(a, b)$ for $f$ and $(i-a, j-b)$ for $g$, respectively:

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$
:eqlabel:`eq_2d-conv-discrete`

This looks similar to :eqref:`eq_conv-layer`, with one major difference.
Rather than using $(i+a, j+b)$, we are using the difference instead.
Note, though, that this distinction is mostly cosmetic
since we can always match the notation between
:eqref:`eq_conv-layer` and :eqref:`eq_2d-conv-discrete`.
Our original definition in :eqref:`eq_conv-layer` more properly
describes a *cross-correlation*.
We will come back to this in the following section.




## "Where's Waldo" Revisited

Returning to our Waldo detector, let us see what this looks like.
The convolutional layer picks windows of a given size
and weighs intensities according to the filter $\mathsf{V}$, as demonstrated in :numref:`fig_waldo_mask`.
We might aim to learn a model so that
wherever the "waldoness" is highest,
we should find a peak in the hidden layer representations.

![Detect Waldo.](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`


### Channels
:label:`subsec_why-conv-channels`

There is just one problem with this approach.
So far, we blissfully ignored that images consist
of 3 channels: red, green, and blue.
In reality, images are not two-dimensional objects
but rather third-order tensors,
characterized by a height, width, and channel,
e.g., with shape $1024 \times 1024 \times 3$ pixels.
While the first two of these axes concern spatial relationships,
the third can be regarded as assigning
a multidimensional representation to each pixel location.
We thus index $\mathsf{X}$ as $[\mathsf{X}]_{i, j, k}$.
The convolutional filter has to adapt accordingly.
Instead of $[\mathbf{V}]_{a,b}$, we now have $[\mathsf{V}]_{a,b,c}$.

Moreover, just as our input consists of a third-order tensor,
it turns out to be a good idea to similarly formulate
our hidden representations as third-order tensors $\mathsf{H}$.
In other words, rather than just having a single hidden representation
corresponding to each spatial location,
we want an entire vector of hidden representations
corresponding to each spatial location.
We could think of the hidden representations as comprising
a number of two-dimensional grids stacked on top of each other.
As in the inputs, these are sometimes called *channels*.
They are also sometimes called *feature maps*,
as each provides a spatialized set
of learned features to the subsequent layer.
Intuitively, you might imagine that at lower layers that are closer to inputs,
some channels could become specialized to recognize edges while
others could recognize textures.


To support multiple channels in both inputs ($\mathsf{X}$) and hidden representations ($\mathsf{H}$),
we can add a fourth coordinate to $\mathsf{V}$: $[\mathsf{V}]_{a, b, c, d}$.
Putting everything together we have:

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$
:eqlabel:`eq_conv-layer-channels`

where $d$ indexes the output channels in the hidden representations $\mathsf{H}$. The subsequent convolutional layer will go on to take a third-order tensor, $\mathsf{H}$, as the input.
Being more general,
:eqref:`eq_conv-layer-channels` is
the definition of a convolutional layer for multiple channels, where $\mathsf{V}$ is a kernel or filter of the layer.

There are still many operations that we need to address.
For instance, we need to figure out how to combine all the hidden representations
to a single output, e.g., whether there is a Waldo *anywhere* in the image.
We also need to decide how to compute things efficiently,
how to combine multiple layers,
appropriate activation functions,
and how to make reasonable design choices
to yield networks that are effective in practice.
We turn to these issues in the remainder of the chapter.


## Summary

* Translation invariance in images implies that all patches of an image will be treated in the same manner.
* Locality means that only a small neighborhood of pixels will be used to compute the corresponding hidden representations.
* In image processing, convolutional layers typically require many fewer parameters than fully-connected layers.
* CNNS are a special family of neural networks that contain convolutional layers.
* Channels on input and output allow our model to capture multiple aspects of an image  at each spatial location.

## Exercises

1. Assume that the size of the convolution kernel is $\Delta = 0$.
   Show that in this case the convolution kernel
   implements an MLP independently for each set of channels.
1. Why might translation invariance not be a good idea after all?
1. What problems must we deal with when deciding how
   to treat hidden representations corresponding to pixel locations
   at the boundary of an image?
1. Describe an analogous convolutional layer for audio.
1. Do you think that convolutional layers might also be applicable for text data?
   Why or why not?
1. Prove that $f * g = g * f$.

[Discussions](https://discuss.d2l.ai/t/64)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI2NTY4MjUzMyw5NTcwMjU3MDUsLTE5Mj
E4MTMwODgsMjQyMjkwODk3LC0xNTg0MzI4NzgwLC0xMTIyNTk3
ODYzLC0xNjU3MzY2MjUwLDE2NzM2MjU2MTAsLTEyOTkyNDE5Nj
RdfQ==
-->
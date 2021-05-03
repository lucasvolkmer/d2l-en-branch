# Region-based CNNs (R-CNNs)


Redes neurais convolucionais baseadas em regiões ou regiões com recursos CNN (R-CNNs) são uma abordagem pioneira que aplica modelos profundos para detecção de objetos :cite:`Girshick.Donahue.Darrell.ea.2014`. Nesta seção, discutiremos R-CNNs e uma série de melhorias feitas a eles: Fast R-CNN :cite:`Girshick.2015`, Faster R-CNN :cite:`Ren.He.Girshick.ea.2015` e Mask R-CNN
 :cite:`He.Gkioxari.Dollar.ea.2017`. Devido às limitações de espaço, limitaremos nossa discussão aos designs desses modelos.


## R-CNNs

Os modelos R-CNN primeiro selecionam várias regiões propostas de uma imagem (por exemplo, as caixas de âncora são um tipo de método de seleção) e, em seguida, rotulam suas categorias e caixas delimitadoras (por exemplo, deslocamentos). Em seguida, eles usam uma CNN para realizar cálculos avançados para extrair recursos de cada área proposta. Depois, usamos os recursos de cada região proposta para prever suas categorias e caixas delimitadoras. :numref:`fig_r-cnn` mostra um modelo R-CNN.

![Modelo R-CNN. ](../img/r-cnn.svg)
:label:`fig_r-cnn`

Especificamente, os R-CNNs são compostos por quatro partes principais:

1. A pesquisa seletiva é realizada na imagem de entrada para selecionar várias
   regiões propostas de alta qualidade
   :cite:`Uijlings.Van-De-Sande.Gevers.ea.2013`. Essas regiões propostas são
   geralmente selecionados em várias escalas e têm diferentes formas e
   tamanhos. A categoria e a caixa delimitadora da verdade fundamental de cada região proposta é
   etiquetada.
1. Um CNN pré-treinado é selecionado e colocado, de forma truncada, antes da
   camada de saída. Ele transforma cada região proposta nas dimensões de entrada
   exigido pela rede e usa computação direta para gerar os recursos
   extraídos das regiões propostas.
1. Os recursos e a categoria rotulada de cada região proposta são combinados como um
   exemplo para treinar várias máquinas de vetores de suporte para classificação de objeto.
   Aqui, cada máquina de vetor de suporte é usada para determinar
   se um exemplo pertence a uma determinada categoria.
1. Os recursos e a caixa delimitadora rotulada de cada região proposta são combinados como
   um exemplo para treinar um modelo de regressão linear para a predição da caixa delimitadora de verdade básica.

Embora os modelos R-CNN usem CNNs pré-treinados para extrair recursos de imagem com eficácia, a principal desvantagem é a velocidade lenta. Como você pode imaginar, podemos selecionar milhares de regiões propostas a partir de uma única imagem, exigindo milhares de cálculos diretos da CNN para realizar a detecção de objetos. Essa enorme carga de computação significa que os R-CNNs não são amplamente usados em aplicativos reais.


## Fast R-CNN

O principal gargalo de desempenho de um modelo R-CNN é a necessidade de extrair recursos de forma independente para cada região proposta. Como essas regiões têm um alto grau de sobreposição, a extração de recursos independentes resulta em um alto volume de cálculos repetitivos. Fast R-CNN melhora o R-CNN por apenas executar CNN
computação progressiva na imagem como um todo.

![Modelo Fast R-CNN.](../img/fast-rcnn.svg)
:label:`fig_fast_r-cnn`


:numref:`fig_fast_r-cnn` mostra um modelo Fast R-CNN. Suas principais etapas de computação são descritas abaixo:

1. Comparado a um modelo R-CNN, um modelo Fast R-CNN usa a imagem inteira como o
   Entrada da CNN para extração de recursos, em vez de cada região proposta. Além disso,
   esta rede é geralmente treinada para atualizar os parâmetros do modelo. Enquanto o
   a entrada é uma imagem inteira, a forma de saída do CNN é $1 \times c \times h_1
   \times w_1$.
1. Supondo que a pesquisa seletiva gere $n$ regiões propostas, seus diferentes
   formas indicam regiões de interesses (RoIs) de diferentes formas na CNN
   resultado. Características das mesmas formas devem ser extraídas dessas RoIs (aqui
   assumimos que a altura é $h_2$ e a largura é $w_2$). R-CNN rápido
   apresenta o pool de RoI, que usa a saída CNN e RoIs como entrada para saída
   uma concatenação dos recursos extraídos de cada região proposta com o
   forma $n \times c \times h_2 \times w_2$.
1. Uma camada totalmente conectada é usada para transformar a forma de saída em $n \times d$, onde $d$ é determinado pelo design do modelo.
1. Durante a previsão da categoria, a forma da saída da camada totalmente conectada é
   novamente transformada em $n \times q$ e usamos a regressão softmax ($q$ é o
   número de categorias). Durante a previsão da caixa delimitadora, a forma do
   a saída da camada conectada é novamente transformada em $n \times 4$. Isso significa que
   prevemos a categoria e a caixa delimitadora para cada região proposta.

The RoI pooling layer in Fast R-CNN is somewhat different from the pooling
layers we have discussed before. In a normal pooling layer, we set the pooling
window, padding, and stride to control the output shape. In an RoI pooling
layer, we can directly specify the output shape of each region, such as
specifying the height and width of each region as $h_2, w_2$. Assuming that the
height and width of the RoI window are $h$ and $w$, this window is divided into
a grid of sub-windows with the shape $h_2 \times w_2$. The size of each
sub-window is about $(h/h_2) \times (w/w_2)$. The sub-window height and width
must always be integers and the largest element is used as the output for a
given sub-window. This allows the RoI pooling layer to extract features of the
same shape from RoIs of different shapes.

In :numref:`fig_roi`, we select an $3\times 3$ region as an RoI of the $4 \times
4$ input. For this RoI, we use a $2\times 2$ RoI pooling layer to obtain a
single $2\times 2$ output. When we divide the region into four sub-windows, they
respectively contain the elements 0, 1, 4, and 5 (5 is the largest); 2 and 6 (6
is the largest); 8 and 9 (9 is the largest); and 10.

![$2\times 2$ RoI pooling layer. ](../img/roi.svg)
:label:`fig_roi`

:begin_tab:`mxnet`
We use the `ROIPooling` function to demonstrate the RoI pooling layer computation. Assume that the CNN extracts the feature `X` with both a height and width of 4 and only a single channel.
:end_tab:

:begin_tab:`pytorch`
We use the `roi_pool` function from `torchvision` to demonstrate the RoI pooling layer computation. Assume that the CNN extracts the feature `X` with both a height and width of 4 and only a single channel.
:end_tab:

```{.python .input}
from mxnet import np, npx

npx.set_np()

X = np.arange(16).reshape(1, 1, 4, 4)
X
```

```{.python .input}
#@tab pytorch
import torch
import torchvision

X = torch.arange(16.).reshape(1, 1, 4, 4)
X
```

Assume that the height and width of the image are both 40 pixels and that selective search generates two proposed regions on the image. Each region is expressed as five elements: the region's object category and the $x, y$ coordinates of its upper-left and bottom-right corners.

```{.python .input}
rois = np.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

```{.python .input}
#@tab pytorch
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

Because the height and width of `X` are $1/10$ of the height and width of the image, the coordinates of the two proposed regions are multiplied by 0.1 according to the `spatial_scale`, and then the RoIs are labeled on `X` as `X[:, :, 0:3, 0:3]` and `X[:, :, 1:4, 0:4]`, respectively. Finally, we divide the two RoIs into a sub-window grid and extract features with a height and width of 2.

```{.python .input}
npx.roi_pooling(X, rois, pooled_size=(2, 2), spatial_scale=0.1)
```

```{.python .input}
#@tab pytorch
torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
```

## Faster R-CNN

In order to obtain precise object detection results, Fast R-CNN generally requires that many proposed regions be generated in selective search. Faster R-CNN replaces selective search with a region proposal network. This reduces the number of proposed regions generated, while ensuring precise object detection.


![Faster R-CNN model. ](../img/faster-rcnn.svg)
:label:`fig_faster_r-cnn`


:numref:`fig_faster_r-cnn` shows a Faster R-CNN model. Compared to Fast R-CNN,
Faster R-CNN only changes the method for generating proposed regions from
selective search to region proposal network. The other parts of the model remain
unchanged. The detailed region proposal network computation process is described
below:

1. We use a $3\times 3$ convolutional layer with a padding of 1 to transform the
   CNN output and set the number of output channels to $c$. This way, each
   element in the feature map the CNN extracts from the image is a new feature
   with a length of $c$.
1. We use each element in the feature map as a center to generate multiple
   anchor boxes of different sizes and aspect ratios and then label them.
1. We use the features of the elements of length $c$ at the center on the anchor
   boxes to predict the binary category (object or background) and bounding box
   for their respective anchor boxes.
1. Then, we use non-maximum suppression to remove similar bounding box results
   that correspond to category predictions of "object". Finally, we output the
   predicted bounding boxes as the proposed regions required by the RoI pooling
   layer.


It is worth noting that, as a part of the Faster R-CNN model, the region
proposal network is trained together with the rest of the model. In addition,
the Faster R-CNN object functions include the category and bounding box
predictions in object detection, as well as the binary category and bounding box
predictions for the anchor boxes in the region proposal network. Finally, the
region proposal network can learn how to generate high-quality proposed regions,
which reduces the number of proposed regions while maintaining the precision of
object detection.


## Mask R-CNN

If training data is labeled with the pixel-level positions of each object in an image, a Mask R-CNN model can effectively use these detailed labels to further improve the precision of object detection.

![Mask R-CNN model. ](../img/mask-rcnn.svg)
:label:`fig_mask_r-cnn`

As shown in :numref:`fig_mask_r-cnn`, Mask R-CNN is a modification to the Faster
R-CNN model. Mask R-CNN models replace the RoI pooling layer with an RoI
alignment layer. This allows the use of bilinear interpolation to retain spatial
information on feature maps, making Mask R-CNN better suited for pixel-level
predictions. The RoI alignment layer outputs feature maps of the same shape for
all RoIs. This not only predicts the categories and bounding boxes of RoIs, but
allows us to use an additional fully convolutional network to predict the
pixel-level positions of objects. We will describe how to use fully
convolutional networks to predict pixel-level semantics in images later in this
chapter.



## Summary

* An R-CNN model selects several proposed regions and uses a CNN to perform
  forward computation and extract the features from each proposed region. It
  then uses these features to predict the categories and bounding boxes of
  proposed regions.
* Fast R-CNN improves on the R-CNN by only performing CNN forward computation on
  the image as a whole. It introduces an RoI pooling layer to extract features
  of the same shape from RoIs of different shapes.
* Faster R-CNN replaces the selective search used in Fast R-CNN with a region
  proposal network. This reduces the number of proposed regions generated, while
  ensuring precise object detection.
* Mask R-CNN uses the same basic structure as Faster R-CNN, but adds a fully
  convolution layer to help locate objects at the pixel level and further
  improve the precision of object detection.


## Exercises

1. Study the implementation of each model in the [GluonCV toolkit](https://github.com/dmlc/gluon-cv/) related to this section.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/374)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1409)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyNDEzMDQwMjEsLTEzNDEzNTgwNzQsMT
E2OTIxOTIwMiwxNjA0MDA5MTIsLTE0OTM0MDA2ODRdfQ==
-->
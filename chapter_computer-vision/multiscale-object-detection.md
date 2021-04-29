# Detecção de Objetos Multiescala


Em :numref:`sec_anchor`, geramos várias caixas de âncora centralizadas em cada pixel da imagem de entrada. Essas caixas de âncora são usadas para amostrar diferentes regiões da imagem de entrada. No entanto, se as caixas de âncora forem geradas centralizadas em cada pixel da imagem, logo haverá muitas caixas de âncora para calcularmos. Por exemplo, assumimos que a imagem de entrada tem uma altura e uma largura de 561 e 728 pixels, respectivamente. Se cinco formas diferentes de caixas de âncora são geradas centralizadas em cada pixel, mais de dois milhões de caixas de âncora ($561 \times 728 \times 5$) precisam ser previstas e rotuladas na imagem.

Não é difícil reduzir o número de caixas de âncora. Uma maneira fácil é aplicar amostragem uniforme em uma pequena parte dos pixels da imagem de entrada e gerar caixas de âncora centralizadas nos pixels amostrados. Além disso, podemos gerar caixas de âncora de números e tamanhos variados em várias escalas. Observe que é mais provável que objetos menores sejam posicionados na imagem do que objetos maiores. Aqui, usaremos um exemplo simples: Objetos com formas de $1 \times 1$, $1 \times 2$, e $2 \times 2$ podem ter 4, 2 e 1 posição(ões) possível(is) em uma imagem com a forma $2 \times 2$.. Portanto, ao usar caixas de âncora menores para detectar objetos menores, podemos amostrar mais regiões; ao usar caixas de âncora maiores para detectar objetos maiores, podemos amostrar menos regiões.

Para demonstrar como gerar caixas de âncora em várias escalas, vamos ler uma imagem primeiro. Tem uma altura e largura de $561 \times 728$ pixels.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[0:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[0:2]
h, w
```


Em :numref:`sec_conv_layer`, a saída da matriz 2D da rede neural convolucional (CNN) é chamada
um mapa de recursos. Podemos determinar os pontos médios de caixas de âncora uniformemente amostradas
em qualquer imagem, definindo a forma do mapa de feições.

A função `display_anchors` é definida abaixo. Vamos gerar caixas de âncora `anchors` centradas em cada unidade (pixel) no mapa de feições `fmap`. Uma vez que as coordenadas dos eixos $x$ e $y$ nas caixas de âncora `anchors` foram divididas pela largura e altura do mapa de feições `fmap`, valores entre 0 e 1 podem ser usados ​​para representar as posições relativas das caixas de âncora em o mapa de recursos. Uma vez que os pontos médios das "âncoras" das caixas de âncora se sobrepõem a todas as unidades no mapa de características "fmap", as posições espaciais relativas dos pontos médios das "âncoras" em qualquer imagem devem ter uma distribuição uniforme. Especificamente, quando a largura e a altura do mapa de feições são definidas para `fmap_w` e` fmap_h` respectivamente, a função irá conduzir uma amostragem uniforme para linhas `fmap_h` e colunas de pixels` fmap_w` e usá-los como pontos médios para gerar caixas de âncora com tamanho `s` (assumimos que o comprimento da lista `s` é 1) e diferentes proporções (`ratios`).

```{.python .input}
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # The values from the first two dimensions will not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # The values from the first two dimensions will not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

We will first focus on the detection of small objects. In order to make it easier to distinguish upon display, the anchor boxes with different midpoints here do not overlap. We assume that the size of the anchor boxes is 0.15 and the height and width of the feature map are 4. We can see that the midpoints of anchor boxes from the 4 rows and 4 columns on the image are uniformly distributed.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

We are going to reduce the height and width of the feature map by half and use a larger anchor box to detect larger objects. When the size is set to 0.4, overlaps will occur between regions of some anchor boxes.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

Finally, we are going to reduce the height and width of the feature map by half and increase the anchor box size to 0.8. Now the midpoint of the anchor box is the center of the image.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

Since we have generated anchor boxes of different sizes on multiple scales, we will use them to detect objects of various sizes at different scales. Now we are going to introduce a method based on convolutional neural networks (CNNs).

At a certain scale, suppose we generate $h \times w$ sets of anchor boxes with different midpoints based on $c_i$ feature maps with the shape $h \times w$ and the number of anchor boxes in each set is $a$. For example, for the first scale of the experiment, we generate 16 sets of anchor boxes with different midpoints based on 10 (number of channels) feature maps with a shape of $4 \times 4$, and each set contains 3 anchor boxes.
Next, each anchor box is labeled with a category and offset based on the classification and position of the ground-truth bounding box. At the current scale, the object detection model needs to predict the category and offset of $h \times w$ sets of anchor boxes with different midpoints based on the input image.

We assume that the $c_i$ feature maps are the intermediate output of the CNN
based on the input image. Since each feature map has $h \times w$ different
spatial positions, the same position will have $c_i$ units.  According to the
definition of receptive field in the
:numref:`sec_conv_layer`, the $c_i$ units of the feature map at the same spatial position have
the same receptive field on the input image. Thus, they represent the
information of the input image in this same receptive field.  Therefore, we can
transform the $c_i$ units of the feature map at the same spatial position into
the categories and offsets of the $a$ anchor boxes generated using that position
as a midpoint.  It is not hard to see that, in essence, we use the information
of the input image in a certain receptive field to predict the category and
offset of the anchor boxes close to the field on the input image.

When the feature maps of different layers have receptive fields of different sizes on the input image, they are used to detect objects of different sizes. For example, we can design a network to have a wider receptive field for each unit in the feature map that is closer to the output layer, to detect objects with larger sizes in the input image.

We will implement a multiscale object detection model in the following section.


## Summary

* We can generate anchor boxes with different numbers and sizes on multiple scales to detect objects of different sizes on multiple scales.
* The shape of the feature map can be used to determine the midpoint of the anchor boxes that uniformly sample any image.
* We use the information for the input image from a certain receptive field to predict the category and offset of the anchor boxes close to that field on the image.


## Exercises

1. Given an input image, assume $1 \times c_i \times h \times w$ to be the shape of the feature map while $c_i, h, w$ are the number, height, and width of the feature map. What methods can you think of to convert this variable into the anchor box's category and offset? What is the shape of the output?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1607)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTM5NDE4MTU4MSwyMDU2NDA5NTcwLC0xNz
gzMzIwMzBdfQ==
-->
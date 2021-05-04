# Segmentação Semântica e o *Dataset*
:label:`sec_semantic_segmentation`

Em nossa discussão sobre os problemas de detecção de objetos nas seções anteriores, usamos apenas caixas delimitadoras retangulares para rotular e prever objetos em imagens. Nesta seção, veremos a segmentação semântica, que tenta segmentar imagens em regiões com diferentes categorias semânticas. Essas regiões semânticas rotulam e prevêem objetos no nível do pixel. :numref:`fig_segmentation` mostra uma imagem semanticamente segmentada, com áreas marcadas como "cachorro" , "gato" e "fundo". Como você pode ver, em comparação com a detecção de objetos, a segmentação semântica rotula áreas com bordas em nível de pixel, para uma precisão significativamente maior.

![Imagem segmentada semanticamente, com áreas rotuladas "cachorro", "gato" e "plano de fundo". ](../img/segmentation.svg)
:label:`fig_segmentation`


## Segmentação de Imagem e Segmentação de Instância


No campo da visão computacional, existem dois métodos importantes relacionados à segmentação semântica: segmentação de imagens e segmentação de instâncias. Aqui, vamos distinguir esses conceitos da segmentação semântica da seguinte forma:

* A segmentação da imagem divide uma imagem em várias regiões constituintes. Este método geralmente usa as correlações entre pixels em uma imagem. Durante o treinamento, os rótulos não são necessários para pixels de imagem. No entanto, durante a previsão, esse método não pode garantir que as regiões segmentadas tenham a semântica que desejamos. Se inserirmos a imagem em 9.10, a segmentação da imagem pode dividir o cão em duas regiões, uma cobrindo a boca do cão e os olhos onde o preto é a cor proeminente e a outra cobrindo o resto do cão onde o amarelo é a cor proeminente.
* A segmentação da instância também é chamada de detecção e segmentação simultâneas. Este método tenta identificar as regiões de nível de pixel de cada instância de objeto em uma imagem. Em contraste com a segmentação semântica, a segmentação de instância não apenas distingue a semântica, mas também diferentes instâncias de objeto. Se uma imagem contém dois cães, a segmentação de instância distingue quais pixels pertencem a cada cachorro.


## O Conjunto de Dados de Segmentação Semântica Pascal VOC2012

No campo de segmentação semântica, um conjunto de dados importante é [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). Para entender melhor este conjunto de dados, devemos primeiro importar o pacote ou módulo necessário para o experimento.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
```

O site original pode ser instável, portanto, baixamos os dados de um site espelho.
O arquivo tem cerca de 2 GB, por isso levará algum tempo para fazer o download.
Depois de descompactar o arquivo, o conjunto de dados está localizado no caminho `../data/VOCdevkit/VOC2012`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

Vá para `../data/VOCdevkit/VOC2012` para ver as diferentes partes do conjunto de dados.
O caminho `ImageSets/Segmentation` contém arquivos de texto que especificam os exemplos de treinamento e teste. Os caminhos `JPEGImages` e` SegmentationClass` contêm as imagens de entrada de exemplo e rótulos, respectivamente. Essas etiquetas também estão em formato de imagem, com as mesmas dimensões das imagens de entrada às quais correspondem. Nos rótulos, os pixels com a mesma cor pertencem à mesma categoria semântica. A função `read_voc_images` definida abaixo lê todas as imagens de entrada e rótulos para a memória.

```{.python .input}
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(image.imread(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(image.imread(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

```{.python .input}
#@tab pytorch
#@save
def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images."""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

train_features, train_labels = read_voc_images(voc_dir, True)
```

Desenhamos as primeiras cinco imagens de entrada e seus rótulos. Nas imagens do rótulo, o branco representa as bordas e o preto representa o fundo. Outras cores correspondem a diferentes categorias.

```{.python .input}
n = 5
imgs = train_features[0:n] + train_labels[0:n]
d2l.show_images(imgs, 2, n);
```

```{.python .input}
#@tab pytorch
n = 5
imgs = train_features[0:n] + train_labels[0:n]
imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs, 2, n);
```

A seguir, listamos cada valor de cor RGB nos rótulos e as categorias que eles rotulam.

```{.python .input}
#@tab all
#@save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

#@save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
```

Depois de definir as duas constantes acima, podemos encontrar facilmente o índice de categoria para cada pixel nos rótulos.

```{.python .input}
#@save
def build_colormap2label():
    """Build an RGB color to label mapping for segmentation."""
    colormap2label = np.zeros(256 ** 3)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0]*256 + colormap[1])*256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map an RGB color to a label."""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

```{.python .input}
#@tab pytorch
#@save
def build_colormap2label():
    """Build an RGB color to label mapping for segmentation."""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0]*256 + colormap[1])*256 + colormap[2]] = i
    return colormap2label

#@save
def voc_label_indices(colormap, colormap2label):
    """Map an RGB color to a label."""
    colormap = colormap.permute(1,2,0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]
```

Por exemplo, na primeira imagem de exemplo, o índice de categoria para a parte frontal do avião é 1 e o índice para o fundo é 0.

```{.python .input}
#@tab all
y = voc_label_indices(train_labels[0], build_colormap2label())
y[105:115, 130:140], VOC_CLASSES[1]
```

### Pré-processamento de Dados

Nos capítulos anteriores, dimensionamos as imagens para que se ajustassem à forma de entrada do modelo. Na segmentação semântica, esse método exigiria que mapeamos novamente as categorias de pixels previstas de volta à imagem de entrada do tamanho original. Seria muito difícil fazer isso com precisão, especialmente em regiões segmentadas com semânticas diferentes. Para evitar esse problema, recortamos as imagens para definir as dimensões e não as dimensionamos. Especificamente, usamos o método de corte aleatório usado no aumento da imagem para cortar a mesma região das imagens de entrada e seus rótulos.

```{.python .input}
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop for both feature and label images."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label
```

```{.python .input}
#@tab pytorch
#@save
def voc_rand_crop(feature, label, height, width):
    """Randomly crop for both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(feature,
                                                        (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label
```

```{.python .input}
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

```{.python .input}
#@tab pytorch
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)

imgs = [img.permute(1,2,0) for img in imgs]
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

### Classes de *Datasets* para Segmentação Semântica Personalizada

Usamos a classe `Dataset` herdada fornecida pelo Gluon para personalizar a classe de conjunto de dados de segmentação semântica `VOCSegDataset`. Implementando a função `__getitem__`, podemos acessar arbitrariamente a imagem de entrada com o índice `idx` e os índices de categoria para cada um de seus pixels do conjunto de dados. Como algumas imagens no conjunto de dados podem ser menores do que as dimensões de saída especificadas para corte aleatório, devemos remover esses exemplos usando uma função `filter` personalizada. Além disso, definimos a função `normalize_image` para normalizar cada um dos três canais RGB das imagens de entrada.

```{.python .input}
#@save
class VOCSegDataset(gluon.data.Dataset):
    """A customized dataset to load VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = build_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype('float32') / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature.transpose(2, 0, 1),
                voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load VOC dataset."""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = build_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float())

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)
```

### Lendo o Dataset

Usando a classe `VOCSegDataset` personalizada, criamos o conjunto de treinamento e as instâncias do conjunto de teste. Assumimos que a operação de corte aleatório produz imagens no formato $320\times 480$. Abaixo, podemos ver o número de exemplos retidos nos conjuntos de treinamento e teste.

```{.python .input}
#@tab all
crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)
```

Definimos o tamanho do lote como 64 e definimos os iteradores para os conjuntos de treinamento e teste. Imprimimos a forma do primeiro minibatch. Em contraste com a classificação de imagens e o reconhecimento de objetos, os rótulos aqui são matrizes tridimensionais.

```{.python .input}
batch_size = 64
train_iter = gluon.data.DataLoader(voc_train, batch_size, shuffle=True,
                                   last_batch='discard',
                                   num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

```{.python .input}
#@tab pytorch
batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break
```

### Juntando Tudo

Finalmente, definimos uma função `load_data_voc` que baixa e carrega este *dataset*, e então retorna os iteradores de dados.

```{.python .input}
#@save
def load_data_voc(batch_size, crop_size):
    """Download and load the VOC2012 semantic dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = gluon.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, last_batch='discard', num_workers=num_workers)
    test_iter = gluon.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        last_batch='discard', num_workers=num_workers)
    return train_iter, test_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_voc(batch_size, crop_size):
    """Download and load the VOC2012 semantic dataset."""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter
```

## Resumo

* A segmentação semântica analisa como as imagens podem ser segmentadas em regiões com diferentes categorias semânticas.
* No campo de segmentação semântica, um conjunto de dados importante é Pascal VOC2012.
* Como as imagens e rótulos de entrada na segmentação semântica têm uma correspondência um a um no nível do pixel, nós os cortamos aleatoriamente em um tamanho fixo, em vez de dimensioná-los.

## Exercícios

1. Lembre-se do conteúdo que vimos em :numref:`sec_image_augmentation`. Qual dos métodos de aumento de imagem usados na classificação de imagens seria difícil de usar na segmentação semântica?

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/375)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1480)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjc2NDM5NjAsLTIwMDUzNDc0MzYsLTEyOT
A0NDU2NjgsMjk0NTE1OTE4LC0yMDE5NTg0MTQ3XX0=
-->
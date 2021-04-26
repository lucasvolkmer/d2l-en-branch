# Aumento de Imagem
:label:`sec_image_augmentation`



Mencionamos que conjuntos de dados em grande escala são pré-requisitos para a aplicação bem-sucedida de redes neurais profundas em :numref:`sec_alexnet`. A tecnologia de aumento de imagem expande a escala dos conjuntos de dados de treinamento, fazendo uma série de alterações aleatórias nas imagens de treinamento para produzir exemplos de treinamento semelhantes, mas diferentes. Outra maneira de explicar o aumento de imagem é
que exemplos de treinamento que mudam aleatoriamente podem reduzir a dependência de um modelo em certas propriedades, melhorando assim sua capacidade de generalização. Por exemplo, podemos recortar as imagens de diferentes maneiras, para que os objetos de interesse apareçam em diferentes posições, reduzindo a dependência do modelo da posição onde os objetos aparecem. Também podemos ajustar o brilho, a cor e outros fatores para reduzir a sensibilidade do modelo à cor. Pode-se dizer que a tecnologia de aumento de imagem contribuiu muito para o sucesso do AlexNet. Nesta seção, discutiremos essa tecnologia, que é amplamente usada na visão computacional.

Primeiro, importe os pacotes ou módulos necessários para o experimento nesta seção.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
```

## Método Comum de Aumento de Imagem

Neste experimento, usaremos uma imagem com um formato de $400\times 500$ como exemplo.

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

A maioria dos métodos de aumento de imagem tem um certo grau de aleatoriedade. Para facilitar a observação do efeito do aumento da imagem, definimos a seguir a função auxiliar `aplicar`. Esta função executa o método de aumento de imagem `aug` várias vezes na imagem de entrada` img` e mostra todos os resultados.

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### Invertendo e Recortando

Virar a imagem para a esquerda e para a direita geralmente não altera a categoria do objeto. Este é um dos métodos mais antigos e mais amplamente usados de aumento de imagem. Em seguida, usamos o módulo `transforms` para criar a instância `RandomFlipLeftRight`, que apresenta uma chance de 50% de que a imagem seja virada para a esquerda e para a direita.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

Virar para cima e para baixo não é tão comumente usado como girar para a esquerda e para a direita. No entanto, pelo menos para esta imagem de exemplo, virar para cima e para baixo não impede o reconhecimento. Em seguida, criamos uma instância `RandomFlipTopBottom` para uma chance de 50% de virar a imagem para cima e para baixo.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

In the example image we used, the cat is in the middle of the image, but this
may not be the case for all images. In
:numref:`sec_pooling`,
we explained that the pooling layer can reduce the sensitivity of the
convolutional layer to the target location. In addition, we can make objects
appear at different positions in the image in different proportions by randomly
cropping the image. This can also reduce the sensitivity of the model to the
target position.

In the following code, we randomly crop a region with an area of 10% to 100% of the original area, and the ratio of width to height of the region is randomly selected from between 0.5 and 2. Then, the width and height of the region are both scaled to 200 pixels. Unless otherwise stated, the random number between $a$ and $b$ in this section refers to a continuous value obtained by uniform sampling in the interval $[a, b]$.

```{.python .input}
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### Changing the Color

Another augmentation method is changing colors. We can change four aspects of the image color: brightness, contrast, saturation, and hue. In the example below, we randomly change the brightness of the image to a value between 50% ($1-0.5$) and 150% ($1+0.5$) of the original image.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

Similarly, we can randomly change the hue of the image.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

We can also create a `RandomColorJitter` instance and set how to randomly change the `brightness`, `contrast`, `saturation`, and `hue` of the image at the same time.

```{.python .input}
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### Overlying Multiple Image Augmentation Methods

In practice, we will overlay multiple image augmentation methods. We can overlay the different image augmentation methods defined above and apply them to each image by using a `Compose` instance.

```{.python .input}
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## Using an Image Augmentation Training Model

Next, we will look at how to apply image augmentation in actual training. Here, we use the CIFAR-10 dataset, instead of the Fashion-MNIST dataset we have been using. This is because the position and size of the objects in the Fashion-MNIST dataset have been normalized, and the differences in color and size of the objects in CIFAR-10 dataset are more significant. The first 32 training images in the CIFAR-10 dataset are shown below.

```{.python .input}
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[0:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

In order to obtain definitive results during prediction, we usually only apply image augmentation to the training example, and do not use image augmentation with random operations during prediction. Here, we only use the simplest random left-right flipping method. In addition, we use a `ToTensor` instance to convert minibatch images into the format required by MXNet, i.e., 32-bit floating point numbers with the shape of (batch size, number of channels, height, width) and value range between 0 and 1.

```{.python .input}
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

Next, we define an auxiliary function to make it easier to read the image and
apply image augmentation. The `transform_first` function provided by Gluon's
dataset applies image augmentation to the first element of each training
example (image and label), i.e., the element at the top of the image. For
detailed descriptions of `DataLoader`, refer to :numref:`sec_fashion_mnist`.

```{.python .input}
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### Using a Multi-GPU Training Model

We train the ResNet-18 model described in :numref:`sec_resnet` on the
CIFAR-10 dataset. We will also apply the methods described in
:numref:`sec_multi_gpu_concise` and use a multi-GPU training model.

Next, we define the training function to train and evaluate the model using multiple GPUs.

```{.python .input}
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The True flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        # Required for BERT Fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Store training_loss, training_accuracy, num_examples, num_features
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Store training_loss, training_accuracy, num_examples, num_features
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

Now, we can define the `train_with_data_aug` function to use image augmentation to train the model. This function obtains all available GPUs and uses Adam as the optimization algorithm for training. It then applies image augmentation to the training dataset, and finally calls the `train_ch13` function just defined to train and evaluate the model.

```{.python .input}
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

Now we train the model using image augmentation of random flipping left and right.

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## Summary

* Image augmentation generates random images based on existing training data to cope with overfitting.
* In order to obtain definitive results during prediction, we usually only apply image augmentation to the training example, and do not use image augmentation with random operations during prediction.
* We can obtain classes related to image augmentation from Gluon's `transforms` module.

## Exercises

1. Train the model without using image augmentation: `train_with_data_aug(no_aug, no_aug)`. Compare training and testing accuracy when using and not using image augmentation. Can this comparative experiment support the argument that image augmentation can mitigate overfitting? Why?
1. Add different image augmentation methods in model training based on the CIFAR-10 dataset. Observe the implementation results.
1. With reference to the MXNet documentation, what other image augmentation methods are provided in Gluon's `transforms` module?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1404)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk1MTc1MDIyMiwxNzAzMDcyOTc5XX0=
-->
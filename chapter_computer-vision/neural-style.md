# Transferência de Estilo Neural

Se você usa aplicativos de compartilhamento social ou é um fotógrafo amador, está familiarizado com os filtros. Os filtros podem alterar os estilos de cor das fotos para tornar o fundo mais nítido ou o rosto das pessoas mais branco. No entanto, um filtro geralmente só pode alterar um aspecto de uma foto. Para criar a foto ideal, muitas vezes você precisa tentar muitas combinações de filtros diferentes. Esse processo é tão complexo quanto ajustar os hiperparâmetros de um modelo.

Nesta seção, discutiremos como podemos usar redes neurais de convolução (CNNs) para aplicar automaticamente o estilo de uma imagem a outra imagem, uma operação conhecida como transferência de estilo :cite:`Gatys.Ecker.Bethge.2016`. Aqui, precisamos de duas imagens de entrada, uma imagem de conteúdo e uma imagem de estilo. Usamos uma rede neural para alterar a imagem do conteúdo de modo que seu estilo espelhe o da imagem do estilo. Em :numref:`fig_style_transfer`, a imagem do conteúdo é uma foto de paisagem que o autor tirou na Mount Rainier National Part perto de Seattle. A imagem do estilo é uma pintura a óleo de carvalhos no outono. A imagem composta de saída retém as formas gerais dos objetos na imagem de conteúdo, mas aplica a pincelada de pintura a óleo da imagem de estilo e torna a cor geral mais vívida.

![Imagens de entrada de conteúdo e estilo e imagem composta produzida por transferência de estilo. ](../img/style-transfer.svg)
:label:`fig_style_transfer`

## Técnica

O modelo de transferência de estilo baseado em CNN é mostrado em :numref:`fig_style_transfer_model`.
Primeiro, inicializamos a imagem composta. Por exemplo, podemos inicializá-la como a imagem do conteúdo. Esta imagem composta é a única variável que precisa ser atualizada no processo de transferência de estilo, ou seja, o parâmetro do modelo a ser atualizado na transferência de estilo. Em seguida, selecionamos uma CNN pré-treinada para extrair recursos de imagem. Esses parâmetros do modelo não precisam ser atualizados durante o treinamento. O CNN profundo usa várias camadas neurais que extraem sucessivamente recursos de imagem. Podemos selecionar a saída de certas camadas para usar como recursos de conteúdo ou recursos de estilo. Se usarmos a estrutura em :numref:`fig_style_transfer_model`, a rede neural pré-treinada contém três camadas convolucionais. A segunda camada produz os recursos de conteúdo da imagem, enquanto as saídas da primeira e terceira camadas são usadas como recursos de estilo. Em seguida, usamos a propagação para a frente (na direção das linhas sólidas) para calcular a função de perda de transferência de estilo e a propagação para trás (na direção das linhas pontilhadas) para atualizar o parâmetro do modelo, atualizando constantemente a imagem composta. As funções de perda usadas na transferência de estilo geralmente têm três partes: 1. A perda de conteúdo é usada para fazer a imagem composta se aproximar da imagem de conteúdo no que diz respeito aos recursos de conteúdo. 2. A perda de estilo é usada para fazer com que a imagem composta se aproxime da imagem de estilo em termos de recursos de estilo. 3. A perda total de variação ajuda a reduzir o ruído na imagem composta. Finalmente, depois de terminar de treinar o modelo, produzimos os parâmetros do modelo de transferência de estilo para obter a imagem composta final.

![Processo de transferência de estilo baseado em CNN. As linhas sólidas mostram a direção da propagação para a frente e as linhas pontilhadas mostram a propagação para trás. ](../img/neural-style.svg)
:label:`fig_style_transfer_model`


A seguir, faremos um experimento para nos ajudar a entender melhor os detalhes técnicos da transferência de estilo.

## Lendo o Conteúdo e as Imagens de Estilo

Primeiro, lemos o conteúdo e as imagens de estilo. Ao imprimir os eixos de coordenadas da imagem, podemos ver que eles têm dimensões diferentes.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

d2l.set_figsize()
content_img = image.imread('../img/rainier.jpg')
d2l.plt.imshow(content_img.asnumpy());
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn

d2l.set_figsize()
content_img = d2l.Image.open('../img/rainier.jpg')
d2l.plt.imshow(content_img);
```

```{.python .input}
style_img = image.imread('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img.asnumpy());
```

```{.python .input}
#@tab pytorch
style_img = d2l.Image.open('../img/autumn-oak.jpg')
d2l.plt.imshow(style_img);
```

## Pré-processamento e Pós-processamento

A seguir, definimos as funções de pré-processamento e pós-processamento de imagens. A função `pré-processamento` normaliza cada um dos três canais RGB das imagens de entrada e transforma os resultados em um formato que pode ser inserido na CNN. A função `postprocess` restaura os valores de pixel na imagem de saída para seus valores originais antes da normalização. Como a função de impressão de imagem requer que cada pixel tenha um valor de ponto flutuante de 0 a 1, usamos a função `clip` para substituir valores menores que 0 ou maiores que 1 por 0 ou 1, respectivamente.

```{.python .input}
rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)

def postprocess(img):
    img = img[0].as_in_ctx(rgb_std.ctx)
    return (img.transpose(1, 2, 0) * rgb_std + rgb_mean).clip(0, 1)
```

```{.python .input}
#@tab pytorch
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))
```

## Extraindo *Features*

Usamos o modelo VGG-19 pré-treinado no conjunto de dados ImageNet para extrair características da imagem [1].

```{.python .input}
pretrained_net = gluon.model_zoo.vision.vgg19(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.vgg19(pretrained=True)
```

Para extrair o conteúdo da imagem e os recursos de estilo, podemos selecionar as saídas de certas camadas na rede VGG. Em geral, quanto mais próxima uma saída estiver da camada de entrada, mais fácil será extrair informações detalhadas da imagem. Quanto mais longe uma saída estiver, mais fácil será extrair informações globais. Para evitar que a imagem composta retenha muitos detalhes da imagem de conteúdo, selecionamos uma camada de rede VGG próxima à camada de saída para produzir os recursos de conteúdo da imagem. Essa camada é chamada de camada de conteúdo. Também selecionamos as saídas de diferentes camadas da rede VGG para combinar os estilos local e global. Elas são chamadas de camadas de estilo. Como mencionamos em :numref:`sec_vgg`, as redes VGG têm cinco blocos convolucionais. Neste experimento, selecionamos a última camada convolucional do quarto bloco convolucional como a camada de conteúdo e a primeira camada de cada bloco como camadas de estilo. Podemos obter os índices para essas camadas imprimindo a instância `pretrained_net`.

```{.python .input}
#@tab all
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
```

Durante a extração de *features*, só precisamos usar todas as camadas VGG da camada de entrada até a camada de conteúdo ou estilo mais próxima da camada de saída. Abaixo, construímos uma nova rede, `net`, que retém apenas as camadas da rede VGG que precisamos usar. Em seguida, usamos `net` para extrair *features*.

```{.python .input}
net = nn.Sequential()
for i in range(max(content_layers + style_layers) + 1):
    net.add(pretrained_net.features[i])
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])
```

Dada a entrada `X`, se simplesmente chamarmos a computação direta de `net (X)`, só podemos obter a saída da última camada. Como também precisamos das saídas das camadas intermediárias, precisamos realizar computação camada por camada e reter o conteúdo e as saídas da camada de estilo.

```{.python .input}
#@tab all
def extract_features(X, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles
```

Next, we define two functions: The `get_contents` function obtains the content features extracted from the content image, while the `get_styles` function obtains the style features extracted from the style image. Because we do not need to change the parameters of the pre-trained VGG model during training, we can extract the content features from the content image and style features from the style image before the start of training. As the composite image is the model parameter that must be updated during style transfer, we can only call the `extract_features` function during training to extract the content and style features of the composite image.

```{.python .input}
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).copyto(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).copyto(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

```{.python .input}
#@tab pytorch
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y
```

## Defining the Loss Function

Next, we will look at the loss function used for style transfer. The loss function includes the content loss, style loss, and total variation loss.

### Content Loss

Similar to the loss function used in linear regression, content loss uses a square error function to measure the difference in content features between the composite image and content image. The two inputs of the square error function are both content layer outputs obtained from the `extract_features` function.

```{.python .input}
def content_loss(Y_hat, Y):
    return np.square(Y_hat - Y).mean()
```

```{.python .input}
#@tab pytorch
def content_loss(Y_hat, Y):
    # we 'detach' the target content from the tree used
    # to dynamically compute the gradient: this is a stated value,
    # not a variable. Otherwise the loss will throw an error.
    return torch.square(Y_hat - Y.detach()).mean()
```

### Style Loss

Style loss, similar to content loss, uses a square error function to measure the difference in style between the composite image and style image. To express the styles output by the style layers, we first use the `extract_features` function to compute the style layer output. Assuming that the output has 1 example, $c$ channels, and a height and width of $h$ and $w$, we can transform the output into the matrix $\mathbf{X}$, which has $c$ rows and $h \cdot w$ columns. You can think of matrix $\mathbf{X}$ as the combination of the $c$ vectors $\mathbf{x}_1, \ldots, \mathbf{x}_c$, which have a length of $hw$. Here, the vector $\mathbf{x}_i$ represents the style feature of channel $i$. In the Gram matrix of these vectors $\mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{c \times c}$, element $x_{ij}$ in row $i$ column $j$ is the inner product of vectors $\mathbf{x}_i$ and $\mathbf{x}_j$. It represents the correlation of the style features of channels $i$ and $j$. We use this type of Gram matrix to represent the style output by the style layers. You must note that, when the $h \cdot w$ value is large, this often leads to large values in the Gram matrix. In addition, the height and width of the Gram matrix are both the number of channels $c$. To ensure that the style loss is not affected by the size of these values, we define the `gram` function below to divide the Gram matrix by the number of its elements, i.e., $c \cdot h \cdot w$.

```{.python .input}
#@tab all
def gram(X):
    num_channels, n = X.shape[1], d2l.size(X) // X.shape[1]
    X = d2l.reshape(X, (num_channels, n))
    return d2l.matmul(X, X.T) / (num_channels * n)
```

Naturally, the two Gram matrix inputs of the square error function for style loss are taken from the composite image and style image style layer outputs. Here, we assume that the Gram matrix of the style image, `gram_Y`, has been computed in advance.

```{.python .input}
def style_loss(Y_hat, gram_Y):
    return np.square(gram(Y_hat) - gram_Y).mean()
```

```{.python .input}
#@tab pytorch
def style_loss(Y_hat, gram_Y):
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()
```

### Total Variance Loss

Sometimes, the composite images we learn have a lot of high-frequency noise, particularly bright or dark pixels. One common noise reduction method is total variation denoising. We assume that $x_{i, j}$ represents the pixel value at the coordinate $(i, j)$, so the total variance loss is:

$$\sum_{i, j} \left|x_{i, j} - x_{i+1, j}\right| + \left|x_{i, j} - x_{i, j+1}\right|.$$

We try to make the values of neighboring pixels as similar as possible.

```{.python .input}
#@tab all
def tv_loss(Y_hat):
    return 0.5 * (d2l.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  d2l.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())
```

### Loss Function

The loss function for style transfer is the weighted sum of the content loss, style loss, and total variance loss. By adjusting these weight hyperparameters, we can balance the retained content, transferred style, and noise reduction in the composite image according to their relative importance.

```{.python .input}
#@tab all
content_weight, style_weight, tv_weight = 1, 1e3, 10

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # Calculate the content, style, and total variance losses respectively
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # Add up all the losses
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l
```

## Creating and Initializing the Composite Image

In style transfer, the composite image is the only variable that needs to be updated. Therefore, we can define a simple model, `GeneratedImage`, and treat the composite image as a model parameter. In the model, forward computation only returns the model parameter.

```{.python .input}
class GeneratedImage(nn.Block):
    def __init__(self, img_shape, **kwargs):
        super(GeneratedImage, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=img_shape)

    def forward(self):
        return self.weight.data()
```

```{.python .input}
#@tab pytorch
class GeneratedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(GeneratedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight
```

Next, we define the `get_inits` function. This function creates a composite image model instance and initializes it to the image `X`. The Gram matrix for the various style layers of the style image, `styles_Y_gram`, is computed prior to training.

```{.python .input}
def get_inits(X, device, lr, styles_Y):
    gen_img = GeneratedImage(X.shape)
    gen_img.initialize(init.Constant(X), ctx=device, force_reinit=True)
    trainer = gluon.Trainer(gen_img.collect_params(), 'adam',
                            {'learning_rate': lr})
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

```{.python .input}
#@tab pytorch
def get_inits(X, device, lr, styles_Y):
    gen_img = GeneratedImage(X.shape).to(device)
    gen_img.weight.data.copy_(X.data)
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer
```

## Training

During model training, we constantly extract the content and style features of
the composite image and calculate the loss function. Recall our discussion of
how synchronization functions force the front end to wait for computation
results in :numref:`sec_async`. Because we only call the `asnumpy` synchronization function every 10
epochs, the process may occupy a great deal of memory. Therefore, we call the
`waitall` synchronization function during every epoch.

```{.python .input}
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        with autograd.record():
            contents_Y_hat, styles_Y_hat = extract_features(
                X, content_layers, style_layers)
            contents_l, styles_l, tv_l, l = compute_loss(
                X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step(1)
        npx.waitall()
        if (epoch + 1) % lr_decay_epoch == 0:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X).asnumpy())
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

```{.python .input}
#@tab pytorch
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    for epoch in range(num_epochs):
        trainer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        trainer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X
```

Next, we start to train the model. First, we set the height and width of the content and style images to 150 by 225 pixels. We use the content image to initialize the composite image.

```{.python .input}
device, image_shape = d2l.try_gpu(), (225, 150)
net.collect_params().reset_ctx(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.01, 500, 200)
```

```{.python .input}
#@tab pytorch
device, image_shape = d2l.try_gpu(), (150, 225) # PIL Image (h, w)
net = net.to(device)
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.01, 500, 200)
```

As you can see, the composite image retains the scenery and objects of the content image, while introducing the color of the style image. Because the image is relatively small, the details are a bit fuzzy.

To obtain a clearer composite image, we train the model using a larger image size: $900 \times 600$. We increase the height and width of the image used before by a factor of four and initialize a larger composite image.

```{.python .input}
image_shape = (900, 600)
_, content_Y = get_contents(image_shape, device)
_, style_Y = get_styles(image_shape, device)
X = preprocess(postprocess(output) * 255, image_shape)
output = train(X, content_Y, style_Y, device, 0.01, 300, 100)
d2l.plt.imsave('../img/neural-style.jpg', postprocess(output).asnumpy())
```

```{.python .input}
#@tab pytorch
image_shape = (600, 900) # PIL Image (h, w)
_, content_Y = get_contents(image_shape, device)
_, style_Y = get_styles(image_shape, device)
X = preprocess(postprocess(output), image_shape).to(device)
output = train(X, content_Y, style_Y, device, 0.01, 300, 100)
d2l.plt.imsave('../img/neural-style.jpg', postprocess(output))
```

As you can see, each epoch takes more time due to the larger image size. As shown in :numref:`fig_style_transfer_large`, the composite image produced retains more detail due to its larger size. The composite image not only has large blocks of color like the style image, but these blocks even have the subtle texture of brush strokes.

![$900 \times 600$ composite image. ](../img/neural-style.jpg)
:width:`500px`
:label:`fig_style_transfer_large`

## Summary

* The loss functions used in style transfer generally have three parts: 1. Content loss is used to make the composite image approximate the content image as regards content features. 2. Style loss is used to make the composite image approximate the style image in terms of style features. 3. Total variation loss helps reduce the noise in the composite image.
* We can use a pre-trained CNN to extract image features and minimize the loss function to continuously update the composite image.
* We use a Gram matrix to represent the style output by the style layers.


## Exercises

1. How does the output change when you select different content and style layers?
1. Adjust the weight hyperparameters in the loss function. Does the output retain more content or have less noise?
1. Use different content and style images. Can you create more interesting composite images?
1. Can we apply style transfer for text? Hint: you may refer to the survey paper by Hu et al. :cite:`Hu.Lee.Aggarwal.2020`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/378)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1476)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU5MzYyNjM0OSwtODQxNDk1OTkyLC05Mz
M5MjU2MiwtMjY5MDY0ODM1LDE4OTE0MTM1MTBdfQ==
-->
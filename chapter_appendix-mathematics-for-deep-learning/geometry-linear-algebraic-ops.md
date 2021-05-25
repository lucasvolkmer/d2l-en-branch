# Operações de Geometria e Álgebra Linear
:label:`sec_geometry-linear-algebraic-ops`

Em :numref:`sec_linear-algebra`, encontramos os fundamentos da álgebra linear e vimos como ela poderia ser usada para expressar operações comuns para transformar nossos dados.
A álgebra linear é um dos principais pilares matemáticos subjacentes a grande parte do trabalho que fazemos no *deep learning* e no *machine learning* de forma mais ampla.
Embora :numref:`sec_linear-algebra` contenha maquinário suficiente para comunicar a mecânica dos modelos modernos de aprendizado profundo, há muito mais sobre o assunto.
Nesta seção, iremos mais fundo, destacando algumas interpretações geométricas de operações de álgebra linear e introduzindo alguns conceitos fundamentais, incluindo de autovalores e autovetores.

## Geometria Vetorial
Primeiro, precisamos discutir as duas interpretações geométricas comuns de vetores,
como pontos ou direções no espaço.
Fundamentalmente, um vetor é uma lista de números, como a lista Python abaixo.

```{.python .input}
#@tab all
v = [1, 7, 0, 1]
```

Os matemáticos geralmente escrevem isso como um vetor *coluna* ou *linha*, ou seja, como

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$

ou

$$
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$


Muitas vezes têm interpretações diferentes,
onde os exemplos de dados são vetores de coluna
e os pesos usados para formar somas ponderadas são vetores de linha.
No entanto, pode ser benéfico ser flexível.
Como descrevemos em :numref:`sec_linear-algebra`,
embora a orientação padrão de um único vetor seja um vetor coluna,
para qualquer matriz que representa um conjunto de dados tabular,
tratando cada exemplo de dados como um vetor de linha
na matriz
é mais convencional.

Dado um vetor, a primeira interpretação
que devemos dar é como um ponto no espaço.
Em duas ou três dimensões, podemos visualizar esses pontos
usando os componentes dos vetores para definir
a localização dos pontos no espaço comparada
a uma referência fixa chamada *origem*. Isso pode ser visto em :numref:`fig_grid`.

![Uma ilustração da visualização de vetores como pontos no plano. O primeiro componente do vetor fornece a coordenada $x$, o segundo componente fornece a coordenada $y$. As dimensões superiores são análogas, embora muito mais difíceis de visualizar.](../img/grid-points.svg)
:label:`fig_grid`


Esse ponto de vista geométrico nos permite considerar o problema em um nível mais abstrato.
Não mais confrontado com algum problema aparentemente intransponível
como classificar fotos como gatos ou cachorros,
podemos começar a considerar as tarefas abstratamente
como coleções de pontos no espaço e retratando a tarefa
como descobrir como separar dois grupos distintos de pontos.

Paralelamente, existe um segundo ponto de vista
que as pessoas costumam tomar de vetores: como direções no espaço.
Não podemos apenas pensar no vetor $\mathbf{v} = [3,2]^\top$
como a localização $3$ unidades à direita e $2$ unidades acima da origem,
também podemos pensar nisso como a própria direção
para mover $3$ passos para a direita e $2$ para cima.
Desta forma, consideramos todos os vetores da figura :numref:`fig_arrow` iguais.

![Qualquer vetor pode ser visualizado como uma seta no plano. Nesse caso, todo vetor desenhado é uma representação do vetor $(3,2)^\top$.](../img/par-vec.svg)
:label:`fig_arrow`

Um dos benefícios dessa mudança é que
podemos dar sentido visual ao ato de adição de vetores.
Em particular, seguimos as instruções dadas por um vetor,
e então siga as instruções dadas pelo outro, como pode ser visto em :numref:`fig_add-vec`.

![Podemos visualizar a adição de vetores seguindo primeiro um vetor e depois outro.](../img/vec-add.svg)
:label:`fig_add-vec`

A subtração de vetores tem uma interpretação semelhante.
Considerando a identidade que $\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$,
vemos que o vetor $\mathbf{u}-\mathbf{v}$ é a direção
que nos leva do ponto $\mathbf{v}$ ao ponto $\mathbf{u}$.


## Produto Escalar e Ângulos
Como vimos em :numref:`sec_linear-algebra`,
se tomarmos dois vetores de coluna $\mathbf{u}$ and $\mathbf{v}$,
podemos formar seu produto escalar computando:

$$\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_dot_def`

Porque :eqref:`eq_dot_def` é simétrico, iremos espelhar a notação
de multiplicação clássica e escrita

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$

para destacar o fato de que a troca da ordem dos vetores produzirá a mesma resposta.

The dot product :eqref:`eq_dot_def` also admits a geometric interpretation: it is closely related to the angle between two vectors.  Consider the angle shown in :numref:`fig_angle`.

![Entre quaisquer dois vetores no plano, existe um ângulo bem definido $\theta$. Veremos que esse ângulo está intimamente ligado ao produto escalar.](../img/vec-angle.svg)
:label:`fig_angle`

Para começar, vamos considerar dois vetores específicos:

$$
\mathbf{v} = (r,0) \; \text{and} \; \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$

O vetor $\mathbf{v}$ tem comprimento $r$ e corre paralelo ao eixo $x$,
e o vetor $\mathbf{w}$ tem comprimento $s$ e está no ângulo $\theta$ com o eixo $x$.
Se calcularmos o produto escalar desses dois vetores, vemos que

$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$

Com alguma manipulação algébrica simples, podemos reorganizar os termos para obter

$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$

Em suma, para esses dois vetores específicos,
o produto escalar combinado com as normas nos informa o ângulo entre os dois vetores. Este mesmo fato é verdade em geral. Não iremos derivar a expressão aqui, no entanto,
se considerarmos escrever $\|\mathbf{v} - \mathbf{w}\|^2$ de duas maneiras:
um com o produto escalar e o outro geometricamente usando a lei dos cossenos,
podemos obter o relacionamento completo.
Na verdade, para quaisquer dois vetores $\mathbf{v}$ e $\mathbf{w}$,
o ângulo entre os dois vetores é

$$\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).$$
:eqlabel:`eq_angle_forumla`


Este é um bom resultado, pois nada no cálculo faz referência a duas dimensões.
Na verdade, podemos usar isso em três ou três milhões de dimensões sem problemas.

Como um exemplo simples, vamos ver como calcular o ângulo entre um par de vetores:

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import gluon, np, npx
npx.set_np()

def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
from torchvision import transforms
import torchvision

def angle(v, w):
    return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))

angle(torch.tensor([0, 1, 2], dtype=torch.float32), torch.tensor([2.0, 3, 4]))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

def angle(v, w):
    return tf.acos(tf.tensordot(v, w, axes=1) / (tf.norm(v) * tf.norm(w)))

angle(tf.constant([0, 1, 2], dtype=tf.float32), tf.constant([2.0, 3, 4]))
```

Não o usaremos agora, mas é útil saber
que iremos nos referir a vetores para os quais o ângulo é $\pi/2$
(ou equivalentemente $90^{\circ}$) como sendo *ortogonal*.
Examinando a equação acima, vemos que isso acontece quando $\theta = \pi/2$,
que é a mesma coisa que $\cos(\theta) = 0$.
A única maneira de isso acontecer é se o produto escalar em si for zero,
e dois vetores são ortogonais se e somente se $\mathbf{v}\cdot\mathbf{w} = 0$.
Esta será uma fórmula útil para compreender objetos geometricamente.

É razoável perguntar: por que calcular o ângulo é útil?
A resposta vem no tipo de invariância que esperamos que os dados tenham.
Considere uma imagem e uma imagem duplicada,
onde cada valor de pixel é o mesmo, mas com $10\%$ do brilho.
Os valores dos pixels individuais estão geralmente longe dos valores originais.
Assim, se computarmos a distância entre a imagem original e a mais escura,
a distância pode ser grande.
No entanto, para a maioria dos aplicativos de ML, o *conteúdo* é o mesmo --- ainda é
uma imagem de um gato no que diz respeito a um classificador gato / cão.
No entanto, se considerarmos o ângulo, não é difícil ver
que para qualquer vetor $\mathbf{v}$, o ângulo
entre $\mathbf{v}$ e $0.1\cdot\mathbf{v}$ é zero.
Isso corresponde ao fato de que os vetores de escala
mantém a mesma direção e apenas altera o comprimento.
O ângulo considera a imagem mais escura idêntica.

Exemplos como este estão por toda parte.
No texto, podemos querer que o tópico seja discutido
para não mudar se escrevermos o dobro do tamanho do documento que diz a mesma coisa.
Para algumas codificações (como contar o número de ocorrências de palavras em algum vocabulário), isso corresponde a uma duplicação do vetor que codifica o documento,
então, novamente, podemos usar o ângulo.

### Semelhança de Cosseno
Em contextos de ML onde o ângulo é empregado
para medir a proximidade de dois vetores,
os profissionais adotam o termo *semelhança de cosseno*
para se referir à porção
$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}.
$$

O cosseno assume um valor máximo de $1$
quando os dois vetores apontam na mesma direção,
um valor mínimo de $-1$ quando apontam em direções opostas,
e um valor de $0$ quando os dois vetores são ortogonais.
Observe que se os componentes de vetores de alta dimensão
são amostrados aleatoriamente com $0$ médio,
seu cosseno será quase sempre próximo a $0$.


## Hiperplanos


Além de trabalhar com vetores, outro objeto-chave
que você deve entender para ir longe na álgebra linear
é o *hiperplano*, uma generalização para dimensões superiores
de uma linha (duas dimensões) ou de um plano (três dimensões).
Em um espaço vetorial $d$-dimensional, um hiperplano tem $d-1$ dimensões
e divide o espaço em dois meios-espaços.

Vamos começar com um exemplo.
Suponha que temos um vetor coluna $\mathbf{w}=[2,1]^\top$.  Queremos saber, "quais são os pontos $\mathbf{v}$ com $\mathbf{w}\cdot\mathbf{v} = 1$?"
Ao relembrar a conexão entre produtos escalares e ângulos acima :eqref:`eq_angle_forumla`,
podemos ver que isso é equivalente a
$$
\|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta) = 1 \; \iff \; \|\mathbf{v}\|\cos(\theta) = \frac{1}{\|\mathbf{w}\|} = \frac{1}{\sqrt{5}}.
$$

![Relembrando a trigonometria, vemos que a fórmula $\|\mathbf{v}\|\cos(\theta)$ é o comprimento da projeção do vetor $\mathbf{v}$ na direção de $\mathbf{w}$](../img/proj-vec.svg)
:label:`fig_vector-project`

Se considerarmos o significado geométrico desta expressão,
vemos que isso é equivalente a dizer
que o comprimento da projeção de $\mathbf{v}$
na direção de $\mathbf{w}$ é exatamente $1/\|\mathbf{w}\|$,  como é mostrado em :numref:`fig_vector-project`.
O conjunto de todos os pontos onde isso é verdade é uma linha
perpendicularmente ao vetor $\mathbf{w}$.
Se quiséssemos, poderíamos encontrar a equação para esta linha
e veja que é $2x + y = 1$ ou equivalentemente $y = 1 - 2x$.

Se agora olharmos para o que acontece quando perguntamos sobre o conjunto de pontos com
$\mathbf{w}\cdot\mathbf{v} > 1$ ou $\mathbf{w}\cdot\mathbf{v} < 1$,
podemos ver que estes são casos em que as projeções
são maiores ou menores que $1/\|\mathbf{w}\|$, respectivamente.
Portanto, essas duas desigualdades definem os dois lados da linha.
Desta forma, descobrimos uma maneira de cortar nosso espaço em duas metades,
onde todos os pontos de um lado têm produto escalar abaixo de um limite,
e o outro lado acima como vemos em :numref:`fig_space-division`.

![Se considerarmos agora a versão da desigualdade da expressão, vemos que nosso hiperplano (neste caso: apenas uma linha) separa o espaço em duas metades.](../img/space-division.svg)
:label:`fig_space-division`

A história em uma dimensão superior é praticamente a mesma.
Se agora tomarmos $\mathbf{w} = [1,2,3]^\top$
e perguntarmos sobre os pontos em três dimensões com $\mathbf{w}\cdot\mathbf{v} = 1$,
obtemos um plano perpendicular ao vetor dado $\mathbf{w}$.
As duas desigualdades definem novamente os dois lados do plano como é mostrado em :numref:`fig_higher-division`.

![Hiperplanos em qualquer dimensão separam o espaço em duas metades.](../img/space-division-3d.svg)
:label:`fig_higher-division`


Embora nossa capacidade de visualizar se esgote neste ponto,
nada nos impede de fazer isso em dezenas, centenas ou bilhões de dimensões.
Isso ocorre frequentemente quando se pensa em modelos aprendidos por máquina.
Por exemplo, podemos entender modelos de classificação linear
como aqueles de :numref:`sec_softmax`,
como métodos para encontrar hiperplanos que separam as diferentes classes de destino.
Nesse contexto, esses hiperplanos são freqüentemente chamados de *planos de decisão*.
A maioria dos modelos de classificação profundamente aprendidos termina
com uma camada linear alimentada em um *softmax*,
para que se possa interpretar o papel da rede neural profunda
encontrar uma incorporação não linear de modo que as classes de destino
podem ser separados de forma limpa por hiperplanos.

Para dar um exemplo feito à mão, observe que podemos produzir um modelo razoável
para classificar pequenas imagens de camisetas e calças do conjunto de dados do Fashion MNIST
(visto em :numref:`sec_fashion_mnist`)
apenas pegando o vetor entre seus meios para definir o plano de decisão
e olho um limiar bruto. Primeiro, carregaremos os dados e calcularemos as médias.

```{.python .input}
# Load in the dataset
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# Compute averages
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab pytorch
# Load in the dataset
trans = []
trans.append(transforms.ToTensor())
trans = transforms.Compose(trans)
train = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                          train=True, download=True)
test = torchvision.datasets.FashionMNIST(root="../data", transform=trans,
                                         train=False, download=True)

X_train_0 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 0]).type(torch.float32)
X_train_1 = torch.stack(
    [x[0] * 256 for x in train if x[1] == 1]).type(torch.float32)
X_test = torch.stack(
    [x[0] * 256 for x in test if x[1] == 0 or x[1] == 1]).type(torch.float32)
y_test = torch.stack([torch.tensor(x[1]) for x in test
                      if x[1] == 0 or x[1] == 1]).type(torch.float32)

# Compute averages
ave_0 = torch.mean(X_train_0, axis=0)
ave_1 = torch.mean(X_train_1, axis=0)
```

```{.python .input}
#@tab tensorflow
# Load in the dataset
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.fashion_mnist.load_data()


X_train_0 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 0]] * 256), dtype=tf.float32)
X_train_1 = tf.cast(tf.stack(train_images[[i for i, label in enumerate(
    train_labels) if label == 1]] * 256), dtype=tf.float32)
X_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 0]] * 256), dtype=tf.float32)
y_test = tf.cast(tf.stack(test_images[[i for i, label in enumerate(
    test_labels) if label == 1]] * 256), dtype=tf.float32)

# Compute averages
ave_0 = tf.reduce_mean(X_train_0, axis=0)
ave_1 = tf.reduce_mean(X_train_1, axis=0)
```

Pode ser informativo examinar essas médias em detalhes, portanto, vamos representar graficamente sua aparência. Nesse caso, vemos que a média realmente se assemelha a uma imagem borrada de uma camiseta.

```{.python .input}
#@tab mxnet, pytorch
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(tf.reshape(ave_0, (28, 28)), cmap='Greys')
d2l.plt.show()
```

No segundo caso, vemos novamente que a média se assemelha a uma imagem borrada de calças.

```{.python .input}
#@tab mxnet, pytorch
# Plot average trousers
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot average trousers
d2l.plt.imshow(tf.reshape(ave_1, (28, 28)), cmap='Greys')
d2l.plt.show()
```

Em uma solução totalmente aprendida pela máquina, aprenderíamos o limite do conjunto de dados. Nesse caso, simplesmente analisamos um limite que parecia bom nos dados de treinamento à mão.

```{.python .input}
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > -1500000

# Accuracy
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

```{.python .input}
#@tab pytorch
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
# '@' is Matrix Multiplication operator in pytorch.
predictions = X_test.reshape(2000, -1) @ (w.flatten()) > -1500000

# Accuracy
torch.mean(predictions.type(y_test.dtype) == y_test, dtype=torch.float64)
```

```{.python .input}
#@tab tensorflow
# Print test set accuracy with eyeballed threshold
w = tf.transpose(ave_1 - ave_0)
predictions = tf.reduce_sum(X_test * tf.nest.flatten(w), axis=0) > -1500000

# Accuracy
tf.reduce_mean(
    tf.cast(tf.cast(predictions, y_test.dtype) == y_test, tf.float32))
```

## Geometria de Transformações Lineares


Por meio de :numref:`sec_linear-algebra` e das discussões acima,
temos um conhecimento sólido da geometria de vetores, comprimentos e ângulos.
No entanto, há um objeto importante que omitimos de discutir,
e essa é uma compreensão geométrica das transformações lineares representadas por matrizes. Totalmente internalizando o que as matrizes podem fazer para transformar dados
entre dois espaços de dimensões elevadas potencialmente diferentes requer prática significativa,
e está além do escopo deste apêndice.
No entanto, podemos começar a construir a intuição em duas dimensões.

Suponha que temos alguma matriz:

$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$

Se quisermos aplicar isso a um vetor arbitrário
$\mathbf{v} = [x, y]^\top$,
nós nos multiplicamos e vemos que

$$
\begin{aligned}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$


Isso pode parecer um cálculo estranho,
onde algo claro se tornou algo impenetrável.
No entanto, isso nos diz que podemos escrever da maneira
que uma matriz transforma *qualquer* vetor
em termos de como ele transforma *dois vetores específicos*:
$[1,0]^\top$ and $[0,1]^\top$.
Vale a pena considerar isso por um momento.
Nós essencialmente reduzimos um problema infinito
(o que acontece com qualquer par de números reais)
para um finito (o que acontece com esses vetores específicos).
Esses vetores são um exemplo de *vetores canônicos*,
onde podemos escrever qualquer vetor em nosso espaço
como uma soma ponderada desses *vetores canônicos*.

Vamos desenhar o que acontece quando usamos a matriz específica

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$


Se olharmos para o vetor específico $\mathbf{v} = [2, -1]^\top$,
vemos que é $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$,
e assim sabemos que a matriz $A$ irá enviar isso para
$2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$.
Se seguirmos essa lógica com cuidado,
digamos, considerando a grade de todos os pares inteiros de pontos,
vemos que o que acontece é que a multiplicação da matriz
pode inclinar, girar e dimensionar a grade,
mas a estrutura da grade deve permanecer como você vê em :numref:`fig_grid-transform`.

![A matriz $\mathbf{A}$ agindo nos vetores de base dados. Observe como toda a grade é transportada junto com ela.](../img/grid-transform.svg)
:label:`fig_grid-transform`

Este é o ponto intuitivo mais importante
para internalizar sobre transformações lineares representadas por matrizes.
As matrizes são incapazes de distorcer algumas partes do espaço de maneira diferente de outras.
Tudo o que elas podem fazer é pegar as coordenadas originais em nosso espaço
e inclinar, girar e dimensioná-las.

Algumas distorções podem ser graves. Por exemplo, a matriz

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix},
$$

comprime todo o plano bidimensional em uma única linha.
Identificar e trabalhar com essas transformações é o tópico de uma seção posterior,
mas geometricamente podemos ver que isso é fundamentalmente diferente
dos tipos de transformações que vimos acima.
Por exemplo, o resultado da matriz $\mathbf{A}$ pode ser "dobrado" para a grade original. Os resultados da matriz $\mathbf{B}$ não podem
porque nunca saberemos de onde o vetor $[1,2]^\top$ veio --- estava
it $[1,1]^\top$ ou $[0, -1]^\top$?

Embora esta imagem fosse para uma matriz $2\times2$,
nada nos impede de levar as lições aprendidas para dimensões superiores.
Se tomarmos vetores de base semelhantes como $[1,0, \ldots,0]$
e ver para onde nossa matriz os envia,
podemos começar a ter uma ideia de como a multiplicação da matriz
distorce todo o espaço em qualquer dimensão de espaço com a qual estamos lidando.

## Dependência Linear

Considere novamente a matriz

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$

This compresses the entire plane down to live on the single line $y = 2x$.
The question now arises: is there some way we can detect this
just looking at the matrix itself?
The answer is that indeed we can.
Let us take $\mathbf{b}_1 = [2,4]^\top$ and $\mathbf{b}_2 = [-1, -2]^\top$
be the two columns of $\mathbf{B}$.
Remember that we can write everything transformed by the matrix $\mathbf{B}$
as a weighted sum of the columns of the matrix:
like $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$.
We call this a *linear combination*.
The fact that $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$
means that we can write any linear combination of those two columns
entirely in terms of say $\mathbf{b}_2$ since

$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$

This means that one of the columns is, in a sense, redundant
because it does not define a unique direction in space.
This should not surprise us too much
since we already saw that this matrix
collapses the entire plane down into a single line.
Moreover, we see that the linear dependence
$\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ captures this.
To make this more symmetrical between the two vectors, we will write this as

$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

In general, we will say that a collection of vectors
$\mathbf{v}_1, \ldots, \mathbf{v}_k$ are *linearly dependent*
if there exist coefficients $a_1, \ldots, a_k$ *not all equal to zero* so that

$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$

In this case, we can solve for one of the vectors
in terms of some combination of the others,
and effectively render it redundant.
Thus, a linear dependence in the columns of a matrix
is a witness to the fact that our matrix
is compressing the space down to some lower dimension.
If there is no linear dependence we say the vectors are *linearly independent*.
If the columns of a matrix are linearly independent,
no compression occurs and the operation can be undone.

## Rank

If we have a general $n\times m$ matrix,
it is reasonable to ask what dimension space the matrix maps into.
A concept known as the *rank* will be our answer.
In the previous section, we noted that a linear dependence
bears witness to compression of space into a lower dimension
and so we will be able to use this to define the notion of rank.
In particular, the rank of a matrix $\mathbf{A}$
is the largest number of linearly independent columns
amongst all subsets of columns. For example, the matrix

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix},
$$

has $\mathrm{rank}(B)=1$, since the two columns are linearly dependent,
but either column by itself is not linearly dependent.
For a more challenging example, we can consider

$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$

and show that $\mathbf{C}$ has rank two since, for instance,
the first two columns are linearly independent,
however any of the four collections of three columns are dependent.

This procedure, as described, is very inefficient.
It requires looking at every subset of the columns of our given matrix,
and thus is potentially exponential in the number of columns.
Later we will see a more computationally efficient way
to compute the rank of a matrix, but for now,
this is sufficient to see that the concept
is well defined and understand the meaning.

## Invertibility

We have seen above that multiplication by a matrix with linearly dependent columns
cannot be undone, i.e., there is no inverse operation that can always recover the input.  However, multiplication by a full-rank matrix
(i.e., some $\mathbf{A}$ that is $n \times n$ matrix with rank $n$),
we should always be able to undo it.  Consider the matrix

$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}.
$$

which is the matrix with ones along the diagonal, and zeros elsewhere.
We call this the *identity* matrix.
It is the matrix which leaves our data unchanged when applied.
To find a matrix which undoes what our matrix $\mathbf{A}$ has done,
we want to find a matrix $\mathbf{A}^{-1}$ such that

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} =  \mathbf{I}.
$$

If we look at this as a system, we have $n \times n$ unknowns
(the entries of $\mathbf{A}^{-1}$) and $n \times n$ equations
(the equality that needs to hold between every entry of the product $\mathbf{A}^{-1}\mathbf{A}$ and every entry of $\mathbf{I}$)
so we should generically expect a solution to exist.
Indeed, in the next section we will see a quantity called the *determinant*,
which has the property that as long as the determinant is not zero, we can find a solution.  We call such a matrix $\mathbf{A}^{-1}$ the *inverse* matrix.
As an example, if $\mathbf{A}$ is the general $2 \times 2$ matrix

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

then we can see that the inverse is

$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}.
$$

We can test to see this by seeing that multiplying
by the inverse given by the formula above works in practice.

```{.python .input}
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

```{.python .input}
#@tab pytorch
M = torch.tensor([[1, 2], [1, 4]], dtype=torch.float32)
M_inv = torch.tensor([[2, -1], [-0.5, 0.5]])
M_inv @ M
```

```{.python .input}
#@tab tensorflow
M = tf.constant([[1, 2], [1, 4]], dtype=tf.float32)
M_inv = tf.constant([[2, -1], [-0.5, 0.5]])
tf.matmul(M_inv, M)
```

### Numerical Issues
While the inverse of a matrix is useful in theory,
we must say that most of the time we do not wish
to *use* the matrix inverse to solve a problem in practice.
In general, there are far more numerically stable algorithms
for solving linear equations like

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$

than computing the inverse and multiplying to get

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$

Just as division by a small number can lead to numerical instability,
so can inversion of a matrix which is close to having low rank.

Moreover, it is common that the matrix $\mathbf{A}$ is *sparse*,
which is to say that it contains only a small number of non-zero values.
If we were to explore examples, we would see
that this does not mean the inverse is sparse.
Even if $\mathbf{A}$ was a $1$ million by $1$ million matrix
with only $5$ million non-zero entries
(and thus we need only store those $5$ million),
the inverse will typically have almost every entry non-negative,
requiring us to store all $1\text{M}^2$ entries---that is $1$ trillion entries!

While we do not have time to dive all the way into the thorny numerical issues
frequently encountered when working with linear algebra,
we want to provide you with some intuition about when to proceed with caution,
and generally avoiding inversion in practice is a good rule of thumb.

## Determinant
The geometric view of linear algebra gives an intuitive way
to interpret a fundamental quantity known as the *determinant*.
Consider the grid image from before, but now with a highlighted region (:numref:`fig_grid-filled`).

![The matrix $\mathbf{A}$ again distorting the grid.  This time, I want to draw particular attention to what happens to the highlighted square.](../img/grid-transform-filled.svg)
:label:`fig_grid-filled`

Look at the highlighted square.  This is a square with edges given
by $(0, 1)$ and $(1, 0)$ and thus it has area one.
After $\mathbf{A}$ transforms this square,
we see that it becomes a parallelogram.
There is no reason this parallelogram should have the same area
that we started with, and indeed in the specific case shown here of

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix},
$$

it is an exercise in coordinate geometry to compute
the area of this parallelogram and obtain that the area is $5$.

In general, if we have a matrix

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

we can see with some computation that the area
of the resulting parallelogram is $ad-bc$.
This area is referred to as the *determinant*.

Let us check this quickly with some example code.

```{.python .input}
import numpy as np
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
torch.det(torch.tensor([[1, -1], [2, 3]], dtype=torch.float32))
```

```{.python .input}
#@tab tensorflow
tf.linalg.det(tf.constant([[1, -1], [2, 3]], dtype=tf.float32))
```

The eagle-eyed amongst us will notice
that this expression can be zero or even negative.
For the negative term, this is a matter of convention
taken generally in mathematics:
if the matrix flips the figure,
we say the area is negated.
Let us see now that when the determinant is zero, we learn more.

Let us consider

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$

If we compute the determinant of this matrix,
we get $2\cdot(-2 ) - 4\cdot(-1) = 0$.
Given our understanding above, this makes sense.
$\mathbf{B}$ compresses the square from the original image
down to a line segment, which has zero area.
And indeed, being compressed into a lower dimensional space
is the only way to have zero area after the transformation.
Thus we see the following result is true:
a matrix $A$ is invertible if and only if
the determinant is not equal to zero.

As a final comment, imagine that we have any figure drawn on the plane.
Thinking like computer scientists, we can decompose
that figure into a collection of little squares
so that the area of the figure is in essence
just the number of squares in the decomposition.
If we now transform that figure by a matrix,
we send each of these squares to parallelograms,
each one of which has area given by the determinant.
We see that for any figure, the determinant gives the (signed) number
that a matrix scales the area of any figure.

Computing determinants for larger matrices can be laborious,
but the  intuition is the same.
The determinant remains the factor
that $n\times n$ matrices scale $n$-dimensional volumes.

## Tensors and Common Linear Algebra Operations

In :numref:`sec_linear-algebra` the concept of tensors was introduced.
In this section, we will dive more deeply into tensor contractions
(the tensor equivalent of matrix multiplication),
and see how it can provide a unified view
on a number of matrix and vector operations.

With matrices and vectors we knew how to multiply them to transform data.
We need to have a similar definition for tensors if they are to be useful to us.
Think about matrix multiplication:

$$
\mathbf{C} = \mathbf{A}\mathbf{B},
$$

or equivalently

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$

This pattern is one we can repeat for tensors.
For tensors, there is no one case of what
to sum over that can be universally chosen,
so we need specify exactly which indices we want to sum over.
For instance we could consider

$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$

Such a transformation is called a *tensor contraction*.
It can represent a far more flexible family of transformations
that matrix multiplication alone.

As a often-used notational simplification,
we can notice that the sum is over exactly those indices
that occur more than once in the expression,
thus people often work with *Einstein notation*,
where the summation is implicitly taken over all repeated indices.
This gives the compact expression:

$$
y_{il} = x_{ijkl}a_{jk}.
$$

### Common Examples from Linear Algebra

Let us see how many of the linear algebraic definitions
we have seen before can be expressed in this compressed tensor notation:

* $\mathbf{v} \cdot \mathbf{w} = \sum_i v_iw_i$
* $\|\mathbf{v}\|_2^{2} = \sum_i v_iv_i$
* $(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij}v_j$
* $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
* $\mathrm{tr}(\mathbf{A}) = \sum_i a_{ii}$

In this way, we can replace a myriad of specialized notations with short tensor expressions.

### Expressing in Code
Tensors may flexibly be operated on in code as well.
As seen in :numref:`sec_linear-algebra`,
we can create tensors as is shown below.

```{.python .input}
# Define tensors
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab pytorch
# Define tensors
B = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = torch.tensor([[1, 2], [3, 4]])
v = torch.tensor([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

```{.python .input}
#@tab tensorflow
# Define tensors
B = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = tf.constant([[1, 2], [3, 4]])
v = tf.constant([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

Einstein summation has been implemented directly.
The indices that occurs in the Einstein summation can be passed as a string,
followed by the tensors that are being acted upon.
For instance, to implement matrix multiplication,
we can consider the Einstein summation seen above
($\mathbf{A}\mathbf{v} = a_{ij}v_j$)
and strip out the indices themselves to get the implementation:

```{.python .input}
# Reimplement matrix multiplication
np.einsum("ij, j -> i", A, v), A.dot(v)
```

```{.python .input}
#@tab pytorch
# Reimplement matrix multiplication
torch.einsum("ij, j -> i", A, v), A@v
```

```{.python .input}
#@tab tensorflow
# Reimplement matrix multiplication
tf.einsum("ij, j -> i", A, v), tf.matmul(A, tf.reshape(v, (2, 1)))
```

This is a highly flexible notation.
For instance if we want to compute
what would be traditionally written as

$$
c_{kl} = \sum_{ij} \mathbf{b}_{ijk}\mathbf{a}_{il}v_j.
$$

it can be implemented via Einstein summation as:

```{.python .input}
np.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab pytorch
torch.einsum("ijk, il, j -> kl", B, A, v)
```

```{.python .input}
#@tab tensorflow
tf.einsum("ijk, il, j -> kl", B, A, v)
```

This notation is readable and efficient for humans,
however bulky if for whatever reason
we need to generate a tensor contraction programmatically.
For this reason, `einsum` provides an alternative notation
by providing integer indices for each tensor.
For example, the same tensor contraction can also be written as:

```{.python .input}
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

```{.python .input}
#@tab pytorch
# PyTorch doesn't support this type of notation.
```

```{.python .input}
#@tab tensorflow
# TensorFlow doesn't support this type of notation.
```

Either notation allows for concise and efficient representation of tensor contractions in code.

## Summary
* Vectors can be interpreted geometrically as either points or directions in space.
* Dot products define the notion of angle to arbitrarily high-dimensional spaces.
* Hyperplanes are high-dimensional generalizations of lines and planes.  They can be used to define decision planes that are often used as the last step in a classification task.
* Matrix multiplication can be geometrically interpreted as uniform distortions of the underlying coordinates. They represent a very restricted, but mathematically clean, way to transform vectors.
* Linear dependence is a way to tell when a collection of vectors are in a lower dimensional space than we would expect (say you have $3$ vectors living in a $2$-dimensional space). The rank of a matrix is the size of the largest subset of its columns that are linearly independent.
* When a matrix's inverse is defined, matrix inversion allows us to find another matrix that undoes the action of the first. Matrix inversion is useful in theory, but requires care in practice owing to numerical instability.
* Determinants allow us to measure how much a matrix expands or contracts a space. A nonzero determinant implies an invertible (non-singular) matrix and a zero-valued determinant means that the matrix is non-invertible (singular).
* Tensor contractions and Einstein summation provide for a neat and clean notation for expressing many of the computations that are seen in machine learning.

## Exercises
1. What is the angle between
$$
\vec v_1 = \begin{bmatrix}
1 \\ 0 \\ -1 \\ 2
\end{bmatrix}, \qquad \vec v_2 = \begin{bmatrix}
3 \\ 1 \\ 0 \\ 1
\end{bmatrix}?
$$
2. True or false: $\begin{bmatrix}1 & 2\\0&1\end{bmatrix}$ and $\begin{bmatrix}1 & -2\\0&1\end{bmatrix}$ are inverses of one another?
3. Suppose that we draw a shape in the plane with area $100\mathrm{m}^2$.  What is the area after transforming the figure by the matrix
$$
\begin{bmatrix}
2 & 3\\
1 & 2
\end{bmatrix}.
$$
4. Which of the following sets of vectors are linearly independent?
 * $\left\{\begin{pmatrix}1\\0\\-1\end{pmatrix}, \begin{pmatrix}2\\1\\-1\end{pmatrix}, \begin{pmatrix}3\\1\\1\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}, \begin{pmatrix}1\\1\\1\end{pmatrix}, \begin{pmatrix}0\\0\\0\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\-1\end{pmatrix}, \begin{pmatrix}1\\0\\1\end{pmatrix}\right\}$
5. Suppose that you have a matrix written as $A = \begin{bmatrix}c\\d\end{bmatrix}\cdot\begin{bmatrix}a & b\end{bmatrix}$ for some choice of values $a, b, c$, and $d$.  True or false: the determinant of such a matrix is always $0$?
6. The vectors $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ and $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$ are orthogonal.  What is the condition on a matrix $A$ so that $Ae_1$ and $Ae_2$ are orthogonal?
7. How can you write $\mathrm{tr}(\mathbf{A}^4)$ in Einstein notation for an arbitrary matrix $A$?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/410)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1084)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1085)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTIyOTA4NDY4NiwxOTkxODk4NTM4LC0xNj
QzNjc4Nzk0LC03NTk4ODE3MDYsMjIyMjc4MTIyLC0zNDg2ODI3
MzEsLTkyMzYxMjc1NywxMTk1NzY0MTgxLC0yMTA3MzY0Mzc2LD
IxMDI4NzE3ODUsLTczMzAxNjM1OCwtMTU4OTIxMzA2NF19
-->
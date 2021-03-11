# Linear Algebra
:label:`sec_linear-algebra`

Agora que você pode armazenar e manipular dados,
vamos revisar brevemente o subconjunto da álgebra linear básica
que você precisa para entender e implementar
a maioria dos modelos cobertos neste livro.
Abaixo, apresentamos os objetos matemáticos básicos, aritméticos,
e operações em álgebra linear,
expressar cada um deles por meio de notação matemática
e a implementação correspondente em código.

## Escalares

Se você nunca estudou álgebra linear ou aprendizado de máquina,
então sua experiência anterior com matemática provavelmente consistia
de pensar em um número de cada vez.
E, se você já equilibrou um talão de cheques
ou até mesmo pagou por um jantar em um restaurante
então você já sabe como fazer coisas básicas
como adicionar e multiplicar pares de números.
Por exemplo, a temperatura em Palo Alto é de $52$ graus Fahrenheit.

Formalmente, chamamos de valores que consistem
de apenas uma quantidade numérica *escalar*.
Se você quiser converter este valor para Celsius
(escala de temperatura mais sensível do sistema métrico),
você avaliaria a expressão $c = \frac{5}{9}(f - 32)$, definindo $f$ para $52$.
Nesta equação, cada um dos termos---$5$, $9$, e $32$---são valores escalares.
Os marcadores $c$ e $f$ são chamados de *variáveis*
e eles representam valores escalares desconhecidos.

Neste livro, adotamos a notação matemática
onde as variáveis escalares são denotadas
por letras minúsculas comuns (por exemplo, $x$, $y$, and $z$).
Denotamos o espaço de todos os escalares (contínuos) *com valor real* por $\mathbb{R}$.
Por conveniência, vamos lançar mão de definições rigorosas
do que exatamente é *espaço*,
mas lembre-se por enquanto que a expressão $x \in \mathbb{R}$
é uma maneira formal de dizer que $x$ é um escalar com valor real.
O símbolo $\in$ pode ser pronunciado "em"
e simplesmente denota associação em um conjunto.
Analogamente, poderíamos escrever $x, y \in \{0, 1\}$
para afirmar que $x$ e $y$ são números
cujo valor só pode ser $0$ ou $1$.

(**Um escalar é representado por um tensor com apenas um elemento.**)
No próximo trecho de código, instanciamos dois escalares
e realizar algumas operações aritméticas familiares com eles,
a saber, adição, multiplicação, divisão e exponenciação.

```{.python .input}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
#@tab pytorch
import torch

x = torch.tensor([3.0])
y = torch.tensor([2.0])

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant([3.0])
y = tf.constant([2.0])

x + y, x * y, x / y, x**y
```

## Vetores

[**Você pode pensar em um vetor simplesmente como uma lista de valores escalares.**]
Chamamos esses valores de *elementos* (*entradas* ou *componentes*) do vetor.
Quando nossos vetores representam exemplos de nosso conjunto de dados,
seus valores têm algum significado no mundo real.
Por exemplo, se estivéssemos treinando um modelo para prever
o risco de inadimplência de um empréstimo,
podemos associar cada candidato a um vetor
cujos componentes correspondem à sua receita,
tempo de emprego, número de inadimplências anteriores e outros fatores.
Se estivéssemos estudando o risco de ataques cardíacos que os pacientes de hospitais potencialmente enfrentam,
podemos representar cada paciente por um vetor
cujos componentes capturam seus sinais vitais mais recentes,
níveis de colesterol, minutos de exercício por dia, etc.
Em notação matemática, geralmente denotamos os vetores em negrito,
letras minúsculas (por exemplo,, $\mathbf{x}$, $\mathbf{y}$, and $\mathbf{z})$.

Trabalhamos com vetores via tensores unidimensionais.
Em geral, os tensores podem ter comprimentos arbitrários,
sujeito aos limites de memória de sua máquina.

```{.python .input}
x = np.arange(4)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4)
x
```

Podemos nos referir a qualquer elemento de um vetor usando um subscrito.
Por exemplo, podemos nos referir ao elemento $i^\mathrm{th}$ element of $\mathbf{x}$  por $x_i$.
Observe que o elemento $x_i$ é um escalar,
portanto, não colocamos a fonte em negrito quando nos referimos a ela.
A literatura extensa considera os vetores de coluna como o padrão
orientação de vetores, este livro também.
Em matemática, um vetor $\mathbf{x}$ pode ser escrito como

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`


onde $x_1, \ldots, x_n$ são elementos do vetor. 
No código (**acessamos qualquer elemento indexando no tensor**)

```{.python .input}
x[3]
```

```{.python .input}
#@tab pytorch
x[3]
```

```{.python .input}
#@tab tensorflow
x[3]
```

### Comprimento, Dimensionalidade e Forma

Vamos revisitar alguns conceitos de: numref: `sec_ndarray`.
Um vetor é apenas uma matriz de números.
E assim como todo array tem um comprimento, todo vetor também.
Em notação matemática, se quisermos dizer que um vetor $\mathbf{x}$
consiste em $n$ escalares com valor real,
podemos expressar isso como $\mathbf{x} \in \mathbb{R}^n$.
O comprimento de um vetor é comumente chamado de *dimensão* do vetor.

Tal como acontece com uma matriz Python comum,
nós [**podemos acessar o comprimento de um tensor**]
chamando a função `len ()` embutida do Python.

```{.python .input}
len(x)
```

```{.python .input}
#@tab pytorch
len(x)
```

```{.python .input}
#@tab tensorflow
len(x)
```

Quando um tensor representa um vetor (com precisamente um eixo),
também podemos acessar seu comprimento por meio do atributo `.shape`.
A forma é uma tupla que lista o comprimento (dimensionalidade)
ao longo de cada eixo do tensor.
(**Para tensores com apenas um eixo, a forma tem apenas um elemento.**)

```{.python .input}
x.shape
```

```{.python .input}
#@tab pytorch
x.shape
```

```{.python .input}
#@tab tensorflow
x.shape
```

Observe que a palavra "dimensão" tende a ficar sobrecarregada
nesses contextos e isso tende a confundir as pessoas.
Para esclarecer, usamos a dimensionalidade de um *vetor* ou um *eixo*
para se referir ao seu comprimento, ou seja, o número de elementos de um vetor ou eixo.
No entanto, usamos a dimensionalidade de um tensor
para se referir ao número de eixos que um tensor possui.
Nesse sentido, a dimensionalidade de algum eixo de um tensor
será o comprimento desse eixo.


## Matrizes

Assim como os vetores generalizam escalares de ordem zero para ordem um,
matrizes generalizam vetores de ordem um para ordem dois.
Matrizes, que normalmente denotamos com letras maiúsculas em negrito
(por exemplo, $\mathbf{X}$, $\mathbf{Y}$, and $\mathbf{Z}$),
são representados no código como tensores com dois eixos.

Em notação matemática, usamos $\mathbf{A} \in \mathbb{R}^{m \times n}$
para expressar que a matriz $\mathbf {A}$ consiste em $m$ linhas e $n$ colunas de escalares com valor real.
Visualmente, podemos ilustrar qualquer matriz $\mathbf{A} \in \mathbb{R}^{m \times n}$ como uma tabela,
onde cada elemento $a_{ij}$ pertence à linha $i^{\mathrm{th}}$ e coluna $j^{\mathrm{th}}$:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

Para qualquer $\mathbf{A} \in \mathbb{R}^{m \times n}$, a forma de $\mathbf{A}$
é ($m$, $n$) ou $m \times n$.
Especificamente, quando uma matriz tem o mesmo número de linhas e colunas,
sua forma se torna um quadrado; portanto, é chamada de *matriz quadrada*.

Podemos [**criar uma matriz $m \times n$**]
especificando uma forma com dois componentes $m$ e $n$
ao chamar qualquer uma de nossas funções favoritas para instanciar um tensor.

```{.python .input}
A = np.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab pytorch
A = torch.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20), (5, 4))
A
```

Podemos acessar o elemento escalar $a_{ij}$ de uma matriz $\mathbf{A}$ em: eqref: `eq_matrix_def`
especificando os índices para a linha ($i$) e coluna ($j$),
como $[\mathbf {A}] _ {ij}$.
Quando os elementos escalares de uma matriz $\mathbf{A}$, como em: eqref: `eq_matrix_def`, não são fornecidos,
podemos simplesmente usar a letra minúscula da matriz $\mathbf{A}$ com o subscrito do índice, $a_{ij}$,
para se referir a $[\mathbf {A}] _ {ij}$.
Para manter a notação simples, as vírgulas são inseridas para separar os índices apenas quando necessário,
como $a_ {2, 3j}$ e $[\mathbf {A}] _ {2i-1, 3}$.

Às vezes, queremos inverter os eixos.
Quando trocamos as linhas e colunas de uma matriz,
o resultado é chamado de *transposição* da matriz.
Formalmente, significamos uma matriz $\mathbf {A}$ transposta por $\mathbf {A} ^ \ top$
e se $\mathbf {B} = \mathbf {A} ^ \top$, então $b_ {ij} = a_ {ji}$ para qualquer $i$ e $j$.
Assim, a transposição de $\mathbf {A}$ em: eqref: `eq_matrix_def` é
uma matriz $n \times m$:
$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

Agora acessamoas a  (**matriz transposta**) via código.

```{.python .input}
A.T
```

```{.python .input}
#@tab pytorch
A.T
```

```{.python .input}
#@tab tensorflow
tf.transpose(A)
```
Como um tipo especial de matriz quadrada,
[**a *matriz simétrica* $\mathbf {A}$ é igual à sua transposta:
$\mathbf{A} = \mathbf{A}^\top$.**]
Aqui definimos uma matriz simétrica `B`.

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab pytorch
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab tensorflow
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

Agora comparamos `B` com sua transposta.


```{.python .input}
B == B.T
```

```{.python .input}
#@tab pytorch
B == B.T
```

```{.python .input}
#@tab tensorflow
B == tf.transpose(B)
```

Matrizes são estruturas de dados úteis:
eles nos permitem organizar dados que têm diferentes modalidades de variação.
Por exemplo, as linhas em nossa matriz podem corresponder a diferentes casas (exemplos de dados),
enquanto as colunas podem corresponder a diferentes atributos.
Isso deve soar familiar se você já usou um software de planilha ou
leu: numref: `sec_pandas`.
Assim, embora a orientação padrão de um único vetor seja um vetor coluna,
em uma matriz que representa um conjunto de dados tabular,
é mais convencional tratar cada exemplo de dados como um vetor linha na matriz.
E, como veremos em capítulos posteriores,
esta convenção permitirá práticas comuns de aprendizado profundo.
Por exemplo, ao longo do eixo mais externo de um tensor,
podemos acessar ou enumerar *minibatches* de exemplos de dados,
ou apenas exemplos de dados se não houver *minibatch*.


## Tensores

Assim como vetores generalizam escalares e matrizes generalizam vetores, podemos construir estruturas de dados com ainda mais eixos.
[**Tensores**]
("tensores" nesta subseção referem-se a objetos algébricos)
(**nos dê uma maneira genérica de descrever matrizes $n$ -dimensionais com um número arbitrário de eixos.**)
Vetores, por exemplo, são tensores de primeira ordem e matrizes são tensores de segunda ordem.
Tensores são indicados com letras maiúsculas de uma fonte especial
(por exemplo, $\mathsf{X}$, $\mathsf{Y}$, e $\mathsf{Z}$)
e seu mecanismo de indexação (por exemplo, $x_{ijk}$ e $[\mathsf{X}]_{1, 2i-1, 3}$) é semelhante ao de matrizes.

Os tensores se tornarão mais importantes quando começarmos a trabalhar com imagens,
  que chegam como matrizes $n$ -dimensionais com 3 eixos correspondentes à altura, largura e um eixo de *canal* para empilhar os canais de cores (vermelho, verde e azul). Por enquanto, vamos pular tensores de ordem superior e nos concentrar no básico.

```{.python .input}
X = np.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab pytorch
X = torch.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(24), (2, 3, 4))
X
```

## Propriedades Básicas de Aritmética de Tensores

Escalares, vetores, matrizes e tensores ("tensores" nesta subseção referem-se a objetos algébricos)
de um número arbitrário de eixos
têm algumas propriedades interessantes que muitas vezes são úteis.
Por exemplo, você deve ter notado
da definição de uma operação elemento a elemento
que qualquer operação unária elementar não altera a forma de seu operando.
Similarmente,
[**dados quaisquer dois tensores com a mesma forma,
o resultado de qualquer operação binária elementar
será um tensor da mesma forma.**]
Por exemplo, adicionar duas matrizes da mesma forma
realiza a adição elemento a elemento sobre essas duas matrizes.

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # Cria uma cópia de `A` em `B` alocando nova memória
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # Cria uma cópia de `A` em `B` alocando nova memória
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  #  Não está clonando `A` para `B` para alocar nova memória
A, A + B
```

Especificamente,
[**a multiplicação elemento a elemento de duas matrizes é chamada de *produto Hadamard***]
(notação matemática $\odot$).
Considere a matriz $\mathbf{B} \in \mathbb{R}^{m \times n}$  cujo elemento da linha $i$ e coluna $j$ é $b_ {ij}$. O produto Hadamard das matrizes $\mathbf {A}$ (definido em :eqref:`eq_matrix_def`) e $\mathbf {B}$

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```

```{.python .input}
#@tab pytorch
A * B
```

```{.python .input}
#@tab tensorflow
A * B
```

[**Multiplicar ou adicionar um tensor por um escalar**] também não muda a forma do tensor,
onde cada elemento do tensor operando será adicionado ou multiplicado pelo escalar.

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

## Redução
:label:`subseq_lin-alg-reduction`

Uma operação útil que podemos realizar com tensores arbitrários
é para
calcule [**a soma de seus elementos.**]
Em notação matemática, expressamos somas usando o símbolo $\sum$.
Para expressar a soma dos elementos em um vetor $\mathbf {x}$ de comprimento $d$,
escrevemos $\sum_ {i = 1} ^ d x_i$.
No código, podemos apenas chamar a função para calcular a soma.

```{.python .input}
x = np.arange(4)
x, x.sum()
```

```{.python .input}
#@tab pytorch
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x, tf.reduce_sum(x)
```

We can express [**sums over the elements of tensors of arbitrary shape.**]
For example, the sum of the elements of an $m \times n$ matrix $\mathbf{A}$ could be written $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.
Podemos expressar [**somas sobre os elementos de tensores de forma arbitrária.**]
Por exemplo, a soma dos elementos de uma matriz $m \times n$
$\mathbf{A}$ poderia ser escrita como $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```{.python .input}
A.shape, A.sum()
```

```{.python .input}
#@tab pytorch
A.shape, A.sum()
```

```{.python .input}
#@tab tensorflow
A.shape, tf.reduce_sum(A)
```

Por padrão, invocar a função para calcular a soma
*reduz* um tensor ao longo de todos os seus eixos a um escalar.
Também podemos [**especificar os eixos ao longo dos quais o tensor é reduzido por meio da soma**]
Pegue as matrizes como exemplo.
Para reduzir a dimensão da linha (eixo 0) somando os elementos de todas as linhas,
especificamos `axis = 0` ao invocar a função.
Uma vez que a matriz de entrada reduz ao longo do eixo 0 para gerar o vetor de saída,
a dimensão do eixo 0 da entrada é perdida na forma de saída.

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis0 = tf.reduce_sum(A, axis=0)
A_sum_axis0, A_sum_axis0.shape
```

Especificando
`eixo = 1` irá reduzir a dimensão da coluna (eixo 1) ao somar os elementos de todas as colunas.
Assim, a dimensão do eixo 1 da entrada é perdida na forma da saída.

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis1 = tf.reduce_sum(A, axis=1)
A_sum_axis1, A_sum_axis1.shape
```

Reduzindo uma matriz ao longo de ambas as linhas e colunas por meio da soma
é equivalente a somar todos os elementos da matriz.

```{.python .input}
A.sum(axis=[0, 1])  # O mesmo que `A.sum()`
```

```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1])  # O mesmo que `A.sum()`
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  # O mesmo que `tf.reduce_sum(A)`
```

[**Uma quantidade relacionada é a *média*, que também é chamada de *média*.**]
Calculamos a média dividindo a soma pelo número total de elementos.
No código, poderíamos apenas chamar a função para calcular a média
em tensores de forma arbitrária.

```{.python .input}
A.mean(), A.sum() / A.size
```

```{.python .input}
#@tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

Da mesma forma, a função de cálculo da média também pode reduzir um tensor ao longo dos eixos especificados.

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

### Soma não reducional
:label:`subseq_lin-alg-non-reduction`

Contudo,
às vezes pode ser útil [**manter o número de eixos inalterado**]
ao invocar a função para calcular a soma ou média.

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A
```

Por exemplo,
uma vez que `sum_A` ainda mantém seus dois eixos após somar cada linha, podemos (**dividir` A` por `sum_A` com *broadcasting*.**)

```{.python .input}
A / sum_A
```

```{.python .input}
#@tab pytorch
A / sum_A
```

```{.python .input}
#@tab tensorflow
A / sum_A
```

Se quisermos calcular [**a soma cumulativa dos elementos de `A` ao longo de algum eixo**], diga` eixo = 0` (linha por linha),
podemos chamar a função `cumsum`. Esta função não reduzirá o tensor de entrada ao longo de nenhum eixo.

```{.python .input}
A.cumsum(axis=0)
```

```{.python .input}
#@tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
#@tab tensorflow
tf.cumsum(A, axis=0)
```

## Produto Escalar

Até agora, realizamos apenas operações elementares, somas e médias. E se isso fosse tudo que pudéssemos fazer, a álgebra linear provavelmente não mereceria sua própria seção. No entanto, uma das operações mais fundamentais é o produto escalar.
Dados dois vetores $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, seu *produto escalar* $\mathbf{x}^\top \mathbf{y}$ (ou $\langle \mathbf{x}, \mathbf{y}  \rangle$) é uma soma sobre os produtos dos elementos na mesma posição: $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

[~~O *produto escalar* de dois vetores é uma soma sobre os produtos dos elementos na mesma posição~~]

```{.python .input}
y = np.ones(4)
x, y, np.dot(x, y)
```

```{.python .input}
#@tab pytorch
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
#@tab tensorflow
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```
Observe que
(**podemos expressar o produto escalar de dois vetores de forma equivalente, realizando uma multiplicação elemento a elemento e, em seguida, uma soma:**)

```{.python .input}
np.sum(x * y)
```

```{.python .input}
#@tab pytorch
torch.sum(x * y)
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x * y)
```

Os produtos escalares são úteis em uma ampla variedade de contextos.
Por exemplo, dado algum conjunto de valores,
denotado por um vetor $\mathbf{x}  \in \mathbb{R}^d$
e um conjunto de pesos denotado por $\mathbf{w} \in \mathbb{R}^d$,,
a soma ponderada dos valores em $\mathbf{x}$
de acordo com os pesos $\mathbf{w}$
pode ser expresso como o produto escalar $\mathbf{x}^\top \mathbf{w}$.
Quando os pesos não são negativos
e soma a um (ou seja, $\left(\sum_{i=1}^{d} {w_i} = 1\right)$),
o produto escalar expressa uma *média ponderada*.
Depois de normalizar dois vetores para ter o comprimento unitário,
os produtos escalares expressam o cosseno do ângulo entre eles.
Apresentaremos formalmente essa noção de *comprimento* posteriormente nesta seção.

## Produtos Matriz-Vetor

Agora que sabemos como calcular produtos escalares,
podemos começar a entender *produtos vetoriais de matriz*.
Lembre-se da matriz $\mathbf{A} \in \mathbb{R}^{m \times n}$
e o vetor $\mathbf{x} \in \mathbb{R}^n$
definido e visualizado em :eqref:`eq_matrix_def` e :eqref:`eq_vec_def` respectivamente.
Vamos começar visualizando a matriz $\mathbf{A}$ em termos de seus vetores linha

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

onde cada $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
é uma linha vetor representando a $i^\mathrm{th}$ linha da matriz $\mathbf{A}$.

[**O produto vetor-matriz $\mathbf{A}\mathbf{x}$
é simplesmente um vetor coluna de comprimento $m$,
cujo elemento $i^\mathrm{th}$ é o produto escalar $\mathbf{a}^\top_i \mathbf{x}$:**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

Podemos pensar na multiplicação por uma matriz $\mathbf{A}\in \mathbb{R}^{m \times n}$
como uma transformação que projeta vetores
de $\mathbb{R}^{n}$ a $\mathbb{R}^{m}$.
Essas transformações revelaram-se extremamente úteis.
Por exemplo, podemos representar rotações
como multiplicações por uma matriz quadrada.
Como veremos nos capítulos subsequentes,
também podemos usar produtos vetoriais de matriz
para descrever os cálculos mais intensivos
necessário ao calcular cada camada em uma rede neural
dados os valores da camada anterior.

Expressando produtos de vetor-matriz em código com tensores,
usamos a mesma função `dot` que para produtos de ponto.
Quando chamamos `np.dot (A, x)` com uma matriz `A` e um vetor` x`,
o produto matriz-vetor é realizado.
Observe que a dimensão da coluna de `A` (seu comprimento ao longo do eixo 1)
deve ser igual à dimensão de `x` (seu comprimento).
```{.python .input}
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
#@tab pytorch
A.shape, x.shape, torch.mv(A, x)
```

```{.python .input}
#@tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

## Multiplicação Matriz Matriz

Se você já pegou o jeito dos produtos escalares e produtos matriciais,
então a *multiplicação matriz-matriz* deve ser direta.
Digamos que temos duas matrizes $\mathbf{A} \in \mathbb{R}^{n \times k}$ e $\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


Denotada por $\mathbf{a}^\top_{i} \in \mathbb{R}^k$
o vetor linha representando a $i^\mathrm{th}$ linha da matriz $\mathbf{A}$,
e $\mathbf{b}_{j} \in \mathbb{R}^k$
seja o vetor coluna da $j^\mathrm{th}$ coluna matriz $\mathbf{B}$.
Para produzir o produto de matrizes $\mathbf{C} = \mathbf{A}\mathbf{B}$, é mais facil pensar $\mathbf{A}$ em termos de seus vetores linha $\mathbf{B}$ em termos de seus vetores coluna:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$

Então, o produto da matriz $\mathbf{C} \in \mathbb{R}^{n \times m}$ é produzido, pois simplesmente calculamos cada elemento $c_ {ij}$ como o produto escalar $\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**Podemos pensar na multiplicação matriz-matriz $\mathbf {AB}$ simplesmente realizando $m$ produtos vetoriais de matriz e juntando os resultados para formar uma matriz $n \times m$.**]
No trecho a seguir, realizamos a multiplicação da matriz em `A` e` B`.
Aqui, `A` é uma matriz com 5 linhas e 4 colunas,
e `B` é uma matriz com 4 linhas e 3 colunas.
Após a multiplicação, obtemos uma matriz com 5 linhas e 3 colunas.

```{.python .input}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

```{.python .input}
#@tab pytorch
B = torch.ones(4, 3)
torch.mm(A, B)
```

```{.python .input}
#@tab tensorflow
B = tf.ones((4, 3), tf.float32)
tf.matmul(A, B)
```

A multiplicação de matriz-matriz pode ser simplesmente chamada de *multiplicação de matrizes* e não deve ser confundida com o produto Hadamard.

## Normas
:label:`subsec_lin-algebra-norms`

Alguns dos operadores mais úteis em álgebra linear são *normas*.
Informalmente, a norma de um vetor nos diz o quão *grande* é um vetor.
A noção de *tamanho* em consideração aqui
preocupa-se não em dimensionalidade
,mas sim a magnitude dos componentes.

Na álgebra linear, uma norma vetorial é uma função $f$ que mapeia um vetor
para um escalar, satisfazendo um punhado de propriedades.
Dado qualquer vetor $\mathbf{x}$,
a primeira propriedade diz
que se escalarmos todos os elementos de um vetor
por um fator constante $\alpha$,
sua norma também escala pelo *valor absoluto*
do mesmo fator constante:

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$


A segunda propriedade é a familiar desigualdade do triângulo:

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$


A terceira propriedade simplesmente diz que a norma deve ser não negativa:

$$f(\mathbf{x}) \geq 0.$$


Isso faz sentido, pois na maioria dos contextos, o menor *tamanho* para qualquer coisa é 0.
A propriedade final requer que a menor norma seja alcançada e somente alcançada
por um vetor que consiste em todos os zeros.

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$



Você pode notar que as normas se parecem muito com medidas de distância.
E se você se lembra das distâncias euclidianas
(pense no teorema de Pitágoras) da escola primária,
então, os conceitos de não negatividade e a desigualdade do triângulo podem ser familiares.
Na verdade, a distância euclidiana é uma norma:
especificamente, é a norma $L_2$.
Suponha que os elementos no vetor $n$ -dimensional
$\mathbf{x}$ são $x_1, \ldots, x_n$.

[**A $L_2$ *norma* de $\mathbf {x}$ é a raiz quadrada da soma dos quadrados dos elementos do vetor:**]

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$**)



onde o subscrito $2$ é frequentemente omitido nas normas $L_2$, ou seja, $\|\mathbf{x}\|$ é equivalente a $\|\mathbf{x}\|_2$. Em código,
podemos calcular a norma $L_2$ de um vetor da seguinte maneira.
```{.python .input}
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
#@tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
#@tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```
Em *Deep Learning*, trabalhamos com mais frequência
com a norma $L_2$ ao quadrado.

Você também encontrará frequentemente [**a norma $L_1$**],
que é expresso como a soma dos valores absolutos dos elementos do vetor:

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**)


As compared with the $L_2$ norm,
it is less influenced by outliers.
To calculate the $L_1$ norm, we compose
the absolute value function with a sum over the elements.

Em comparação com a norma $L_2$,
é menos influenciado por outliers.
Para calcular a norma $L_1$, nós compomos
a função de valor absoluto com uma soma sobre os elementos.

```{.python .input}
np.abs(u).sum()
```

```{.python .input}
#@tab pytorch
torch.abs(u).sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(tf.abs(u))
```


Both the $L_2$ norm and the $L_1$ norm
are special cases of the more general $L_p$ *norm*:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Analogous to $L_2$ norms of vectors,
[**the *Frobenius norm* of a matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$**]
is the square root of the sum of the squares of the matrix elements:

[**$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

The Frobenius norm satisfies all the properties of vector norms.
It behaves as if it were an $L_2$ norm of a matrix-shaped vector.
Invoking the following function will calculate the Frobenius norm of a matrix.

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
#@tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
#@tab tensorflow
tf.norm(tf.ones((4, 9)))
```

### Norms and Objectives
:label:`subsec_norms_and_objectives`

While we do not want to get too far ahead of ourselves,
we can plant some intuition already about why these concepts are useful.
In deep learning, we are often trying to solve optimization problems:
*maximize* the probability assigned to observed data;
*minimize* the distance between predictions
and the ground-truth observations.
Assign vector representations to items (like words, products, or news articles)
such that the distance between similar items is minimized,
and the distance between dissimilar items is maximized.
Oftentimes, the objectives, perhaps the most important components
of deep learning algorithms (besides the data),
are expressed as norms.



## More on Linear Algebra

In just this section,
we have taught you all the linear algebra
that you will need to understand
a remarkable chunk of modern deep learning.
There is a lot more to linear algebra
and a lot of that mathematics is useful for machine learning.
For example, matrices can be decomposed into factors,
and these decompositions can reveal
low-dimensional structure in real-world datasets.
There are entire subfields of machine learning
that focus on using matrix decompositions
and their generalizations to high-order tensors
to discover structure in datasets and solve prediction problems.
But this book focuses on deep learning.
And we believe you will be much more inclined to learn more mathematics
once you have gotten your hands dirty
deploying useful machine learning models on real datasets.
So while we reserve the right to introduce more mathematics much later on,
we will wrap up this section here.

If you are eager to learn more about linear algebra,
you may refer to either the
[online appendix on linear algebraic operations](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html)
or other excellent resources :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`.



## Summary

* Scalars, vectors, matrices, and tensors are basic mathematical objects in linear algebra.
* Vectors generalize scalars, and matrices generalize vectors.
* Scalars, vectors, matrices, and tensors have zero, one, two, and an arbitrary number of axes, respectively.
* A tensor can be reduced along the specified axes by `sum` and `mean`.
* Elementwise multiplication of two matrices is called their Hadamard product. It is different from matrix multiplication.
* In deep learning, we often work with norms such as the $L_1$ norm, the $L_2$ norm, and the Frobenius norm.
* We can perform a variety of operations over scalars, vectors, matrices, and tensors.

## Exercises

1. Prove that the transpose of a matrix $\mathbf{A}$'s transpose is $\mathbf{A}$: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Given two matrices $\mathbf{A}$ and $\mathbf{B}$, show that the sum of transposes is equal to the transpose of a sum: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Given any square matrix $\mathbf{A}$, is $\mathbf{A} + \mathbf{A}^\top$ always symmetric? Why?
1. We defined the tensor `X` of shape (2, 3, 4) in this section. What is the output of `len(X)`?
1. For a tensor `X` of arbitrary shape, does `len(X)` always correspond to the length of a certain axis of `X`? What is that axis?
1. Run `A / A.sum(axis=1)` and see what happens. Can you analyze the reason?
1. When traveling between two points in Manhattan, what is the distance that you need to cover in terms of the coordinates, i.e., in terms of avenues and streets? Can you travel diagonally?
1. Consider a tensor with shape (2, 3, 4). What are the shapes of the summation outputs along axis 0, 1, and 2?
1. Feed a tensor with 3 or more axes to the `linalg.norm` function and observe its output. What does this function compute for tensors of arbitrary shape?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk5MjExNzUzLDE2NjM5NjYwODUsLTE3NT
YyNTE1NTgsMjA3NDQzOTE5MywtNDkyNTAxNjQxLDExMDU3Mzc5
NDgsNjAxODg3MjM2LDg1NzQ4ODk5OSwxNjgxMDUyNDIxLDE2Mz
k2MTI3OTksMjAzNzk2Nzg2NCwxODQ5NDk0Nzc0LDEwMTkyODgw
MzZdfQ==
-->
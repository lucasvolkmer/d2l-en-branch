# Manipulação de Dados
:label:`sec_ndarray`

Para fazer qualquer coisa, precisamos de alguma forma de armazenar e manipular dados.
Geralmente, há duas coisas importantes que precisamos fazer com os dados: (i) adquirir
eles; e (ii) processá-los assim que estiverem dentro do computador. Não há
sentido em adquirir dados sem alguma forma de armazená-los, então vamos  brincar com dados sintéticos. Para começar, apresentamos o
*array* $n$-dimensional, também chamado de *tensor*.

Se você trabalhou com NumPy, o mais amplamente utilizado
pacote de computação científica em Python,
então você achará esta seção familiar.
Não importa qual estrutura você usa,
sua *classe de tensor* (`ndarray` em MXNet,
`Tensor` em PyTorch e TensorFlow) é semelhante ao` ndarray` do NumPy com
alguns recursos interessantes.
Primeiro, a GPU é bem suportada para acelerar a computação
enquanto o NumPy suporta apenas computação de CPU.
Em segundo lugar, a classe tensor
suporta diferenciação automática.
Essas propriedades tornam a classe tensor adequada para aprendizado profundo.
Ao longo do livro, quando dizemos tensores,
estamos nos referindo a instâncias da classe tensorial, a menos que seja declarado de outra forma.

## Iniciando

Nesta seção, nosso objetivo é colocá-lo em funcionamento,
equipando você com as ferramentas básicas de matemática e computação numérica
que você desenvolverá conforme progride no livro.
Não se preocupe se você lutar para grocar alguns dos
os conceitos matemáticos ou funções de biblioteca.
As seções a seguir revisitarão este material
no contexto de exemplos práticos e irá afundar.
Por outro lado, se você já tem alguma experiência
e quiser se aprofundar no conteúdo matemático, basta pular esta seção.

:begin_tab:`mxnet`
Para começar, importamos o `np` (` numpy`) e
Módulos `npx` (` numpy_extension`) da MXNet.
Aqui, o módulo `np` inclui funções suportadas por NumPy,
enquanto o módulo `npx` contém um conjunto de extensões
desenvolvido para capacitar o *Deep Learning* em um ambiente semelhante ao NumPy.
Ao usar tensores, quase sempre invocamos a função `set_np`:
isso é para compatibilidade de processamento de tensor por outros componentes do MXNet.
:end_tab:

:begin_tab:`pytorch`
(**Para começar, importamos `torch`. Note que apesar de ser chamado PyTorch, devemos importar `torch` ao invés de `pytorch`.**)
:end_tab:

:begin_tab:`tensorflow`
Importamos `tensorflow`. Como o nome é longo, importamos abreviando `tf`.
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

[**Um tensor representa uma matriz (possivelmente multidimensional) de valores numéricos.**]
Com uma dimensão, um tensor corresponde (em matemática) a um *vetor*.
Com duas dimensões, um tensor corresponde a uma * matriz *.
Tensores com mais de dois eixos não possuem
nomes matemáticos.

Para começar, podemos usar `arange` para criar um vetor linha `x`
contendo os primeiros 12 inteiros começando com 0,
embora eles sejam criados como *float* por padrão.
Cada um dos valores em um tensor é chamado de * elemento * do tensor.
Por exemplo, existem 12 elementos no tensor `x`.
A menos que especificado de outra forma, um novo tensor
será armazenado na memória principal e designado para computação baseada em CPU.


```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12)
x
```

(**Podemos acessar o formato do tensor**) (~~e o número total de elementos~~) (o comprimento em cada coordenada)
inspecionando sua propriedade `shape` .

```{.python .input}
#@tab all
x.shape
```

Se quisermos apenas saber o número total de elementos em um tensor,
ou seja, o produto de todos os *shapes*,
podemos inspecionar seu tamanho.
Porque estamos lidando com um vetor aqui,
o único elemento de seu `shape` é idêntico ao seu tamanho.

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

Para [**mudar o *shape* de um tensor sem alterar
o número de elementos ou seus valores**],
podemos invocar a função `reshape`.
Por exemplo, podemos transformar nosso tensor, `x`,
de um vetor linha com forma (12,) para uma matriz com forma (3, 4).
Este novo tensor contém exatamente os mesmos valores,
mas os vê como uma matriz organizada em 3 linhas e 4 colunas.
Para reiterar, embora a forma tenha mudado,
os elementos não.
Observe que o tamanho não é alterado pela remodelagem.


```{.python .input}
#@tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(x, (3, 4))
X
```
A remodelação especificando manualmente todas as dimensões é desnecessária.
Se nossa forma de destino for uma matriz com forma (altura, largura),
então, depois de sabermos a largura, a altura é dada implicitamente.
Por que devemos realizar a divisão nós mesmos?
No exemplo acima, para obter uma matriz com 3 linhas,
especificamos que deve ter 3 linhas e 4 colunas.
Felizmente, os tensores podem calcular automaticamente uma dimensão considerando o resto.
Invocamos esse recurso colocando `-1` para a dimensão
que gostaríamos que os tensores inferissem automaticamente.
No nosso caso, em vez de chamar `x.reshape (3, 4)`,
poderíamos ter chamado equivalentemente `x.reshape (-1, 4)` ou `x.reshape (3, -1)`.

Normalmente, queremos que nossas matrizes sejam inicializadas
seja com zeros, uns, algumas outras constantes,
ou números amostrados aleatoriamente de uma distribuição específica.
[**Podemos criar um tensor representando um tensor com todos os elementos
definido como 0**] (~~ou 1~~)
e uma forma de (2, 3, 4) como a seguir:

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

Da mesma forma, podemos criar tensores com cada elemento definido como 1 da seguinte maneira:

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```
Frequentemente, queremos [**amostrar aleatoriamente os valores
para cada elemento em um tensor**]
de alguma distribuição de probabilidade.
Por exemplo, quando construímos matrizes para servir
como parâmetros em uma rede neural, vamos
normalmente inicializar seus valores aleatoriamente.
O fragmento a seguir cria um tensor com forma (3, 4).
Cada um de seus elementos é amostrado aleatoriamente
de uma distribuição gaussiana (normal) padrão
com uma média de 0 e um desvio padrão de 1.


```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```
Podemos também [**especificar os valores exatos para cada elemento**] no tensor desejado
fornecendo uma lista Python (ou lista de listas) contendo os valores numéricos.
Aqui, a lista externa corresponde ao eixo 0 e a lista interna ao eixo 1.


```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Operações

Este livro não é sobre engenharia de software.
Nossos interesses não se limitam a simplesmente
leitura e gravação de dados de/para matrizes.
Queremos realizar operações matemáticas nessas matrizes.
Algumas das operações mais simples e úteis
são as operações *elementwise*.
Estes aplicam uma operação escalar padrão
para cada elemento de uma matriz.
Para funções que usam dois arrays como entradas,
as operações elementwise aplicam algum operador binário padrão
em cada par de elementos correspondentes das duas matrizes.
Podemos criar uma função *elementwise* a partir de qualquer função
que mapeia de um escalar para um escalar.

Em notação matemática, denotaríamos tal
um operador escalar * unário * (tomando uma entrada)
pela assinatura $f: \mathbb{R} \rightarrow \mathbb{R}$.
Isso significa apenas que a função está mapeando
de qualquer número real ($\mathbb{R}$) para outro.
Da mesma forma, denotamos um operador escalar *binário*
(pegando duas entradas reais e produzindo uma saída)
pela assinatura $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Dados quaisquer dois vetores $\mathbf{u}$ e $\mathbf{v}$ de mesmo *shape*, 
e um operador binário $f$, podemos produzir um vetor
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$
definindo $c_i \gets f(u_i, v_i)$ para todos $i$,
onde $c_i, u_i$ e $v_i$ são os elementos $i^\mathrm{th}$
dos vetores $\mathbf{c}, \mathbf{u}$, e $\mathbf{v}$.
Aqui, nós produzimos o valor vetorial
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
*transformando* a função escalar para uma operação de vetor elemento a elemento.

Os operadores aritméticos padrão comuns
(`+`, `-`,` * `,` / `e` ** `)
foram todos transformados em operações elemento a elemento
para quaisquer tensores de formato idêntico de forma arbitrária.
Podemos chamar operações elemento a elemento em quaisquer dois tensores da mesma forma.
No exemplo a seguir, usamos vírgulas para formular uma tupla de 5 elementos,
onde cada elemento é o resultado de uma operação *elementwise*.

### Operações

[**Os operadores aritméticos padrão comuns
(`+`, `-`,` * `,` / `e` ** `)
foram todos transformados em operações elemento a elemento.**]

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # O ** é o operador exponenciação
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  #  O ** é o operador exponenciação
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  #  O ** é o operador exponenciação
```

Muitos (**mais operações podem ser aplicadas elemento a elemento**),
incluindo operadores unários como exponenciação.

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

Além de cálculos *elementwise*,
também podemos realizar operações de álgebra linear,
incluindo produtos escalar de vetor e multiplicação de matrizes.
Explicaremos as partes cruciais da álgebra linear
(sem nenhum conhecimento prévio assumido) em :numref:`sec_linear-algebra`.



Também podemos [***concatenar* vários tensores juntos,**]
empilhando-os ponta a ponta para formar um tensor maior.
Só precisamos fornecer uma lista de tensores
e informar ao sistema ao longo de qual eixo concatenar.
O exemplo abaixo mostra o que acontece quando concatenamos
duas matrizes ao longo das linhas (eixo 0, o primeiro elemento da forma)
vs. colunas (eixo 1, o segundo elemento da forma).
Podemos ver que o comprimento do eixo 0 do primeiro tensor de saída ($6$)
é a soma dos comprimentos do eixo 0 dos dois tensores de entrada ($3 + 3$);
enquanto o comprimento do eixo 1 do segundo tensor de saída ($8$)
é a soma dos comprimentos do eixo 1 dos dois tensores de entrada ($4 + 4$).
```{.python .input}
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
#@tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

Às vezes, queremos [**construir um tensor binário por meio de *declarações lógicas*.**]
Tome `X == Y` como exemplo.
Para cada posição, se `X` e` Y` forem iguais nessa posição,
a entrada correspondente no novo tensor assume o valor 1,
o que significa que a declaração lógica `X == Y` é verdadeira nessa posição;
caso contrário, essa posição assume 0.

```{.python .input}
#@tab all
X == Y
```
[**Somando todos os elementos no tensor**] resulta em um tensor com apenas um elemento.

```{.python .input}
#@tab mxnet, pytorch
X.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(X)
```

## Mecanismo de *Broadcasting* 
:label:`subsec_broadcasting`

In the above section, we saw how to perform elementwise operations
on two tensors of the same shape. Under certain conditions,
even when shapes differ, we can still [**perform elementwise operations
by invoking the *broadcasting mechanism*.**]
This mechanism works in the following way:
First, expand one or both arrays
by copying elements appropriately
so that after this transformation,
the two tensors have the same shape.
Second, carry out the elementwise operations
on the resulting arrays.
Na seção acima, vimos como realizar operações elementwise
em dois tensores da mesma forma. Sob certas condições,
mesmo quando as formas são diferentes, ainda podemos [** realizar operações elementar
invocando o * mecanismo de transmissão *. **]
Esse mecanismo funciona da seguinte maneira:
Primeiro, expanda um ou ambos os arrays
copiando elementos de forma adequada
de modo que após esta transformação,
os dois tensores têm a mesma forma.
Em segundo lugar, execute as operações elementwise
nas matrizes resultantes.

In most cases, we broadcast along an axis where an array
initially only has length 1, such as in the following example:


```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

Since `a` and `b` are $3\times1$ and $1\times2$ matrices respectively,
their shapes do not match up if we want to add them.
We *broadcast* the entries of both matrices into a larger $3\times2$ matrix as follows:
for matrix `a` it replicates the columns
and for matrix `b` it replicates the rows
before adding up both elementwise.


```{.python .input}
#@tab all
a + b
```

## Indexing and Slicing

Just as in any other Python array, elements in a tensor can be accessed by index.
As in any Python array, the first element has index 0
and ranges are specified to include the first but *before* the last element.
As in standard Python lists, we can access elements
according to their relative position to the end of the list
by using negative indices.

Thus, [**`[-1]` selects the last element and `[1:3]`
selects the second and the third elements**] as follows:


```{.python .input}
#@tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
Beyond reading, (**we can also write elements of a matrix by specifying indices.**)
:end_tab:

:begin_tab:`tensorflow`
`Tensors` in TensorFlow are immutable, and cannot be assigned to.
`Variables` in TensorFlow are mutable containers of state that support
assignments. Keep in mind that gradients in TensorFlow do not flow backwards
through `Variable` assignments.

Beyond assigning a value to the entire `Variable`, we can write elements of a
`Variable` by specifying indices.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X[1, 2] = 9
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```


If we want [**to assign multiple elements the same value,
we simply index all of them and then assign them the value.**]
For instance, `[0:2, :]` accesses the first and second rows,
where `:` takes all the elements along axis 1 (column).
While we discussed indexing for matrices,
this obviously also works for vectors
and for tensors of more than 2 dimensions.

```{.python .input}
#@tab mxnet, pytorch
X[0:2, :] = 12
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
X_var
```

## Saving Memory

[**Running operations can cause new memory to be
allocated to host results.**]
For example, if we write `Y = X + Y`,
we will dereference the tensor that `Y` used to point to
and instead point `Y` at the newly allocated memory.
In the following example, we demonstrate this with Python's `id()` function,
which gives us the exact address of the referenced object in memory.
After running `Y = Y + X`, we will find that `id(Y)` points to a different location.
That is because Python first evaluates `Y + X`,
allocating new memory for the result and then makes `Y`
point to this new location in memory.

```{.python .input}
#@tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

This might be undesirable for two reasons.
First, we do not want to run around
allocating memory unnecessarily all the time.
In machine learning, we might have
hundreds of megabytes of parameters
and update all of them multiple times per second.
Typically, we will want to perform these updates *in place*.
Second, we might point at the same parameters from multiple variables.
If we do not update in place, other references will still point to
the old memory location, making it possible for parts of our code
to inadvertently reference stale parameters.

:begin_tab:`mxnet, pytorch`
Fortunately, (**performing in-place operations**) is easy.
We can assign the result of an operation
to a previously allocated array with slice notation,
e.g., `Y[:] = <expression>`.
To illustrate this concept, we first create a new matrix `Z`
with the same shape as another `Y`,
using `zeros_like` to allocate a block of $0$ entries.
:end_tab:

:begin_tab:`tensorflow`
`Variables` are mutable containers of state in TensorFlow. They provide
a way to store your model parameters.
We can assign the result of an operation
to a `Variable` with `assign`.
To illustrate this concept, we create a `Variable` `Z`
with the same shape as another tensor `Y`,
using `zeros_like` to allocate a block of $0$ entries.
:end_tab:

```{.python .input}
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
[**If the value of `X` is not reused in subsequent computations,
we can also use `X[:] = X + Y` or `X += Y`
to reduce the memory overhead of the operation.**]
:end_tab:

:begin_tab:`tensorflow`
Even once you store state persistently in a `Variable`, you
may want to reduce your memory usage further by avoiding excess
allocations for tensors that are not your model parameters.

Because TensorFlow `Tensors` are immutable and gradients do not flow through
`Variable` assignments, TensorFlow does not provide an explicit way to run
an individual operation in-place.

However, TensorFlow provides the `tf.function` decorator to wrap computation
inside of a TensorFlow graph that gets compiled and optimized before running.
This allows TensorFlow to prune unused values, and to re-use
prior allocations that are no longer needed. This minimizes the memory
overhead of TensorFlow computations.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be re-used when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```


## Conversion to Other Python Objects

[**Converting to a NumPy tensor**], or vice versa, is easy.
The converted result does not share memory.
This minor inconvenience is actually quite important:
when you perform operations on the CPU or on GPUs,
you do not want to halt computation, waiting to see
whether the NumPy package of Python might want to be doing something else
with the same chunk of memory.


```{.python .input}
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
#@tab pytorch
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```

```{.python .input}
#@tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

To (**convert a size-1 tensor to a Python scalar**),
we can invoke the `item` function or Python's built-in functions.


```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## Summary

* The main interface to store and manipulate data for deep learning is the tensor ($n$-dimensional array). It provides a variety of functionalities including basic mathematics operations, broadcasting, indexing, slicing, memory saving, and conversion to other Python objects.


## Exercises

1. Run the code in this section. Change the conditional statement `X == Y` in this section to `X < Y` or `X > Y`, and then see what kind of tensor you can get.
1. Replace the two tensors that operate by element in the broadcasting mechanism with other shapes, e.g., 3-dimensional tensors. Is the result the same as expected?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE2OTI4NTU4LC0xNjk2MjgxNDE1LC0xMz
A0NzE1NDgwXX0=
-->
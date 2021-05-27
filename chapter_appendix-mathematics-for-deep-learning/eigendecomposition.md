# Autovalores e Autovetores
:label:`sec_eigendecompositions`


Os autovalores são frequentemente uma das noções mais úteis
encontraremos ao estudar álgebra linear,
entretanto, como um iniciante, é fácil ignorar sua importância.
Abaixo, apresentamos a decomposção e cálculo destes, e
tentamos transmitir algum sentido de por que é tão importante.

Suponha que tenhamos uma matriz $A$ com as seguintes entradas:

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$


Se aplicarmos $A$ a qualquer vetor $\mathbf{v} = [x, y]^\top$, 
obtemos um vetor $\mathbf{A}\mathbf{v} = [2x, -y]^\top$.
Isso tem uma interpretação intuitiva:
estique o vetor para ser duas vezes mais largo na direção $x$,
e, em seguida, inverta-o na direção $y$-direcionamento.

No entanto, existem *alguns* vetores para os quais algo permanece inalterado.
A saber, $[1, 0]^\top$ é enviado para $[2, 0]^\top$
e $[0, 1]^\top$ é enviado para $[0, -1]^\top$.
Esses vetores ainda estão na mesma linha,
e a única modificação é que a matriz os estica
por um fator de $2$ e $-1$ respectivamente.
Chamamos esses vetores de *autovetores*
e o fator eles são estendidos por *autovalores*.

Em geral, se pudermos encontrar um número $\lambda$ 
e um vetor $\mathbf{v}$ tal que

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

Dizemos que $\mathbf{v}$ é um autovetor para $A$ e $\lambda$ é um autovalor.

## Encontrando Autovalores
Vamos descobrir como encontrá-los. Subtraindo $\lambda \mathbf{v}$ de ambos os lados,
e então fatorar o vetor,
vemos que o acima é equivalente a:

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

Para :eqref:`eq_eigvalue_der` acontecer, vemos que $(\mathbf{A} - \lambda \mathbf{I})$ 
deve comprimir alguma direção até zero,
portanto, não é invertível e, portanto, o determinante é zero.
Assim, podemos encontrar os *valores próprios*
descobrindo quanto $\lambda$ is $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$.
Depois de encontrar os valores próprios, podemos resolver
$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ 
para encontrar os *autovetores* associados.

### Um Exemplo
Vamos ver isso com uma matriz mais desafiadora

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}.
$$

Se considerarmos $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$, 
vemos que isso é equivalente à equação polinomial
$0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$.
Assim, dois valores próprios são $4$ e $1$.
Para encontrar os vetores associados, precisamos resolver

$$
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \text{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$


Podemos resolver isso com os vetores $[1, -1]^\top$ and $[1, 2]^\top$ respectivamente.

Podemos verificar isso no código usando a rotina incorporada `numpy.linalg.eig`.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64),
          eigenvectors=True)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

Observe que `numpy` normaliza os vetores próprios para ter comprimento um,
ao passo que consideramos o nosso comprimento arbitrário.
Além disso, a escolha do sinal é arbitrária.
No entanto, os vetores calculados são paralelos
aos que encontramos à mão com os mesmos autovalores.

## Matrizes de Decomposição
Vamos continuar o exemplo anterior um passo adiante. Deixe

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

ser a matriz onde as colunas são os autovetores da matriz $\mathbf{A}$. Deixe

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

ser a matriz com os autovalores associados na diagonal.
Então, a definição de autovalores e autovetores nos diz que

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

A matriz $W$ é invertível, então podemos multiplicar ambos os lados por $W^{-1}$ à direita,
nós vemos que podemos escrever

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

Na próxima seção, veremos algumas consequências interessantes disso,
mas por agora só precisamos saber que tal decomposição
existirá enquanto pudermos encontrar uma coleção completa
de autovetores linearmente independentes (de forma que $W$ seja invertível).

## Operações em Autovalores e Autovetores
Uma coisa boa sobre autovalores e autovetores :eqref:`eq_eig_decomp` é que
podemos escrever muitas operações que geralmente encontramos de forma limpa
em termos da decomposição automática. Como primeiro exemplo, considere:

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\text{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\text{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\text{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

Isso nos diz que para qualquer poder positivo de uma matriz,
a autodecomposição é obtida apenas elevando os autovalores à mesma potência.
O mesmo pode ser mostrado para potências negativas,
então, se quisermos inverter uma matriz, precisamos apenas considerar

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$


ou em outras palavras, apenas inverta cada autovalor.
Isso funcionará, desde que cada autovalor seja diferente de zero,
portanto, vemos que invertível é o mesmo que não ter autovalores zero.

De fato, um trabalho adicional pode mostrar que se $\lambda_1, \ldots, \lambda_n$ 
são os autovalores de uma matriz, então o determinante dessa matriz é

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$


ou o produto de todos os autovalores.
Isso faz sentido intuitivamente porque qualquer coisa que esticar $\mathbf{W}$ faz,
$W^{-1}$ desfaz, então, no final, o único alongamento que acontece é
por multiplicação pela matriz diagonal $\boldsymbol{\Sigma}$, 
que estica os volumes pelo produto dos elementos diagonais.

Por fim, lembre-se de que a classificação era o número máximo
de colunas linearmente independentes de sua matriz.
Examinando a decomposição de perto,
podemos ver que a classificação é a mesma
que o número de autovalores diferentes de zero de $\mathbf{A}$.

Os exemplos podem continuar, mas espero que o ponto esteja claro:
A autodecomposição pode simplificar muitos cálculos algébricos lineares
e é uma operação fundamental subjacente a muitos algoritmos numéricos
e muitas das análises que fazemos em álgebra linear.

## Composições Originais de Matrizes Simétricas
Nem sempre é possível encontrar autovetores independentes linearmente suficientes
para que o processo acima funcione. Por exemplo, a matriz

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$


tem apenas um único autovetor, a saber $(1, 0)^\top$. 
Para lidar com essas matrizes, exigimos técnicas mais avançadas
do que podemos cobrir (como a Forma Normal de Jordan ou Decomposição de Valor Singular).
Frequentemente precisaremos restringir nossa atenção a essas matrizes
onde podemos garantir a existência de um conjunto completo de autovetores.

A família mais comumente encontrada são as *matrizes simétricas*,
que são aquelas matrizes onde $\mathbf{A} = \mathbf{A}^\top$. 

Neste caso, podemos tomar $W$ como uma *matriz ortogonal* - uma matriz cujas colunas são todas vetores de comprimento unitário que estão em ângulos retos entre si, onde
$\mathbf{W}^\top = \mathbf{W}^{-1}$ -- e todos os autovalores serão reais.
Assim, neste caso especial, podemos escrever :eqref:`eq_eig_decomp` como

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## Teorema do Círculo de Gershgorin

Os valores próprios costumam ser difíceis de raciocinar intuitivamente.
Se for apresentada uma matriz arbitrária, pouco pode ser dito
sobre quais são os valores próprios sem computá-los.
Há, no entanto, um teorema que pode facilitar uma boa aproximação
se os maiores valores estiverem na diagonal.

Seja $\mathbf{A} = (a_{ij})$ qualquer matriz quadrada($n\times n$).
Definiremos $r_i = \sum_{j \neq i} |a_{ij}|$.
Deixe $\mathcal{D}_i$ representar o disco no plano complexo
com centro $a_{ii}$ radius $r_i$.
Então, cada autovalor de $\mathbf{A}$ está contido em um dos $\mathcal{D}_i$.

Isso pode ser um pouco difícil de descompactar, então vejamos um exemplo.
Considere a matriz:

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

Temos $r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.8$ e $r_4 = 0.9$.
A matriz é simétrica, portanto, todos os autovalores são reais.
Isso significa que todos os nossos valores próprios estarão em um dos intervalos de

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$


Performing the numerical computation shows 
that the eigenvalues are approximately $0.99$, $2.97$, $4.95$, $9.08$,
all comfortably inside the ranges provided.

```{.python .input}
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

In this way, eigenvalues can be approximated, 
and the approximations will be fairly accurate 
in the case that the diagonal is 
significantly larger than all the other elements.  

It is a small thing, but with a complex 
and subtle topic like eigendecomposition, 
it is good to get any intuitive grasp we can.

## A Useful Application: The Growth of Iterated Maps

Now that we understand what eigenvectors are in principle,
let us see how they can be used to provide a deep understanding 
of a problem central to neural network behavior: proper weight initialization. 

### Eigenvectors as Long Term Behavior

The full mathematical investigation of the initialization 
of deep neural networks is beyond the scope of the text, 
but we can see a toy version here to understand
how eigenvalues can help us see how these models work.
As we know, neural networks operate by interspersing layers 
of linear transformations with non-linear operations.
For simplicity here, we will assume that there is no non-linearity,
and that the transformation is a single repeated matrix operation $A$,
so that the output of our model is

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

When these models are initialized, $A$ is taken to be 
a random matrix with Gaussian entries, so let us make one of those. 
To be concrete, we start with a mean zero, variance one Gaussian distributed $5 \times 5$ matrix.

```{.python .input}
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### Behavior on Random Data
For simplicity in our toy model, 
we will assume that the data vector we feed in $\mathbf{v}_{in}$ 
is a random five dimensional Gaussian vector.
Let us think about what we want to have happen.
For context, lets think of a generic ML problem,
where we are trying to turn input data, like an image, into a prediction, 
like the probability the image is a picture of a cat.
If repeated application of $\mathbf{A}$ 
stretches a random vector out to be very long, 
then small changes in input will be amplified 
into large changes in output---tiny modifications of the input image
would lead to vastly different predictions.
This does not seem right!

On the flip side, if $\mathbf{A}$ shrinks random vectors to be shorter,
then after running through many layers, the vector will essentially shrink to nothing, 
and the output will not depend on the input. This is also clearly not right either!

We need to walk the narrow line between growth and decay 
to make sure that our output changes depending on our input, but not much!

Let us see what happens when we repeatedly multiply our matrix $\mathbf{A}$ 
against a random input vector, and keep track of the norm.

```{.python .input}
# Calculate the sequence of norms after repeatedly applying `A`
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Calculate the sequence of norms after repeatedly applying `A`
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Calculate the sequence of norms after repeatedly applying `A`
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

The norm is growing uncontrollably! 
Indeed if we take the list of quotients, we will see a pattern.

```{.python .input}
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

If we look at the last portion of the above computation, 
we see that the random vector is stretched by a factor of `1.974459321485[...]`,
where the portion at the end shifts a little, 
but the stretching factor is stable.  

### Relating Back to Eigenvectors

We have seen that eigenvectors and eigenvalues correspond 
to the amount something is stretched, 
but that was for specific vectors, and specific stretches.
Let us take a look at what they are for $\mathbf{A}$.
A bit of a caveat here: it turns out that to see them all,
we will need to go to complex numbers.
You can think of these as stretches and rotations.
By taking the norm of the complex number
(square root of the sums of squares of real and imaginary parts)
we can measure that stretching factor. Let us also sort them.

```{.python .input}
# Compute the eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# Compute the eigenvalues
eigs = torch.eig(A)[0][:,0].tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# Compute the eigenvalues
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### An Observation

We see something a bit unexpected happening here: 
that number we identified before for the 
long term stretching of our matrix $\mathbf{A}$ 
applied to a random vector is *exactly* 
(accurate to thirteen decimal places!) 
the largest eigenvalue of $\mathbf{A}$.
This is clearly not a coincidence!

But, if we now think about what is happening geometrically,
this starts to make sense. Consider a random vector. 
This random vector points a little in every direction, 
so in particular, it points at least a little bit 
in the same direction as the eigenvector of $\mathbf{A}$
associated with the largest eigenvalue.
This is so important that it is called 
the *principle eigenvalue* and *principle eigenvector*.
After applying $\mathbf{A}$, our random vector 
gets stretched in every possible direction,
as is associated with every possible eigenvector,
but it is stretched most of all in the direction 
associated with this principle eigenvector.
What this means is that after apply in $A$, 
our random vector is longer, and points in a direction 
closer to being aligned with the principle eigenvector.
After applying the matrix many times, 
the alignment with the principle eigenvector becomes closer and closer until, 
for all practical purposes, our random vector has been transformed 
into the principle eigenvector!
Indeed this algorithm is the basis 
for what is known as the *power iteration*
for finding the largest eigenvalue and eigenvector of a matrix. For details see, for example, :cite:`Van-Loan.Golub.1983`.

### Fixing the Normalization

Now, from above discussions, we concluded 
that we do not want a random vector to be stretched or squished at all,
we would like random vectors to stay about the same size throughout the entire process.
To do so, we now rescale our matrix by this principle eigenvalue 
so that the largest eigenvalue is instead now just one.
Let us see what happens in this case.

```{.python .input}
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

We can also plot the ratio between consecutive norms as before and see that indeed it stabilizes.

```{.python .input}
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## Conclusions

We now see exactly what we hoped for!
After normalizing the matrices by the principle eigenvalue,
we see that the random data does not explode as before,
but rather eventually equilibrates to a specific value.
It would be nice to be able to do these things from first principles,
and it turns out that if we look deeply at the mathematics of it,
we can see that the largest eigenvalue 
of a large random matrix with independent mean zero,
variance one Gaussian entries is on average about $\sqrt{n}$,
or in our case $\sqrt{5} \approx 2.2$,
due to a fascinating fact known as the *circular law* :cite:`Ginibre.1965`.
The relationship between the eigenvalues (and a related object called singular values) of random matrices has been shown to have deep connections to proper initialization of neural networks as was discussed in :cite:`Pennington.Schoenholz.Ganguli.2017` and subsequent works.

## Summary
* Eigenvectors are vectors which are stretched by a matrix without changing direction.
* Eigenvalues are the amount that the eigenvectors are stretched by the application of the matrix.
* The eigendecomposition of a matrix can allow for many operations to be reduced to operations on the eigenvalues.
* The Gershgorin Circle Theorem can provide approximate values for the eigenvalues of a matrix.
* The behavior of iterated matrix powers depends primarily on the size of the largest eigenvalue.  This understanding has many applications in the theory of neural network initialization.

## Exercises
1. What are the eigenvalues and eigenvectors of
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1.  What are the eigenvalues and eigenvectors of the following matrix, and what is strange about this example compared to the previous one?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. Without computing the eigenvalues, is it possible that the smallest eigenvalue of the following matrix is less that $0.5$? *Note*: this problem can be done in your head.
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1086)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1087)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4NzA4NDg1NzUsMTU2ODM0MDE4NCwtMT
E0ODUzOTM5OSwxNDM0NTYwMTY0LDY5MTE5ODU2OSw0ODM4MTEz
MDYsMjYwNjIxODU3LC0xNjA2NDkxMzg4LC0xNzI5MDE0ODY0XX
0=
-->
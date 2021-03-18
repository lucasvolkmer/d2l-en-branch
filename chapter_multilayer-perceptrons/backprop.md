# Propagação Direta, Propagação Reversa e Gráficos Computacionais
:label:`sec_backprop`


Até agora, treinamos nossos modelos
com gradiente descendente estocástico de *minibatch*.
No entanto, quando implementamos o algoritmo,
nós apenas nos preocupamos com os cálculos envolvidos
em *propagação direta* através do modelo.
Quando chegou a hora de calcular os gradientes,
acabamos de invocar a função *backpropagation* (propagação reversa) fornecida pela estrutura de *deep learning*.

O cálculo automático de gradientes (diferenciação automática) simplifica profundamente
a implementação de algoritmos de *deep learning*.
Antes da diferenciação automática,
mesmo pequenas mudanças em modelos complicados exigiam
recalcular derivadas complicadas manualmente.
Surpreendentemente, muitas vezes, os trabalhos acadêmicos tiveram que alocar
várias páginas para derivar regras de atualização.
Embora devamos continuar a confiar na diferenciação automática
para que possamos nos concentrar nas partes interessantes,
você deve saber como esses gradientes
são calculados sob o capô
se você quiser ir além de uma rasa
compreensão da aprendizagem profunda.

Nesta seção, fazemos um mergulho profundo
nos detalhes de *propagação para trás*
(mais comumente chamado de *backpropagation*).
Para transmitir alguns *insights* para ambas as
técnicas e suas implementações,
contamos com alguma matemática básica e gráficos computacionais.
Para começar, focamos nossa exposição em
um MLP de uma camada oculta
com *weight decay* (regularização$L_2$).

## Propagação Direta


*Propagação direta* (ou *passagem direta*) refere-se ao cálculo e armazenamento
de variáveis intermediárias (incluindo saídas)
para uma rede neural em ordem
da camada de entrada para a camada de saída.
Agora trabalhamos passo a passo com a mecânica
de uma rede neural com uma camada oculta.
Isso pode parecer tedioso, mas nas palavras eternas
do virtuoso do funk James Brown,
você deve "pagar o custo para ser o chefe".


Por uma questão de simplicidade, vamos assumir
que o exemplo de entrada é  $\mathbf{x}\in \mathbb{R}^d$
e que nossa camada oculta não inclui um termo de *bias*.
Aqui, a variável intermediária é:

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

onde $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
é o parâmetro de peso da camada oculta.
Depois de executar a variável intermediária
$\mathbf{z}\in \mathbb{R}^h$ através da
função de ativação $\phi$
obtemos nosso vetor de ativação oculto de comprimento $h$,

$$\mathbf{h}= \phi (\mathbf{z}).$$

A variável oculta $\mathbf{h}$
também é uma variável intermediária.
Supondo que os parâmetros da camada de saída
só possuem um peso de
$\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$,
podemos obter uma variável de camada de saída
com um vetor de comprimento $q$:

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

Supondo que a função de perda seja $l$
e o *label* de exemplo é $y$,
podemos então calcular o prazo de perda
para um único exemplo de dados,

$$L = l(\mathbf{o}, y).$$

De acordo com a definição de regularização de $L_2$
dado o hiperparâmetro $\lambda$,
o prazo de regularização é

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$
:eqlabel:`eq_forward-s`

onde a norma Frobenius da matriz
é simplesmente a norma $L_2$ aplicada
depois de achatar a matriz em um vetor.
Por fim, a perda regularizada do modelo
em um dado exemplo de dados é:

$$J = L + s.$$

Referimo-nos a $J$ como a *função objetivo*
na discussão a seguir.


## Computational Graph of Forward Propagation

Plotting *computational graphs* helps us visualize
the dependencies of operators
and variables within the calculation.
:numref:`fig_forward` contains the graph associated
with the simple network described above,
where squares denote variables and circles denote operators.
The lower-left corner signifies the input
and the upper-right corner is the output.
Notice that the directions of the arrows
(which illustrate data flow)
are primarily rightward and upward.

![Computational graph of forward propagation.](../img/forward.svg)
:label:`fig_forward`

## Backpropagation

*Backpropagation* refers to the method of calculating
the gradient of neural network parameters.
In short, the method traverses the network in reverse order,
from the output to the input layer,
according to the *chain rule* from calculus.
The algorithm stores any intermediate variables
(partial derivatives)
required while calculating the gradient
with respect to some parameters.
Assume that we have functions
$\mathsf{Y}=f(\mathsf{X})$
and $\mathsf{Z}=g(\mathsf{Y})$,
in which the input and the output
$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$
are tensors of arbitrary shapes.
By using the chain rule,
we can compute the derivative
of $\mathsf{Z}$ with respect to $\mathsf{X}$ via

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

Here we use the $\text{prod}$ operator
to multiply its arguments
after the necessary operations,
such as transposition and swapping input positions,
have been carried out.
For vectors, this is straightforward:
it is simply matrix-matrix multiplication.
For higher dimensional tensors,
we use the appropriate counterpart.
The operator $\text{prod}$ hides all the notation overhead.

Recall that
the parameters of the simple network with one hidden layer,
whose computational graph is in :numref:`fig_forward`,
are $\mathbf{W}^{(1)}$ and $\mathbf{W}^{(2)}$.
The objective of backpropagation is to
calculate the gradients $\partial J/\partial \mathbf{W}^{(1)}$
and $\partial J/\partial \mathbf{W}^{(2)}$.
To accomplish this, we apply the chain rule
and calculate, in turn, the gradient of
each intermediate variable and parameter.
The order of calculations are reversed
relative to those performed in forward propagation,
since we need to start with the outcome of the computational graph
and work our way towards the parameters.
The first step is to calculate the gradients
of the objective function $J=L+s$
with respect to the loss term $L$
and the regularization term $s$.

$$\frac{\partial J}{\partial L} = 1 \; \text{and} \; \frac{\partial J}{\partial s} = 1.$$

Next, we compute the gradient of the objective function
with respect to variable of the output layer $\mathbf{o}$
according to the chain rule:

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

Next, we calculate the gradients
of the regularization term
with respect to both parameters:

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

Now we are able to calculate the gradient
$\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$
of the model parameters closest to the output layer.
Using the chain rule yields:

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

To obtain the gradient with respect to $\mathbf{W}^{(1)}$
we need to continue backpropagation
along the output layer to the hidden layer.
The gradient with respect to the hidden layer's outputs
$\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ is given by


$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

Since the activation function $\phi$ applies elementwise,
calculating the gradient $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$
of the intermediate variable $\mathbf{z}$
requires that we use the elementwise multiplication operator,
which we denote by $\odot$:

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

Finally, we can obtain the gradient
$\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
of the model parameters closest to the input layer.
According to the chain rule, we get

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$



## Training Neural Networks

When training neural networks,
forward and backward propagation depend on each other.
In particular, for forward propagation,
we traverse the computational graph in the direction of dependencies
and compute all the variables on its path.
These are then used for backpropagation
where the compute order on the graph is reversed.

Take the aforementioned simple network as an example to illustrate.
On one hand,
computing the regularization term :eqref:`eq_forward-s`
during forward propagation
depends on the current values of model parameters $\mathbf{W}^{(1)}$ and $\mathbf{W}^{(2)}$.
They are given by the optimization algorithm according to backpropagation in the latest iteration.
On the other hand,
the gradient calculation for the parameter
:eqref:`eq_backprop-J-h` during backpropagation
depends on the current value of the hidden variable $\mathbf{h}$,
which is given by forward propagation.


Therefore when training neural networks, after model parameters are initialized,
we alternate forward propagation with backpropagation,
updating model parameters using gradients given by backpropagation.
Note that backpropagation reuses the stored intermediate values from forward propagation to avoid duplicate calculations.
One of the consequences is that we need to retain
the intermediate values until backpropagation is complete.
This is also one of the reasons why training
requires significantly more memory than plain prediction.
Besides, the size of such intermediate values is roughly
proportional to the number of network layers and the batch size.
Thus,
training deeper networks using larger batch sizes
more easily leads to *out of memory* errors.


## Summary

* Forward propagation sequentially calculates and stores intermediate variables within the computational graph defined by the neural network. It proceeds from the input to the output layer.
* Backpropagation sequentially calculates and stores the gradients of intermediate variables and parameters within the neural network in the reversed order.
* When training deep learning models, forward propagation and back propagation are interdependent.
* Training requires significantly more memory than prediction.


## Exercises

1. Assume that the inputs $\mathbf{X}$ to some scalar function $f$ are $n \times m$ matrices. What is the dimensionality of the gradient of $f$ with respect to $\mathbf{X}$?
1. Add a bias to the hidden layer of the model described in this section (you do not need to include bias in the regularization term).
    1. Draw the corresponding computational graph.
    1. Derive the forward and backward propagation equations.
1. Compute the memory footprint for training and prediction in the model described in this section.
1. Assume that you want to compute second derivatives. What happens to the computational graph? How long do you expect the calculation to take?
1. Assume that the computational graph is too large for your GPU.
    1. Can you partition it over more than one GPU?
    1. What are the advantages and disadvantages over training on a smaller minibatch?

[Discussions](https://discuss.d2l.ai/t/102)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5MDY0NTc4MjYsODU3NDY5ODM2LDEwMT
ExNDMzMzcsMTM1NzcyMDQyOV19
-->
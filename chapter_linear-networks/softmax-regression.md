# Regressão *Softmax*
:label:`sec_softmax`


Em :numref:`sec_linear_regression`, introduzimos a regressão linear,
trabalhando através de implementações do zero em :numref:`sec_linear_scratch`
e novamente usando APIs de alto nível de uma estrutura de *deep learning*
em :numref:`sec_linear_concise` para fazer o trabalho pesado.

A regressão é o martelo que procuramos quando
queremos responder a perguntas*quanto?* ou *quantas?*.
Se você deseja prever o número de dólares (preço)
a que uma casa será vendida,
ou o número de vitórias que um time de beisebol pode ter,
ou o número de dias que um paciente permanecerá hospitalizado antes de receber alta,
então provavelmente você está procurando um modelo de regressão.

Na prática, estamos mais frequentemente interessados ​​na *classificação*:
perguntando não "quanto", mas "qual":

* Este e-mail pertence à pasta de spam ou à caixa de entrada?
* É mais provável que este cliente *se inscreva* ou *não se inscreva* em um serviço de assinatura?
* Esta imagem retrata um burro, um cachorro, um gato ou um galo?
* Qual filme Aston tem mais probabilidade de assistir a seguir?

Coloquialmente, praticantes de *machine learning*
sobrecarregam a palavra *classificação*
para descrever dois problemas sutilmente diferentes:
(i) aqueles em que estamos interessados ​​apenas em
atribuições difíceis de exemplos a categorias (classes);
e (ii) aqueles em que desejamos fazer atribuições leves,
ou seja, para avaliar a probabilidade de que cada categoria se aplica.
A distinção tende a ficar confusa, em parte,
porque muitas vezes, mesmo quando nos preocupamos apenas com tarefas difíceis,
ainda usamos modelos que fazem atribuições suaves.


## Problema de Classificação
:label:`subsec_classification-problem`


Para molhar nossos pés, vamos começar com
um problema simples de classificação de imagens.
Aqui, cada entrada consiste em uma imagem em tons de cinza $2\times 2$.
Podemos representar cada valor de pixel com um único escalar,
dando-nos quatro características $x_1, x_2, x_3, x_4$.
Além disso, vamos supor que cada imagem pertence a uma
entre as categorias "gato", "frango" e "cachorro".

A seguir, temos que escolher como representar os *labels*.
Temos duas escolhas óbvias.
Talvez o impulso mais natural seja escolher $y\in \{1, 2, 3 \}$,
onde os inteiros representam $\{\text{cachorro}, \text{gato}, \text{frango}\}$ respectivamente.
Esta é uma ótima maneira de *armazenar* essas informações em um computador.
Se as categorias tivessem alguma ordem natural entre elas,
digamos se estivéssemos tentando prever $\{\text {bebê}, \text{criança}, \text{adolescente}, \text{jovem adulto}, \text{adulto}, \text{idoso}\}$,
então pode até fazer sentido lançar este problema como uma regressão
e manter os rótulos neste formato.

Mas os problemas gerais de classificação não vêm com ordenações naturais entre as classes.
Felizmente, os estatísticos há muito tempo inventaram uma maneira simples
para representar dados categóricos: a *codificação one-hot*.
Uma codificação *one-hot* é um vetor com tantos componentes quantas categorias temos.
O componente correspondente à categoria da instância em particular é definido como 1
e todos os outros componentes são definidos como 0.
Em nosso caso, um rótulo $y$ seria um vetor tridimensional,
com $(1, 0, 0)$ correspondendo a "gato", $(0, 1, 0)$ a "galinha",
e $(0, 0, 1)$ para "cachorro":

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

## Arquitetura de Rede

A fim de estimar as probabilidades condicionais associadas a todas as classes possíveis,
precisamos de um modelo com várias saídas, uma por classe.
Para abordar a classificação com modelos lineares,
precisaremos de tantas funções afins quantas forem as saídas.
Cada saída corresponderá a sua própria função afim.
No nosso caso, uma vez que temos 4 *features* e 3 categorias de saída possíveis,
precisaremos de 12 escalares para representar os pesos ($w$ com subscritos),
e 3 escalares para representar os *offsets* ($b$ com subscritos).
Calculamos esses três *logits*, $o_1, o_2$, and $o_3$, para cada entrada:

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

Podemos representar esse cálculo com o diagrama da rede neural mostrado em :numref:`fig_softmaxreg`.
Assim como na regressão linear, a regressão *softmax* também é uma rede neural de camada única.
E desde o cálculo de cada saída, $o_1, o_2$, e $o_3$,
depende de todas as entradas, $x_1$, $x_2$, $x_3$, e $x_4$,
a camada de saída da regressão *softmax* também pode ser descrita como uma camada totalmente conectada.

![Softmax regression is a single-layer neural network.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

To express the model more compactly, we can use linear algebra notation.
In vector form, we arrive at
$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$,
a form better suited both for mathematics, and for writing code.
Note that we have gathered all of our weights into a $3 \times 4$ matrix
and that for features of a given data example $\mathbf{x}$,
our outputs are given by a matrix-vector product of our weights by our input features
plus our biases $\mathbf{b}$.


## Parameterization Cost of Fully-Connected Layers
:label:`subsec_parameterization-cost-fc-layers`

As we will see in subsequent chapters,
fully-connected layers are ubiquitous in deep learning.
However, as the name suggests,
fully-connected layers are *fully* connected
with potentially many learnable parameters.
Specifically,
for any fully-connected layer
with $d$ inputs and $q$ outputs,
the parameterization cost is $\mathcal{O}(dq)$,
which can be prohibitively high in practice.
Fortunately,
this cost 
of transforming $d$ inputs into $q$ outputs
can be reduced to $\mathcal{O}(\frac{dq}{n})$,
where the hyperparameter $n$ can be flexibly specified
by us to balance between parameter saving and model effectiveness in real-world applications :cite:`Zhang.Tay.Zhang.ea.2021`.





## Softmax Operation
:label:`subsec_softmax_operation`

The main approach that we are going to take here
is to interpret the outputs of our model as probabilities.
We will optimize our parameters to produce probabilities
that maximize the likelihood of the observed data.
Then, to generate predictions, we will set a threshold,
for example, choosing the label with the maximum predicted probabilities.

Put formally, we would like any output $\hat{y}_j$
to be interpreted as the probability
that a given item belongs to class $j$.
Then we can choose the class with the largest output value
as our prediction $\operatorname*{argmax}_j y_j$.
For example, if $\hat{y}_1$, $\hat{y}_2$, and $\hat{y}_3$
are 0.1, 0.8, and 0.1, respectively,
then we predict category 2, which (in our example) represents "chicken".

You might be tempted to suggest that we interpret
the logits $o$ directly as our outputs of interest.
However, there are some problems with directly
interpreting the output of the linear layer as a probability.
On one hand,
nothing constrains these numbers to sum to 1.
On the other hand, depending on the inputs, they can take negative values.
These violate basic axioms of probability presented in :numref:`sec_prob`

To interpret our outputs as probabilities,
we must guarantee that (even on new data),
they will be nonnegative and sum up to 1.
Moreover, we need a training objective that encourages
the model to estimate faithfully probabilities.
Of all instances when a classifier outputs 0.5,
we hope that half of those examples
will actually belong to the predicted class.
This is a property called *calibration*.

The *softmax function*, invented in 1959 by the social scientist
R. Duncan Luce in the context of *choice models*,
does precisely this.
To transform our logits such that they become nonnegative and sum to 1,
while requiring that the model remains differentiable,
we first exponentiate each logit (ensuring non-negativity)
and then divide by their sum (ensuring that they sum to 1):

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}. $$
:eqlabel:`eq_softmax_y_and_o`

It is easy to see $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$
with $0 \leq \hat{y}_j \leq 1$ for all $j$.
Thus, $\hat{\mathbf{y}}$ is a proper probability distribution
whose element values can be interpreted accordingly.
Note that the softmax operation does not change the ordering among the logits $\mathbf{o}$,
which are simply the pre-softmax values
that determine the probabilities assigned to each class.
Therefore, during prediction we can still pick out the most likely class by

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$

Although softmax is a nonlinear function,
the outputs of softmax regression are still *determined* by
an affine transformation of input features;
thus, softmax regression is a linear model.



## Vectorization for Minibatches
:label:`subsec_softmax_vectorization`

To improve computational efficiency and take advantage of GPUs,
we typically carry out vector calculations for minibatches of data.
Assume that we are given a minibatch $\mathbf{X}$ of examples
with feature dimensionality (number of inputs) $d$ and batch size $n$.
Moreover, assume that we have $q$ categories in the output.
Then the minibatch features $\mathbf{X}$ are in $\mathbb{R}^{n \times d}$,
weights $\mathbf{W} \in \mathbb{R}^{d \times q}$,
and the bias satisfies $\mathbf{b} \in \mathbb{R}^{1\times q}$.

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_minibatch_softmax_reg`

This accelerates the dominant operation into
a matrix-matrix product $\mathbf{X} \mathbf{W}$
vs. the matrix-vector products we would be executing
if we processed one example at a time.
Since each row in $\mathbf{X}$ represents a data example,
the softmax operation itself can be computed *rowwise*:
for each row of $\mathbf{O}$, exponentiate all entries and then normalize them by the sum.
Triggering broadcasting during the summation $\mathbf{X} \mathbf{W} + \mathbf{b}$ in :eqref:`eq_minibatch_softmax_reg`,
both the minibatch logits $\mathbf{O}$ and output probabilities $\hat{\mathbf{Y}}$
are $n \times q$ matrices.

## Loss Function

Next, we need a loss function to measure
the quality of our predicted probabilities.
We will rely on maximum likelihood estimation,
the very same concept that we encountered
when providing a probabilistic justification
for the mean squared error objective in linear regression
(:numref:`subsec_normal_distribution_and_squared_loss`).


### Log-Likelihood

The softmax function gives us a vector $\hat{\mathbf{y}}$,
which we can interpret as estimated conditional probabilities
of each class given any input $\mathbf{x}$, e.g.,
$\hat{y}_1$ = $P(y=\text{cat} \mid \mathbf{x})$.
Suppose that the entire dataset $\{\mathbf{X}, \mathbf{Y}\}$ has $n$ examples,
where the example indexed by $i$
consists of a feature vector $\mathbf{x}^{(i)}$ and a one-hot label vector $\mathbf{y}^{(i)}$.
We can compare the estimates with reality
by checking how probable the actual classes are
according to our model, given the features:

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$

According to maximum likelihood estimation,
we maximize $P(\mathbf{Y} \mid \mathbf{X})$,
which is
equivalent to minimizing the negative log-likelihood:

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

where for any pair of label $\mathbf{y}$ and model prediction $\hat{\mathbf{y}}$ over $q$ classes,
the loss function $l$ is

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
:eqlabel:`eq_l_cross_entropy`

For reasons explained later on, the loss function in :eqref:`eq_l_cross_entropy`
is commonly called the *cross-entropy loss*.
Since $\mathbf{y}$ is a one-hot vector of length $q$,
the sum over all its coordinates $j$ vanishes for all but one term.
Since all $\hat{y}_j$ are predicted probabilities,
their logarithm is never larger than $0$.
Consequently, the loss function cannot be minimized any further
if we correctly predict the actual label with *certainty*,
i.e., if the predicted probability $P(\mathbf{y} \mid \mathbf{x}) = 1$ for the actual label $\mathbf{y}$.
Note that this is often impossible.
For example, there might be label noise in the dataset
(some examples may be mislabeled).
It may also not be possible when the input features
are not sufficiently informative
to classify every example perfectly.

### Softmax and Derivatives
:label:`subsec_softmax_and_derivatives`

Since the softmax and the corresponding loss are so common,
it is worth understanding a bit better how it is computed.
Plugging :eqref:`eq_softmax_y_and_o` into the definition of the loss
in :eqref:`eq_l_cross_entropy`
and using the definition of the softmax we obtain:

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

To understand a bit better what is going on,
consider the derivative with respect to any logit $o_j$. We get

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

In other words, the derivative is the difference
between the probability assigned by our model,
as expressed by the softmax operation,
and what actually happened, as expressed by elements in the one-hot label vector.
In this sense, it is very similar to what we saw in regression,
where the gradient was the difference
between the observation $y$ and estimate $\hat{y}$.
This is not coincidence.
In any exponential family (see the
[online appendix on distributions](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/distributions.html)) model,
the gradients of the log-likelihood are given by precisely this term.
This fact makes computing gradients easy in practice.

### Cross-Entropy Loss

Now consider the case where we observe not just a single outcome
but an entire distribution over outcomes.
We can use the same representation as before for the label $\mathbf{y}$.
The only difference is that rather than a vector containing only binary entries,
say $(0, 0, 1)$, we now have a generic probability vector, say $(0.1, 0.2, 0.7)$.
The math that we used previously to define the loss $l$
in :eqref:`eq_l_cross_entropy`
still works out fine,
just that the interpretation is slightly more general.
It is the expected value of the loss for a distribution over labels.
This loss is called the *cross-entropy loss* and it is
one of the most commonly used losses for classification problems.
We can demystify the name by introducing just the basics of information theory.
If you wish to understand more details of information theory,
you may further refer to the [online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html).



## Information Theory Basics
:label:`subsec_info_theory_basics`

*Information theory* deals with the problem of encoding, decoding, transmitting,
and manipulating information (also known as data) in as concise form as possible.


### Entropy

The central idea in information theory is to quantify the information content in data.
This quantity places a hard limit on our ability to compress the data.
In information theory, this quantity is called the *entropy* of a distribution $P$,
and it is captured by the following equation:

$$H[P] = \sum_j - P(j) \log P(j).$$
:eqlabel:`eq_softmax_reg_entropy`

One of the fundamental theorems of information theory states
that in order to encode data drawn randomly from the distribution $P$,
we need at least $H[P]$ "nats" to encode it.
If you wonder what a "nat" is, it is the equivalent of bit
but when using a code with base $e$ rather than one with base 2.
Thus, one nat is $\frac{1}{\log(2)} \approx 1.44$ bit.


### Surprisal

You might be wondering what compression has to do with prediction.
Imagine that we have a stream of data that we want to compress.
If it is always easy for us to predict the next token,
then this data is easy to compress!
Take the extreme example where every token in the stream always takes the same value.
That is a very boring data stream!
And not only it is boring, but it is also easy to predict.
Because they are always the same, we do not have to transmit any information
to communicate the contents of the stream.
Easy to predict, easy to compress.

However if we cannot perfectly predict every event,
then we might sometimes be surprised.
Our surprise is greater when we assigned an event lower probability.
Claude Shannon settled on $\log \frac{1}{P(j)} = -\log P(j)$
to quantify one's *surprisal* at observing an event $j$
having assigned it a (subjective) probability $P(j)$.
The entropy defined in :eqref:`eq_softmax_reg_entropy` is then the *expected surprisal*
when one assigned the correct probabilities
that truly match the data-generating process.


### Cross-Entropy Revisited

So if entropy is level of surprise experienced
by someone who knows the true probability,
then you might be wondering, what is cross-entropy?
The cross-entropy *from* $P$ *to* $Q$, denoted $H(P, Q)$,
is the expected surprisal of an observer with subjective probabilities $Q$
upon seeing data that were actually generated according to probabilities $P$.
The lowest possible cross-entropy is achieved when $P=Q$.
In this case, the cross-entropy from $P$ to $Q$ is $H(P, P)= H(P)$.

In short, we can think of the cross-entropy classification objective
in two ways: (i) as maximizing the likelihood of the observed data;
and (ii) as minimizing our surprisal (and thus the number of bits)
required to communicate the labels.


## Model Prediction and Evaluation

After training the softmax regression model, given any example features,
we can predict the probability of each output class.
Normally, we use the class with the highest predicted probability as the output class.
The prediction is correct if it is consistent with the actual class (label).
In the next part of the experiment,
we will use *accuracy* to evaluate the model's performance.
This is equal to the ratio between the number of correct predictions and the total number of predictions.


## Summary

* The softmax operation takes a vector and maps it into probabilities.
* Softmax regression applies to classification problems. It uses the probability distribution of the output class in the softmax operation.
* Cross-entropy is a good measure of the difference between two probability distributions. It measures the number of bits needed to encode the data given our model.

## Exercises

1. We can explore the connection between exponential families and the softmax in some more depth.
    1. Compute the second derivative of the cross-entropy loss $l(\mathbf{y},\hat{\mathbf{y}})$ for the softmax.
    1. Compute the variance of the distribution given by $\mathrm{softmax}(\mathbf{o})$ and show that it matches the second derivative computed above.
1. Assume that we have three classes which occur with equal probability, i.e., the probability vector is $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    1. What is the problem if we try to design a binary code for it?
    1. Can you design a better code? Hint: what happens if we try to encode two independent observations? What if we encode $n$ observations jointly?
1. Softmax is a misnomer for the mapping introduced above (but everyone in deep learning uses it). The real softmax is defined as $\mathrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
    1. Prove that $\mathrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$.
    1. Prove that this holds for $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$, provided that $\lambda > 0$.
    1. Show that for $\lambda \to \infty$ we have $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$.
    1. What does the soft-min look like?
    1. Extend this to more than two numbers.

[Discussions](https://discuss.d2l.ai/t/46)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTg1NDM0ODQ0MSwtMTI5MTE0NzU5NiwxMD
Y2OTU2MDkzLDE4NDMzNzUwOTQsMTM5NDcyMzQ2MF19
-->
# BackRetropropagation Through Timeção ao Longo do Tempo
:label:`sec_bptt`


Até agora, temos repetidamente aludido a coisas como
*gradientes explosivos*,
*gradientes de desaparecimento*,
e a necessidade de
*destacar o gradiente* para RNNs.
Por exemplo, em :numref:`sec_rnn_scratch`
invocamos a função `detach` na sequência.
Nada disso foi realmente completamente
explicado, no interesse de ser capaz de construir um modelo rapidamente e
para ver como funciona.
Nesta secção,
vamos nos aprofundar um pouco mais
nos detalhes de retropropagação para modelos de sequência e por que (e como) a matemática funciona.

Encontramos alguns dos efeitos da explosão de gradiente quando primeiro
RNNs implementados (:numref:`sec_rnn_scratch`).
No
especial,
se você resolveu os exercícios,
você poderia
ter visto que o corte de gradiente é vital para garantir
convergência.
Para fornecer uma melhor compreensão deste problema, esta
seção irá rever como os gradientes são calculados para modelos de sequência.
Observe
que não há nada conceitualmente novo em como funciona. Afinal, ainda estamos apenas aplicando a regra da cadeia para calcular gradientes. No entanto,
vale a pena revisar a retropropagação (:numref:`sec_backprop`) novamente.


Descrevemos propagações para frente e para trás
e gráficos computacionais
em MLPs em :numref:`sec_backprop`.
A propagação direta em uma RNN é relativamente
para a frente.
*Retropropagação através do tempo* é, na verdade, uma aplicação específica de retropropagação
em RNNs :cite:`Werbos.1990`.
Isto
exige que expandamos o
gráfico computacional de uma RNN
um passo de cada vez para
obter as dependências
entre variáveis ​​e parâmetros do modelo.
Então,
com base na regra da cadeia,
aplicamos retropropagação para calcular e
gradientes de loja.
Uma vez que as sequências podem ser bastante longas, a dependência pode ser bastante longa.
Por exemplo, para uma sequência de 1000 caracteres,
o primeiro token pode ter uma influência significativa sobre o token na posição final.
Isso não é realmente viável computacionalmente
(leva muito tempo e requer muita memória) e requer mais de 1000 produtos de matriz antes de chegarmos a esse gradiente muito indescritível.
Este é um processo repleto de incertezas computacionais e estatísticas.
A seguir iremos elucidar o que acontece
e como resolver isso na prática.

## Análise de Gradientes em RNNs
:label:`subsec_bptt_analysis`


Começamos com um modelo simplificado de como funciona uma RNN.
Este modelo ignora detalhes sobre as especificações do estado oculto e como ele é atualizado.
A notação matemática aqui
não distingue explicitamente
escalares, vetores e matrizes como costumava fazer.
Esses detalhes são irrelevantes para a análise
e serviriam apenas para bagunçar a notação
nesta subseção.

Neste modelo simplificado,
denotamos $h_t$ como o estado oculto,
$x_t$ como a entrada e $o_t$ como a saída
no passo de tempo $t$.
Lembre-se de nossas discussões em
:numref:`subsec_rnn_w_hidden_states`
que a entrada e o estado oculto
podem ser concatenados ao
serem multiplicados por uma variável de peso na camada oculta.
Assim, usamos $w_h$ e $w_o$ para
indicar os pesos da camada oculta e da camada de saída, respectivamente.
Como resultado, os estados ocultos e saídas em cada etapa de tempo podem ser explicados como

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

onde $f$ e $g$ são transformações
da camada oculta e da camada de saída, respectivamente.
Portanto, temos uma cadeia de valores $\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$ que dependem uns dos outros por meio de computação recorrente.
A propagação direta é bastante direta.
Tudo o que precisamos é percorrer as triplas $(x_t, h_t, o_t)$ um passo de tempo de cada vez.
A discrepância entre a saída $o_t$ e o rótulo desejado $y_t$ é então avaliada por uma função objetivo
em todas as etapas de tempo $T$
como

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$



Para retropropagação, as coisas são um pouco mais complicadas, especialmente quando calculamos os gradientes em relação aos parâmetros $w_h$ da função objetivo $L$. Para ser específico, pela regra da cadeia,

$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_o)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

O primeiro e o segundo fatores do
produto em :eqref:`eq_bptt_partial_L_wh`
são fáceis de calcular.
O terceiro fator $\partial h_t/\partial w_h$ é onde as coisas ficam complicadas, já que precisamos calcular recorrentemente o efeito do parâmetro $w_h$ em $h_t$.
De acordo com o cálculo recorrente
em :eqref:`eq_bptt_ht_ot`,
$h_t$ depende de $h_{t-1}$ e $w_h$,
onde cálculo de $h_{t-1}$
também depende de $w_h$.
Assim,
usando a regra da cadeia temos

$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`


Para derivar o gradiente acima, suponha que temos três sequências $\{a_{t}\},\{b_{t}\},\{c_{t}\}$ satisfatória
$a_{0}=0$ and $a_{t}=b_{t}+c_{t}a_{t-1}$ for $t=1, 2,\ldots$.
Então, para $t\geq 1$, é fácil mostrar

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

Substituindo $$a_t$, $b_t$, e $c_t$
de acordo com

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

o cálculo do gradiente em: eqref: `eq_bptt_partial_ht_wh_recur` satisfaz
$a_{t}=b_{t}+c_{t}a_{t-1}$.
Assim,
por :eqref:`eq_bptt_at`,
podemos remover o cálculo recorrente em :eqref:`eq_bptt_partial_ht_wh_recur`
com

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

Embora possamos usar a regra da cadeia para calcular \partial h_t/\partial w_h$ recursivamente, esta cadeia pode ficar muito longa sempre que $t$ for grande. Vamos discutir uma série de estratégias para lidar com esse problema.

### Computação Completa

Obviamente,
podemos apenas calcular a soma total em
:eqref:`eq_bptt_partial_ht_wh_gen`.
Porém,
isso é muito lento e os gradientes podem explodir,
uma vez que mudanças sutis nas condições iniciais podem afetar muito o resultado.
Ou seja, poderíamos ver coisas semelhantes ao efeito borboleta, em que mudanças mínimas nas condições iniciais levam a mudanças desproporcionais no resultado.
Na verdade, isso é bastante indesejável em termos do modelo que queremos estimar.
Afinal, estamos procurando estimadores robustos que generalizem bem. Portanto, essa estratégia quase nunca é usada na prática.

### Truncamento de Etapas de Tempo

Alternativamente,
podemos truncar a soma em
:eqref:`eq_bptt_partial_ht_wh_gen`
após $\tau$  passos.
Isso é o que estivemos discutindo até agora,
como quando separamos os gradientes em :numref:`sec_rnn_scratch`. 
Isso leva a uma *aproximação* do gradiente verdadeiro, simplesmente terminando a soma em
$\partial h_{t-\tau}/\partial w_h$. 
Na prática, isso funciona muito bem. É o que é comumente referido como retropropagação truncada ao longo do tempo :cite:`Jaeger.2002`.
Uma das consequências disso é que o modelo se concentra principalmente na influência de curto prazo, e não nas consequências de longo prazo. Na verdade, isso é *desejável*, pois inclina a estimativa para modelos mais simples e estáveis.

### Truncamento Randomizado ### 

Last, we can replace $\partial h_t/\partial w_h$
by a random variable which is correct in expectation but  truncates the sequence.
This is achieved by using a sequence of $\xi_t$
with predefined $0 \leq \pi_t \leq 1$,
where $P(\xi_t = 0) = 1-\pi_t$ and  $P(\xi_t = \pi_t^{-1}) = \pi_t$, thus $E[\xi_t] = 1$.
We use this to replace the gradient
$\partial h_t/\partial w_h$
in :eqref:`eq_bptt_partial_ht_wh_recur`
with

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$


It follows from the definition of $\xi_t$ that $E[z_t] = \partial h_t/\partial w_h$.
Whenever $\xi_t = 0$ the recurrent computation
terminates at that time step $t$.
This leads to a weighted sum of sequences of varying lengths where long sequences are rare but appropriately overweighted. 
This idea was proposed by Tallec and Ollivier
:cite:`Tallec.Ollivier.2017`.

### Comparing Strategies

![Comparing strategies for computing gradients in RNNs. From top to bottom: randomized truncation, regular truncation, and full computation.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`


:numref:`fig_truncated_bptt` illustrates the three strategies when analyzing the first few characters of *The Time Machine* book using backpropagation through time for RNNs:

* The first row is the randomized truncation that partitions the text into segments of varying lengths.
* The second row is the regular truncation that breaks the text into subsequences of the same length. This is what we have been doing in RNN experiments.
* The third row is the full backpropagation through time that leads to a computationally infeasible expression.


Unfortunately, while appealing in theory, randomized truncation does not work much better than regular truncation, most likely due to a number of factors.
First, the effect of an observation after a number of backpropagation steps into the past is quite sufficient to capture dependencies in practice. 
Second, the increased variance counteracts the fact that the gradient is more accurate with more steps. 
Third, we actually *want* models that have only a short range of interactions. Hence, regularly truncated backpropagation through time has a slight regularizing effect that can be desirable.

## Backpropagation Through Time in Detail

After discussing the general principle,
let us discuss backpropagation through time in detail.
Different from the analysis in
:numref:`subsec_bptt_analysis`,
in the following
we will show
how to compute
the gradients of the objective function
with respect to all the decomposed model parameters.
To keep things simple, we consider 
an RNN without bias parameters,
whose 
activation function
in the hidden layer
uses the identity mapping ($\phi(x)=x$).
For time step $t$,
let the single example input and the label be
$\mathbf{x}_t \in \mathbb{R}^d$ and $y_t$, respectively. 
The hidden state $\mathbf{h}_t \in \mathbb{R}^h$ 
and the output $\mathbf{o}_t \in \mathbb{R}^q$
are computed as

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

where $\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$, and
$\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$
are the weight parameters.
Denote by $l(\mathbf{o}_t, y_t)$
the loss at time step $t$. 
Our objective function,
the loss over $T$ time steps
from the beginning of the sequence
is thus

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$


In order to visualize the dependencies among
model variables and parameters during computation
of the RNN,
we can draw a computational graph for the model,
as shown in :numref:`fig_rnn_bptt`.
For example, the computation of the hidden states of time step 3, $\mathbf{h}_3$, depends on the model parameters $\mathbf{W}_{hx}$ and $\mathbf{W}_{hh}$,
the hidden state of the last time step $\mathbf{h}_2$,
and the input of the current time step $\mathbf{x}_3$.

![Computational graph showing dependencies for an RNN model with three time steps. Boxes represent variables (not shaded) or parameters (shaded) and circles represent operators.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

As just mentioned, the model parameters in :numref:`fig_rnn_bptt` are $\mathbf{W}_{hx}$, $\mathbf{W}_{hh}$, and $\mathbf{W}_{qh}$. 
Generally,
training this model
requires 
gradient computation with respect to these parameters
$\partial L/\partial \mathbf{W}_{hx}$, $\partial L/\partial \mathbf{W}_{hh}$, and $\partial L/\partial \mathbf{W}_{qh}$.
According to the dependencies in :numref:`fig_rnn_bptt`,
we can traverse 
in the opposite direction of the arrows
to calculate and store the gradients in turn.
To flexibly express the multiplication
of matrices, vectors, and scalars of different shapes
in the chain rule,
we continue to use 
the 
$\text{prod}$ operator as described in
:numref:`sec_backprop`.


First of all,
differentiating the objective function
with respect to the model output
at any time step $t$
is fairly straightforward:

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

Now, we can calculate the gradient of the objective function
with respect to
the parameter $\mathbf{W}_{qh}$
in the output layer:
$\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$. Based on :numref:`fig_rnn_bptt`, 
the objective function
$L$ depends on $\mathbf{W}_{qh}$ via $\mathbf{o}_1, \ldots, \mathbf{o}_T$. Using the chain rule yields

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

where $\partial L/\partial \mathbf{o}_t$
is given by :eqref:`eq_bptt_partial_L_ot`.

Next, as shown in :numref:`fig_rnn_bptt`,
at the final time step $T$
the objective function
$L$ depends on the hidden state $\mathbf{h}_T$ only via $\mathbf{o}_T$.
Therefore, we can easily find
the gradient 
$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$
using the chain rule:

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

It gets trickier for any time step $t < T$,
where the objective function $L$ depends on $\mathbf{h}_t$ via $\mathbf{h}_{t+1}$ and $\mathbf{o}_t$.
According to the chain rule,
the gradient of the hidden state
$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$
at any time step $t < T$ can be recurrently computed as:


$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

For analysis,
expanding the recurrent computation
for any time step $1 \leq t \leq T$
gives

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

We can see from :eqref:`eq_bptt_partial_L_ht` that
this simple linear example already
exhibits some key problems of long sequence models: it involves potentially very large powers of $\mathbf{W}_{hh}^\top$.
In it, eigenvalues smaller than 1 vanish
and eigenvalues larger than 1 diverge.
This is numerically unstable,
which manifests itself in the form of vanishing 
and exploding gradients.
One way to address this is to truncate the time steps
at a computationally convenient size as discussed in :numref:`subsec_bptt_analysis`. 
In practice, this truncation is effected by detaching the gradient after a given number of time steps.
Later on 
we will see how more sophisticated sequence models such as long short-term memory can alleviate this further. 

Finally,
:numref:`fig_rnn_bptt` shows that
the objective function
$L$ depends on model parameters
$\mathbf{W}_{hx}$ and $\mathbf{W}_{hh}$
in the hidden layer
via hidden states
$\mathbf{h}_1, \ldots, \mathbf{h}_T$.
To compute gradients
with respect to such parameters
$\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$ and $\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$,
we apply the chain rule that gives

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

where
$\partial L/\partial \mathbf{h}_t$
that is recurrently computed by
:eqref:`eq_bptt_partial_L_hT_final_step`
and
:eqref:`eq_bptt_partial_L_ht_recur`
is the key quantity
that affects the numerical stability.



Since backpropagation through time
is the application of backpropagation in RNNs,
as we have explained in :numref:`sec_backprop`,
training RNNs
alternates forward propagation with
backpropagation through time.
Besides,
backpropagation through time
computes and stores the above gradients
in turn.
Specifically,
stored intermediate values
are reused
to avoid duplicate calculations,
such as storing 
$\partial L/\partial \mathbf{h}_t$
to be used in computation of both $\partial L / \partial \mathbf{W}_{hx}$ and $\partial L / \partial \mathbf{W}_{hh}$.


## Summary

* Backpropagation through time is merely an application of backpropagation to sequence models with a hidden state.
* Truncation is needed for computational convenience and numerical stability, such as regular truncation and randomized truncation.
* High powers of matrices can lead to divergent or vanishing eigenvalues. This manifests itself in the form of exploding or vanishing gradients.
* For efficient computation, intermediate values are cached during backpropagation through time.



## Exercises

1. Assume that we have a symmetric matrix $\mathbf{M} \in \mathbb{R}^{n \times n}$ with eigenvalues $\lambda_i$ whose corresponding eigenvectors are $\mathbf{v}_i$ ($i = 1, \ldots, n$). Without loss of generality, assume that they are ordered in the order $|\lambda_i| \geq |\lambda_{i+1}|$. 
   1. Show that $\mathbf{M}^k$ has eigenvalues $\lambda_i^k$.
   1. Prove that for a random vector $\mathbf{x} \in \mathbb{R}^n$, with high probability $\mathbf{M}^k \mathbf{x}$ will be very much aligned with the eigenvector $\mathbf{v}_1$ 
of $\mathbf{M}$. Formalize this statement.
   1. What does the above result mean for gradients in RNNs?
1. Besides gradient clipping, can you think of any other methods to cope with gradient explosion in recurrent neural networks?

[Discussions](https://discuss.d2l.ai/t/334)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTM1NjI4MDQyMSwxOTA1NDcxOTUzLDUwMD
Q2NDUzMiwtMTEwMjc2OTQwNF19
-->
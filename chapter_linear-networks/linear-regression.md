# Linear Regression
:label:`sec_linear_regression`

*Regression* refers to a set of methods for modeling
the relationship between one or more independent variables
and a dependent variable.
In the natural sciences and social sciences,
the purpose of regression is most often to
*characterize* the relationship between the inputs and outputs.
Machine learning, on the other hand,
is most often concerned with *prediction*.

Regression problems pop up whenever we want to predict a numerical value.
Common examples include predicting prices (of homes, stocks, etc.),
predicting length of stay (for patients in the hospital),
demand forecasting (for retail sales), among countless others.
Not every prediction problem is a classic regression problem.
In subsequent sections, we will introduce classification problems,
where the goal is to predict membership among a set of categories.


## Elementos Básicos de Regressão Linear


*Regressão linear* pode ser a mais simples
e mais popular entre as ferramentas padrão para regressão.
Datado do início do século 19,
A regressão linear flui a partir de algumas suposições simples.
Primeiro, assumimos que a relação entre
as variáveis ​​independentes $\mathbf{x}$ e a variável dependente $y$ é linear,
ou seja, esse $y$ pode ser expresso como uma soma ponderada
dos elementos em $\mathbf{x}$,
dado algum ruído nas observações.
Em segundo lugar, assumimos que qualquer ruído é bem comportado
(seguindo uma distribuição gaussiana).

Para motivar a abordagem, vamos começar com um exemplo de execução.
Suponha que desejamos estimar os preços das casas (em dólares)
com base em sua área (em pés quadrados) e idade (em anos).
Para realmente desenvolver um modelo para prever os preços das casas,
precisaríamos colocar as mãos em um conjunto de dados
consistindo em vendas para as quais sabemos
o preço de venda, área e idade de cada casa.
Na terminologia de *machine learning*,
o conjunto de dados é chamado de *dataset de treinamento* ou *conjunto de treinamento*,
e cada linha (aqui os dados correspondentes a uma venda)
é chamado de *exemplo* (ou *tupla*, *instância de dados*, * amostra *).
O que estamos tentando prever (preço)
é chamado de * label* (ou *rótulo*).
As variáveis ​​independentes (idade e área)
em que as previsões são baseadas
são chamadas de *features* (ou *covariáveis*).

Normalmente, usaremos $n$ para denotar
o número de exemplos em nosso conjunto de dados.
Nós indexamos os exemplos de dados por $i$, denotando cada entrada
como $\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^\top$
e o *label* correspondente como $y^{(i)}$.


### Modelo Linear
:label:`subsec_linear_model`


A suposição de linearidade apenas diz que o alvo (preço)
pode ser expresso como uma soma ponderada das características (área e idade):

$$\mathrm{preço} = w_{\mathrm{área}}\cdot\mathrm{área} + w_ \mathrm{idade}}\cdot\mathrm{idade} + b.$$:eqlabel:`eq_price-area`

In :eqref:`eq_price-area`, $ w_{\mathrm{area}}$ e $w_{\mathrm{age}}$
são chamados de *pesos* e $b$ é chamado de *bias*
(também chamado de *deslocamento* ou *offset*).
Os pesos determinam a influência de cada *feature*
em nossa previsão e o *bias* apenas diz
qual valor o preço previsto deve assumir
quando todos os *features* assumem o valor 0.
Mesmo que nunca vejamos nenhuma casa com área zero,
ou que têm exatamente zero anos de idade,
ainda precisamos do *bias* ou então vamos
limitar a expressividade do nosso modelo.
Estritamente falando, :eqref:`eq_price-area` é uma *transformação afim*
de *features* de entrada,
que é caracterizada por
uma *transformação linear* de *features* via soma ponderada, combinada com
uma *tradução* por meio do *bias* adicionado.

Dado um *dataset*, nosso objetivo é escolher
os pesos $\mathbf{w}$ e o *bias* $b$ de modo que, em média,
as previsões feitas de acordo com nosso modelo
 se ajustem o melhor possível aos preços reais observados nos dados.
Modelos cuja previsão de saída
é determinada pela transformação afim de *features* de entrada
são *modelos lineares*,
onde a transformação afim é especificada pelos pesos e *bias* escolhidos.


Em disciplinas onde é comum se concentrar
em conjuntos de dados com apenas alguns *features*,
expressar explicitamente modelos de formato longo como esse é comum.
No  *machine learning*, geralmente trabalhamos com *datasets* de alta dimensão,
portanto, é mais conveniente empregar a notação de álgebra linear.
Quando nossas entradas consistem em $d$ *features*,
expressamos nossa previsão $\hat{y}$ (em geral, o símbolo "chapéu" ou  "acento circunflexo" denota estimativas) como

$$\hat{y} = w_1  x_1 + ... + w_d  x_d + b.$$

Coletando todas as *features* em um vetor $\mathbf{x}\in\mathbb{R}^d$
e todos os pesos em um vetor $\mathbf{w}\in\mathbb{R}^d$,
podemos expressar nosso modelo compactamente usando um produto escalar:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`


Em :eqref:`eq_linreg-y`, o vetor $\mathbf{x}$ corresponde às *features* de um único exemplo de dados.
Frequentemente acharemos conveniente
para se referir a recursos de todo o nosso *dataset* de $n$ exemplos
através da *matriz de design* $\mathbf{X}\in\mathbb{R}^{n\times d}$.
Aqui, $\mathbf{X}$ contém uma linha para cada exemplo
e uma coluna para cada *feature*.

Para uma coleção de *features* $\mathbf{X}$,
as previsões $\hat{\mathbf{y}}\in\mathbb{R}^n$
pode ser expresso por meio do produto matriz-vetor:
$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$

onde a transmissão (veja :numref:`subsec_broadcasting`) é aplicada durante o somatório.
Dadas as *features* de um *dataset* de treinamento $\mathbf{X}$
e *labels* correspondentes (conhecidos) $\mathbf{y}$,
o objetivo da regressão linear é encontrar
o vetor de pesos $\mathbf{w}$ e o termo de polarização $b$
que dadas as *features* de um novo exemplo de dados
amostrado da mesma distribuição de $\mathbf{X}$,
o *label* do novo exemplo será (na expectativa) previsto com o menor erro.


Mesmo se acreditarmos que o melhor modelo para
predizer $y$ dado $\mathbf{x}$ é linear,
não esperaríamos encontrar um *dataset* do mundo real de $n$ exemplos onde
$y^{(i)}$ é exatamente igual a $\mathbf{w}^\top\mathbf{x}^{(i)} + b$
para todos $1\leq i \leq n$.
Por exemplo, quaisquer instrumentos que usarmos para observar
as *features* $\mathbf{X}$ e os *labels* $\mathbf{y}$
podem sofrer uma pequena quantidade de erro de medição.
Assim, mesmo quando estamos confiantes
que a relação subjacente é linear,
vamos incorporar um termo de ruído para contabilizar esses erros.

Antes de começarmos a pesquisar os melhores *parâmetros* (ou *parâmetros do modelo*) $\mathbf{w}$ e $b$,
precisaremos de mais duas coisas:
(i) uma medida de qualidade para algum modelo dado;
e (ii) um procedimento de atualização do modelo para melhorar sua qualidade.


### Função de Perda

Antes de começarmos a pensar sobre como *ajustar* os dados ao nosso modelo,
precisamos determinar uma medida de *aptidão*.
A *função de perda* quantifica a distância
entre o valor *real* e *previsto* do *target*.
A perda geralmente será um número não negativo
onde valores menores são melhores
e previsões perfeitas incorrem em uma perda de 0.
A função de perda mais popular em problemas de regressão
é o erro quadrático.
Quando nossa previsão para um exemplo $i$ é $\hat{y}^{(i)}$
e o *label verdadeiro correspondente é $y^{(i)}$,
o quadrado do erro é dado por:

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

A constante $\frac{1}{2}$ não faz diferença real
mas será notacionalmente conveniente,
cancelando quando tomamos a derivada da perda.
Como o conjunto de dados de treinamento é fornecido a nós e, portanto, está fora de nosso controle,
o erro empírico é apenas função dos parâmetros do modelo.
Para tornar as coisas mais concretas, considere o exemplo abaixo
onde traçamos um problema de regressão para um caso unidimensional
como mostrado em :numref:`fig_fit_linreg`.

![Fit data with a linear model.](../img/fit-linreg.svg)
:label:`fig_fit_linreg`

Observe que grandes diferenças entre
estimativas $\hat{y}^{(i)}$ e observações $y^{(i)}$
levam a contribuições ainda maiores para a perda,
devido à dependência quadrática.
Para medir a qualidade de um modelo em todo o conjunto de dados de $n$ exemplos,
nós simplesmente calculamos a média (ou equivalentemente, somamos)
as perdas no conjunto de treinamento.

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Ao treinar o modelo, queremos encontrar os parâmetros ($\mathbf{w}^*, b^*$)
que minimizam a perda total em todos os exemplos de treinamento:

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$


### Solução Analítica

A regressão linear passa a ser um problema de otimização incomumente simples.
Ao contrário da maioria dos outros modelos que encontraremos neste livro,
a regressão linear pode ser resolvida analiticamente aplicando uma fórmula simples.
Para começar, podemos incluir o *bias* $b$ no parâmetro $\mathbf{w}$
anexando uma coluna à matriz de design que consiste em todas as unidades.
Então nosso problema de previsão é minimizar $\|\mathbf{y} -\mathbf{X}\mathbf{w}\|^2$.
Há apenas um ponto crítico na superfície de perda
e corresponde ao mínimo de perda em todo o domínio.
Tirando a derivada da perda em relação a $\mathbf{w}$
e defini-lo igual a zero produz a solução analítica (de forma fechada):

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}.$$

Embora problemas simples como regressão linear
podem admitir soluções analíticas,
você não deve se acostumar com essa boa sorte.
Embora as soluções analíticas permitam uma boa análise matemática,
o requisito de uma solução analítica é tão restritivo
que isso excluiria todo o *deep learning*.

### Gradiente Descendente Estocástico com *Minibatch* 


Mesmo nos casos em que não podemos resolver os modelos analiticamente,
acontece que ainda podemos treinar modelos efetivamente na prática.
Além disso, para muitas tarefas, aqueles modelos difíceis de otimizar
acabam sendo muito melhores do que descobrir como treiná-los
acaba valendo a pena.

A principal técnica para otimizar quase qualquer modelo de *deep learning*,
e que recorreremos ao longo deste livro,
consiste em reduzir iterativamente o erro
atualizando os parâmetros na direção
que diminui gradativamente a função de perda.
Este algoritmo é denominado *gradiente descendente*.

A aplicação mais ingênua de gradiente descendente
consiste em obter a derivada da função de perda,
que é uma média das perdas calculadas
em cada exemplo no *dataset*.
Na prática, isso pode ser extremamente lento:
devemos passar por todo o conjunto de dados antes de fazer uma única atualização.
Assim, frequentemente nos contentaremos em amostrar um *minibatch* aleatório de exemplos
toda vez que precisamos calcular a atualização,
uma variante chamada *gradiente descendente estocástico de minibatch*.

Em cada iteração, primeiro amostramos aleatoriamente um *minibatch* $\mathcal{B}$
consistindo em um número fixo de exemplos de treinamento.
Em seguida, calculamos a derivada (gradiente) da perda média
no *minibatch* em relação aos parâmetros do modelo.
Finalmente, multiplicamos o gradiente por um valor positivo predeterminado $\eta$
e subtraimos o termo resultante dos valores dos parâmetros atuais.

Podemos expressar a atualização matematicamente da seguinte forma
($\partial$ denota a derivada parcial):

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$


Para resumir, as etapas do algoritmo são as seguintes:
(i) inicializamos os valores dos parâmetros do modelo, normalmente de forma aleatória;
(ii) amostramos iterativamente *minibatches* aleatórios dos dados,
atualizando os parâmetros na direção do gradiente negativo.
Para perdas quadráticas e transformações afins,
podemos escrever isso explicitamente da seguinte maneira:

$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`



Observe que $\mathbf{w}$ e $\mathbf{x}$ são vetores em :eqref:`eq_linreg_batch_update`.
Aqui, a notação vetorial mais elegante torna a matemática
muito mais legível do que expressar coisas em termos de coeficientes,
diga $w_1, w_2, \ldots, w_d$.
A cardinalidade definida
$|\mathcal{B}|$ representa
o número de exemplos em cada *minibatch* (o *tamanho do lote*)
e $\eta$ denota a *taxa de aprendizagem*.
Enfatizamos que os valores do tamanho do lote e da taxa de aprendizagem
são pré-especificados manualmente e normalmente não aprendidos por meio do treinamento do modelo.
Esses parâmetros são ajustáveis, mas não atualizados
no loop de treinamento são chamados de *hiperparâmetros*.
*Ajuste de hiperparâmetros* é o processo pelo qual os hiperparâmetros são escolhidos,
e normalmente requer que os ajustemos
com base nos resultados do ciclo de treinamento
conforme avaliado em um *dataset de validação* separado (ou *conjunto de validação*).

Após o treinamento para algum número predeterminado de iterações
(ou até que algum outro critério de parada seja atendido),
registramos os parâmetros estimados do modelo,
denotado $\hat{\mathbf{w}}, \hat{b}$.
Observe que mesmo que nossa função seja verdadeiramente linear e sem ruídos,
esses parâmetros não serão os minimizadores exatos da perda
porque, embora o algoritmo convirja lentamente para os minimizadores,
não pode alcançá-los exatamente em um número finito de etapas.

A regressão linear passa a ser um problema de aprendizagem onde há apenas um mínimo
em todo o domínio.
No entanto, para modelos mais complicados, como redes profundas,
as superfícies de perda contêm muitos mínimos.
Felizmente, por razões que ainda não são totalmente compreendidas,
praticantes de *deep learning* raramente se esforçam para encontrar parâmetros
que minimizem a perda *em conjuntos de treinamento*.
A tarefa mais formidável é encontrar parâmetros
que irão atingir baixa perda de dados
que não vimos antes,
um desafio chamado *generalização*.
Retornamos a esses tópicos ao longo do livro.


### Fazendo Predições com o Modelo Aprendido



Dado o modelo de regressão linear aprendido
$\hat{\mathbf{w}}^\top\mathbf{x} + \hat{b}$,
agora podemos estimar o preço de uma nova casa
(não contido nos dados de treinamento)
dada sua área $x_1$ e idade $x_2$.
Estimar *labels* dadas as características é
comumente chamado de *predição* ou *inferência*.

Tentaremos manter o termo *predição* porque
chamando esta etapa de *inferência*,
apesar de emergir como jargão padrão no *deep learning*,
é um nome impróprio.
Em estatísticas, *inferência* denota mais frequentemente
estimar parâmetros com base em um conjunto de dados.
Este uso indevido de terminologia é uma fonte comum de confusão
quando os profissionais de *machine learning* conversam com os estatísticos.

## Vetorização para Velocidade

Ao treinar nossos modelos, normalmente queremos processar
*minibatches* inteiros de exemplos simultaneamente.
Fazer isso de forma eficiente requer que (**nós**) (~~devemos~~) (**vetorizar os cálculos
e aproveitar as bibliotecas de álgebra linear rápida
em vez de escrever *loops for* custosos em Python.**)

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

Para ilustrar por que isso é tão importante,
podemos (**considerer dois métodos para adicionar vetores.**)
Para começar, instanciamos dois vetores de 10000 dimensões
contendo todos os outros.
Em um método, faremos um loop sobre os vetores com um *loop for* Python.
No outro método, contaremos com uma única chamada para `+`.

```{.python .input}
#@tab all
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

Uma vez que iremos comparar o tempo de execução com freqüência neste livro,
[**vamos definir um cronômetro**].

```{.python .input}
#@tab all
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
```

Agora podemos avaliar as cargas de trabalho.
Primeiro, [**nós os adicionamos, uma coordenada por vez,
usando um *loop for*.**]

```{.python .input}
#@tab mxnet, pytorch
c = d2l.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'
```

```{.python .input}
#@tab tensorflow
c = tf.Variable(d2l.zeros(n))
timer = Timer()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{timer.stop():.5f} sec'
```

(**Alternativamente, contamos com o operador recarregado `+` para calcular a soma elemento a elemento.**)

```{.python .input}
#@tab all
timer.start()
d = a + b
f'{timer.stop():.5f} sec'
```

Você provavelmente percebeu que o segundo método
é dramaticamente mais rápido que o primeiro.
A vetorização do código geralmente produz acelerações da ordem de magnitude.
Além disso, colocamos mais matemática na biblioteca
e não precisamos escrever tantos cálculos nós mesmos,
reduzindo o potencial de erros.

## A Distribuição Normal e Perda Quadrada
:label:`subsec_normal_distribution_and_squared_loss`


Embora você já possa sujar as mãos usando apenas as informações acima,
a seguir, podemos motivar mais formalmente o objetivo de perda quadrado
através de suposições sobre a distribuição do ruído.

A regressão linear foi inventada por Gauss em 1795,
que também descobriu a distribuição normal (também chamada de *Gaussiana*).
Acontece que a conexão entre
a distribuição normal e regressão linear
é mais profunda do que o parentesco comum.
Para refrescar sua memória, a densidade de probabilidade
de uma distribuição normal com média $\mu$ e variância $\sigma^2 $(desvio padrão $\sigma$)
é dada como
$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

Below [**we define a Python function to compute the normal distribution**].

```{.python .input}
#@tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
```

We can now (**visualize the normal distributions**).

```{.python .input}
#@tab all
# Use numpy again for visualization
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

As we can see, changing the mean corresponds to a shift along the $x$-axis,
and increasing the variance spreads the distribution out, lowering its peak.

One way to motivate linear regression with the mean squared error loss function (or simply squared loss)
is to formally assume that observations arise from noisy observations,
where the noise is normally distributed as follows:

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

Thus, we can now write out the *likelihood*
of seeing a particular $y$ for a given $\mathbf{x}$ via

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

Now, according to the principle of maximum likelihood,
the best values of parameters $\mathbf{w}$ and $b$ are those
that maximize the *likelihood* of the entire dataset:

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).$$

Estimators chosen according to the principle of maximum likelihood
are called *maximum likelihood estimators*.
While, maximizing the product of many exponential functions,
might look difficult,
we can simplify things significantly, without changing the objective,
by maximizing the log of the likelihood instead.
For historical reasons, optimizations are more often expressed
as minimization rather than maximization.
So, without changing anything we can minimize the *negative log-likelihood*
$-\log P(\mathbf y \mid \mathbf X)$.
Working out the mathematics gives us:

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

Now we just need one more assumption that $\sigma$ is some fixed constant.
Thus we can ignore the first term because
it does not depend on $\mathbf{w}$ or $b$.
Now the second term is identical to the squared error loss introduced earlier,
except for the multiplicative constant $\frac{1}{\sigma^2}$.
Fortunately, the solution does not depend on $\sigma$.
It follows that minimizing the mean squared error
is equivalent to maximum likelihood estimation
of a linear model under the assumption of additive Gaussian noise.

## From Linear Regression to Deep Networks

So far we only talked about linear models.
While neural networks cover a much richer family of models,
we can begin thinking of the linear model
as a neural network by expressing it in the language of neural networks.
To begin, let us start by rewriting things in a "layer" notation.

### Neural Network Diagram

Deep learning practitioners like to draw diagrams
to visualize what is happening in their models.
In :numref:`fig_single_neuron`,
we depict our linear regression model as a neural network.
Note that these diagrams highlight the connectivity pattern
such as how each input is connected to the output,
but not the values taken by the weights or biases.

![Linear regression is a single-layer neural network.](../img/singleneuron.svg)
:label:`fig_single_neuron`

For the neural network shown in :numref:`fig_single_neuron`,
the inputs are $x_1, \ldots, x_d$,
so the *number of inputs* (or *feature dimensionality*) in the input layer is $d$.
The output of the network in :numref:`fig_single_neuron` is $o_1$,
so the *number of outputs* in the output layer is 1.
Note that the input values are all *given*
and there is just a single *computed* neuron.
Focusing on where computation takes place,
conventionally we do not consider the input layer when counting layers.
That is to say,
the *number of layers* for the neural network in :numref:`fig_single_neuron` is 1.
We can think of linear regression models as neural networks
consisting of just a single artificial neuron,
or as single-layer neural networks.

Since for linear regression, every input is connected
to every output (in this case there is only one output),
we can regard this transformation (the output layer in :numref:`fig_single_neuron`)
as a *fully-connected layer* or *dense layer*.
We will talk a lot more about networks composed of such layers
in the next chapter.


### Biology

Since linear regression (invented in 1795)
predates computational neuroscience,
it might seem anachronistic to describe
linear regression as a neural network.
To see why linear models were a natural place to begin
when the cyberneticists/neurophysiologists
Warren McCulloch and Walter Pitts began to develop
models of artificial neurons,
consider the cartoonish picture
of a biological neuron in :numref:`fig_Neuron`, consisting of
*dendrites* (input terminals),
the *nucleus* (CPU), the *axon* (output wire),
and the *axon terminals* (output terminals),
enabling connections to other neurons via *synapses*.

![The real neuron.](../img/neuron.svg)
:label:`fig_Neuron`

Information $x_i$ arriving from other neurons
(or environmental sensors such as the retina)
is received in the dendrites.
In particular, that information is weighted by *synaptic weights* $w_i$
determining the effect of the inputs
(e.g., activation or inhibition via the product $x_i w_i$).
The weighted inputs arriving from multiple sources
are aggregated in the nucleus as a weighted sum $y = \sum_i x_i w_i + b$,
and this information is then sent for further processing in the axon $y$,
typically after some nonlinear processing via $\sigma(y)$.
From there it either reaches its destination (e.g., a muscle)
or is fed into another neuron via its dendrites.

Certainly, the high-level idea that many such units
could be cobbled together with the right connectivity
and right learning algorithm,
to produce far more interesting and complex behavior
than any one neuron alone could express
owes to our study of real biological neural systems.

At the same time, most research in deep learning today
draws little direct inspiration in neuroscience.
We invoke Stuart Russell and Peter Norvig who,
in their classic AI text book
*Artificial Intelligence: A Modern Approach* :cite:`Russell.Norvig.2016`,
pointed out that although airplanes might have been *inspired* by birds,
ornithology has not been the primary driver
of aeronautics innovation for some centuries.
Likewise, inspiration in deep learning these days
comes in equal or greater measure from mathematics,
statistics, and computer science.

## Summary

* Key ingredients in a machine learning model are training data, a loss function, an optimization algorithm, and quite obviously, the model itself.
* Vectorizing makes everything better (mostly math) and faster (mostly code).
* Minimizing an objective function and performing maximum likelihood estimation can mean the same thing.
* Linear regression models are neural networks, too.


## Exercises

1. Assume that we have some data $x_1, \ldots, x_n \in \mathbb{R}$. Our goal is to find a constant $b$ such that $\sum_i (x_i - b)^2$ is minimized.
    1. Find a analytic solution for the optimal value of $b$.
    1. How does this problem and its solution relate to the normal distribution?
1. Derive the analytic solution to the optimization problem for linear regression with squared error. To keep things simple, you can omit the bias $b$ from the problem (we can do this in principled fashion by adding one column to $\mathbf X$ consisting of all ones).
    1. Write out the optimization problem in matrix and vector notation (treat all the data as a single matrix, and all the target values as a single vector).
    1. Compute the gradient of the loss with respect to $w$.
    1. Find the analytic solution by setting the gradient equal to zero and solving the matrix equation.
    1. When might this be better than using stochastic gradient descent? When might this method break?
1. Assume that the noise model governing the additive noise $\epsilon$ is the exponential distribution. That is, $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    1. Write out the negative log-likelihood of the data under the model $-\log P(\mathbf y \mid \mathbf X)$.
    1. Can you find a closed form solution?
    1. Suggest a stochastic gradient descent algorithm to solve this problem. What could possibly go wrong (hint: what happens near the stationary point as we keep on updating the parameters)? Can you fix this?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/40)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/259)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbODMwMDY4MDYyLDIzMTI0NDQyNCwxMzE4MT
c5ODcsLTE1MTAxMDMwNzIsMjI3MjM3NjM4LC0xNDQ3NTc3NTIw
LDE5MTQ2NjU3MDIsMzgwNjQzODA1LC0xNjc1ODAwNTcsLTE3OT
g3OTA0MjRdfQ==
-->
# Adagrad
:label:`sec_adagrad`

Vamos começar considerando os problemas de aprendizado com recursos que ocorrem com pouca frequência.

## Recursos esparsos e taxas de aprendizado

Imagine que estamos treinando um modelo de linguagem. Para obter uma boa precisão, normalmente queremos diminuir a taxa de aprendizado à medida que continuamos treinando, normalmente a uma taxa de $\mathcal{O}(t^{-\frac{1}{2}})$ ou mais lenta. Agora, considere um treinamento de modelo em recursos esparsos, ou seja, recursos que ocorrem raramente. Isso é comum para a linguagem natural, por exemplo, é muito menos provável que vejamos a palavra *précondicionamento* do que *aprendizagem*. No entanto, também é comum em outras áreas, como publicidade computacional e filtragem colaborativa personalizada. Afinal, existem muitas coisas que interessam apenas a um pequeno número de pessoas.

Os parâmetros associados a recursos pouco frequentes recebem apenas atualizações significativas sempre que esses recursos ocorrem. Dada uma taxa de aprendizado decrescente, podemos acabar em uma situação em que os parâmetros para características comuns convergem rapidamente para seus valores ideais, enquanto para características raras ainda não podemos observá-los com frequência suficiente antes que seus valores ideais possam ser determinados. Em outras palavras, a taxa de aprendizado diminui muito lentamente para recursos freqüentes ou muito rapidamente para recursos pouco frequentes.

Um possível hack para corrigir esse problema seria contar o número de vezes que vemos um determinado recurso e usar isso como um relógio para ajustar as taxas de aprendizagem. Ou seja, em vez de escolher uma taxa de aprendizagem da forma $\eta = \frac{\eta_0}{\sqrt{t + c}}$ poderiamos usar $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$ conta o número de valores diferentes de zero para o recurso $i$ que observamos até o momento $t$. Na verdade, isso é muito fácil de implementar sem sobrecarga significativa. No entanto, ele falha sempre que não temos esparsidade, mas apenas dados em que os gradientes são frequentemente muito pequenos e raramente grandes. Afinal, não está claro onde se traçaria a linha entre algo que se qualifica como uma característica observada ou não.

Adagrad by :cite:`Duchi.Hazan.Singer.2011` aborda isso substituindo o contador bastante bruto $s(i, t)$ por um agregado de quadrados de gradientes previamente observados. Em particular, ele usa $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$ como um meio de ajustar a taxa de aprendizagem. Isso tem dois benefícios: primeiro, não precisamos mais decidir apenas quando um gradiente é grande o suficiente. Em segundo lugar, ele é dimensionado automaticamente com a magnitude dos gradientes. As coordenadas que normalmente correspondem a grandes gradientes são reduzidas significativamente, enquanto outras com pequenos gradientes recebem um tratamento muito mais suave. Na prática, isso leva a um procedimento de otimização muito eficaz para publicidade computacional e problemas relacionados. Mas isso oculta alguns dos benefícios adicionais inerentes ao Adagrad que são mais bem compreendidos no contexto do pré-condicionamento.


## Precondicionamento

Problemas de otimização convexa são bons para analisar as características dos algoritmos. Afinal, para a maioria dos problemas não-convexos, é difícil derivar garantias teóricas significativas, mas a *intuição* e o *insight* geralmente são transmitidos. Vejamos o problema de minimizar $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$.

Como vimos em :numref:`sec_momentum`, é possível reescrever este problema em termos de sua composição automática $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ para chegar a um problema muito simplificado onde cada coordenada pode ser resolvida individualmente:

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

Aqui usamos $\mathbf{x} = \mathbf{U} \mathbf{x}$ e consequentemente $\mathbf{c} = \mathbf{U} \mathbf{c}$. O problema modificado tem como minimizador $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$ e valor mínimo $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$. Isso é muito mais fácil de calcular, pois $\boldsymbol{\Lambda}$ é uma matriz diagonal contendo os autovalores de $\mathbf{Q}$.

Se perturbarmos $\mathbf{c}$ ligeiramente, esperaríamos encontrar apenas pequenas mudanças no minimizador de $ f $. Infelizmente, esse não é o caso. Embora pequenas mudanças em $\mathbf{c}$ levem a mudanças igualmente pequenas em $\bar{\mathbf{c}}$, este não é o caso para o minimizador de $f$ (e de $\bar{f}$ respectivamente). Sempre que os autovalores $\boldsymbol{\Lambda}_i$ forem grandes, veremos apenas pequenas mudanças em $\bar{x}_i$ e no mínimo de $\bar{f}$. Por outro lado, para pequenas $\boldsymbol{\Lambda}_i$, as mudanças em $\bar{x}_i$ podem ser dramáticas. A razão entre o maior e o menor autovalor é chamada de número de condição de um problema de otimização.

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

Se o número de condição $\kappa$ for grande, será difícil resolver o problema de otimização com precisão. Precisamos garantir que somos cuidadosos ao acertar uma ampla faixa dinâmica de valores. Nossa análise leva a uma questão óbvia, embora um tanto ingênua: não poderíamos simplesmente "consertar" o problema distorcendo o espaço de forma que todos os autovalores sejam $1$. Em teoria, isso é muito fácil: precisamos apenas dos autovalores e autovetores de $\mathbf{Q}$ para redimensionar o problema de $\mathbf{x}$ para um em $\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$. No novo sistema de coordenadas $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ poderia ser simplificado para $\|\mathbf{z}\|^2$. Infelizmente, esta é uma sugestão pouco prática. O cálculo de autovalores e autovetores é em geral muito mais caro do que resolver o problema real.

Embora o cálculo exato dos autovalores possa ser caro, adivinhá-los e computá-los de forma aproximada já pode ser muito melhor do que não fazer nada. Em particular, poderíamos usar as entradas diagonais de $\mathbf{Q}$ e redimensioná-las de acordo. Isso é *muito* mais barato do que calcular valores próprios.

$$\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

Neste caso, temos $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ e especificamente $\tilde{\mathbf{Q}}_{ii} = 1$ para todo $i$. Na maioria dos casos, isso simplifica consideravelmente o número da condição. Por exemplo, nos casos que discutimos anteriormente, isso eliminaria totalmente o problema em questão, uma vez que o problema está alinhado ao eixo.

Infelizmente, enfrentamos ainda outro problema: no aprendizado profundo, normalmente nem mesmo temos acesso à segunda derivada da função objetivo: para $\mathbf{x} \in \mathbb{R}^d$ a segunda derivada, mesmo em um minibatch pode exigir $\mathcal{O}(d^2)$ espaço e trabalho para computar, tornando-o praticamente inviável. A ideia engenhosa do Adagrad é usar um proxy para aquela diagonal indescritível do Hessian que é relativamente barato para calcular e eficaz--- a magnitude do gradiente em si.

Para ver por que isso funciona, vamos dar uma olhada em $\bar{f}(\bar{\mathbf{x}})$. Nós temos isso

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

onde $\bar{\mathbf{x}}_0$ é o minimizador de $\bar{f}$. Portanto, a magnitude do gradiente depende tanto de $\boldsymbol{\Lambda}$ quanto da distância da otimalidade. Se $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ não mudou, isso seria tudo o que é necessário. Afinal, neste caso, a magnitude do gradiente $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ é suficiente. Como o AdaGrad é um algoritmo descendente de gradiente estocástico, veremos gradientes com variância diferente de zero mesmo em otimização. Como resultado, podemos usar com segurança a variância dos gradientes como um proxy barato para a escala de Hessian. Uma análise completa está além do escopo desta seção (seriam várias páginas). Recomendamos ao leitor :cite:`Duchi.Hazan.Singer.2011` para detalhes.

## O Algoritmo

Let us formalize the discussion from above. We use the variable $\mathbf{s}_t$ to accumulate past gradient variance as follows.

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

Here the operation are applied coordinate wise. That is, $\mathbf{v}^2$ has entries $v_i^2$. Likewise $\frac{1}{\sqrt{v}}$ has entries $\frac{1}{\sqrt{v_i}}$ and $\mathbf{u} \cdot \mathbf{v}$ has entries $u_i v_i$. As before $\eta$ is the learning rate and $\epsilon$ is an additive constant that ensures that we do not divide by $0$. Last, we initialize $\mathbf{s}_0 = \mathbf{0}$.

Just like in the case of momentum we need to keep track of an auxiliary variable, in this case to allow for an individual learning rate per coordinate. This does not increase the cost of Adagrad significantly relative to SGD, simply since the main cost is typically to compute $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ and its derivative.

Note that accumulating squared gradients in $\mathbf{s}_t$ means that $\mathbf{s}_t$ grows essentially at linear rate (somewhat slower than linearly in practice, since the gradients initially diminish). This leads to an $\mathcal{O}(t^{-\frac{1}{2}})$ learning rate, albeit adjusted on a per coordinate basis. For convex problems this is perfectly adequate. In deep learning, though, we might want to decrease the learning rate rather more slowly. This led to a number of Adagrad variants that we will discuss in the subsequent chapters. For now let us see how it behaves in a quadratic convex problem. We use the same problem as before:

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

We are going to implement Adagrad using the same learning rate previously, i.e., $\eta = 0.4$. As we can see, the iterative trajectory of the independent variable is smoother. However, due to the cumulative effect of $\boldsymbol{s}_t$, the learning rate continuously decays, so the independent variable does not move as much during later stages of iteration.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

As we increase the learning rate to $2$ we see much better behavior. This already indicates that the decrease in learning rate might be rather aggressive, even in the noise-free case and we need to ensure that parameters converge appropriately.

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## Implementation from Scratch

Just like the momentum method, Adagrad needs to maintain a state variable of the same shape as the parameters.

```{.python .input}
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Compared to the experiment in :numref:`sec_minibatch_sgd` we use a
larger learning rate to train the model.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## Concise Implementation

Using the `Trainer` instance of the algorithm `adagrad`, we can invoke the Adagrad algorithm in Gluon.

```{.python .input}
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## Summary

* Adagrad decreases the learning rate dynamically on a per-coordinate basis.
* It uses the magnitude of the gradient as a means of adjusting how quickly progress is achieved - coordinates with large gradients are compensated with a smaller learning rate.
* Computing the exact second derivative is typically infeasible in deep learning problems due to memory and computational constraints. The gradient can be a useful proxy.
* If the optimization problem has a rather uneven structure Adagrad can help mitigate the distortion.
* Adagrad is particularly effective for sparse features where the learning rate needs to decrease more slowly for infrequently occurring terms.
* On deep learning problems Adagrad can sometimes be too aggressive in reducing learning rates. We will discuss strategies for mitigating this in the context of :numref:`sec_adam`.

## Exercises

1. Prove that for an orthogonal matrix $\mathbf{U}$ and a vector $\mathbf{c}$ the following holds: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. Why does this mean that the magnitude of perturbations does not change after an orthogonal change of variables?
1. Try out Adagrad for $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ and also for the objective function was rotated by 45 degrees, i.e., $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Does it behave differently?
1. Prove [Gerschgorin's circle theorem](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) which states that eigenvalues $\lambda_i$ of a matrix $\mathbf{M}$ satisfy $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$ for at least one choice of $j$.
1. What does Gerschgorin's theorem tell us about the eigenvalues of the diagonally preconditioned matrix $\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$?
1. Try out Adagrad for a proper deep network, such as :numref:`sec_lenet` when applied to Fashion MNIST.
1. How would you need to modify Adagrad to achieve a less aggressive decay in learning rate?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/355)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1072)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1073)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA5MDIwMzUwMCwxNTM2ODY3Mzc1LDE2NT
IxNDMwNjNdfQ==
-->
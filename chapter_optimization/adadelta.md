# Adadelta
:label:`sec_adadelta`

Adadelta é outra variante do AdaGrad (:numref:`sec_adagrad`). A principal diferença reside no fato de que diminui a quantidade pela qual a taxa de aprendizagem é adaptável às coordenadas. Além disso, tradicionalmente é referido como não tendo uma taxa de aprendizagem, uma vez que usa a quantidade de mudança em si como calibração para mudanças futuras. O algoritmo foi proposto em :cite:`Zeiler.2012`. É bastante simples, dada a discussão de algoritmos anteriores até agora.

## O Algoritmo

Em poucas palavras, Adadelta usa duas variáveis de estado, $\mathbf{s}_t$ para armazenar uma média de vazamento do segundo momento do gradiente e $\Delta\mathbf{x}_t$ para armazenar uma média de vazamento do segundo momento da mudança de parâmetros no próprio modelo. Observe que usamos a notação original e a nomenclatura dos autores para compatibilidade com outras publicações e implementações (não há outra razão real para usar variáveis gregas diferentes para indicar um parâmetro que serve ao mesmo propósito em momentum, Adagrad, RMSProp e Adadelta)

Aqui estão os detalhes técnicos do Adadelta. Dado que o parâmetro du jour é $\rho$, obtemos as seguintes atualizações vazadas de forma semelhante a :numref:`sec_rmsprop`:

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

A diferença para :numref:`sec_rmsprop` é que realizamos atualizações com o gradiente redimensionado $\mathbf{g}_t'$, ou seja,

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

Então, qual é o gradiente redimensionado $\mathbf{g}_t'$? Podemos calculá-lo da seguinte maneira:

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

onde $\Delta \mathbf{x}_{t-1}$ é a média de vazamento dos gradientes redimensionados ao quadrado $\mathbf{g}_t'$. Inicializamos $\Delta \mathbf{x}_{0}$ para ser $0$ e atualizamos em cada etapa com $\mathbf{g}_t'$, ou seja,

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

e $\epsilon$ (um pequeno valor como $10^{-5}$) é adicionado para manter a estabilidade numérica.


## Implementação

Adadelta precisa manter duas variáveis de estado para cada variável, $\mathbf{s}_t$ e $\Delta\mathbf{x}_t$. Isso produz a seguinte implementação.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

Escolher $\rho = 0,9$ equivale a um tempo de meia-vida de 10 para cada atualização de parâmetro. Isso tende a funcionar muito bem. Obtemos o seguinte comportamento.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

For a concise implementation we simply use the `adadelta` algorithm from the `Trainer` class. This yields the following one-liner for a much more compact invocation.

```{.python .input}
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it's converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## Summary

* Adadelta has no learning rate parameter. Instead, it uses the rate of change in the parameters itself to adapt the learning rate. 
* Adadelta requires two state variables to store the second moments of gradient and the change in parameters. 
* Adadelta uses leaky averages to keep a running estimate of the appropriate statistics. 

## Exercises

1. Adjust the value of $\rho$. What happens?
1. Show how to implement the algorithm without the use of $\mathbf{g}_t'$. Why might this be a good idea?
1. Is Adadelta really learning rate free? Could you find optimization problems that break Adadelta?
1. Compare Adadelta to Adagrad and RMS prop to discuss their convergence behavior.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1076)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1077)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTg0MjAxNjYwOV19
-->
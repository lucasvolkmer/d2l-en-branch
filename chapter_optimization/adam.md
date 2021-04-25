# Adam
:label:`sec_adam`

Nas discussões que levaram a esta seção, encontramos várias técnicas para otimização eficiente. Vamos recapitulá-los em detalhes aqui:

* Vimos que :numref:`sec_sgd` é mais eficaz do que Gradient Descent ao resolver problemas de otimização, por exemplo, devido à sua resiliência inerente a dados redundantes.
* Vimos que :numref:`sec_minibatch_sgd` proporciona eficiência adicional significativa decorrente da vetorização, usando conjuntos maiores de observações em um minibatch. Esta é a chave para um processamento paralelo eficiente em várias máquinas, várias GPUs e em geral.
* :numref:`sec_momentum` adicionado um mecanismo para agregar um histórico de gradientes anteriores para acelerar a convergência.
* :numref:`sec_adagrad` usado por escala de coordenada para permitir um pré-condicionador computacionalmente eficiente.
* :numref:`sec_rmsprop` desacoplado por escala de coordenada de um ajuste de taxa de aprendizagem.

Adam :cite:`Kingma.Ba.2014` combina todas essas técnicas em um algoritmo de aprendizagem eficiente. Como esperado, este é um algoritmo que se tornou bastante popular como um dos algoritmos de otimização mais robustos e eficazes para uso no aprendizado profundo. Não é sem problemas, no entanto. Em particular, :cite:`Reddi.Kale.Kumar.2019` mostra que há situações em que Adam pode divergir devido a um controle de variação insuficiente. Em um trabalho de acompanhamento :cite:`Zaheer.Reddi.Sachan.ea.2018` propôs um hotfix para Adam, chamado Yogi, que trata dessas questões. Mais sobre isso mais tarde. Por enquanto, vamos revisar o algoritmo de Adam.

## O Algoritmo

Um dos componentes principais de Adam é que ele usa médias móveis exponenciais ponderadas (também conhecidas como média com vazamento) para obter uma estimativa do momento e também do segundo momento do gradiente. Ou seja, ele usa as variáveis de estado

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

Aqui $\beta_1$ e $\beta_2$ são parâmetros de ponderação não negativos. As escolhas comuns para eles são $\beta_1 = 0.9$ e $\beta_2 = 0.999$. Ou seja, a estimativa da variância se move *muito mais lentamente* do que o termo de momentum. Observe que se inicializarmos $\mathbf{v}_0 = \mathbf{s}_0 = 0$, teremos uma quantidade significativa de tendência inicialmente para valores menores. Isso pode ser resolvido usando o fato de que $\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$ para normalizar os termos novamente. Correspondentemente, as variáveis de estado normalizadas são fornecidas por

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

Armados com as estimativas adequadas, podemos agora escrever as equações de atualização. Primeiro, nós redimensionamos o gradiente de uma maneira muito semelhante à do RMSProp para obter

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

Ao contrário de RMSProp, nossa atualização usa o momento $\hat{\mathbf{v}}_t$ em vez do gradiente em si. Além disso, há uma pequena diferença estética, pois o redimensionamento acontece usando $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ em vez de $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$. O primeiro funciona sem dúvida um pouco melhor na prática, daí o desvio de RMSProp. Normalmente escolhemos $\epsilon = 10^{-6}$ para uma boa troca entre estabilidade numérica e fidelidade.

Agora temos todas as peças no lugar para computar as atualizações. Isso é um pouco anticlimático e temos uma atualização simples do formulário

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Revendo o projeto de Adam, sua inspiração é clara. Momentum e escala são claramente visíveis nas variáveis de estado. Sua definição um tanto peculiar nos força a termos de debias (isso poderia ser corrigido por uma inicialização ligeiramente diferente e condição de atualização). Em segundo lugar, a combinação de ambos os termos é bastante direta, dado o RMSProp. Por último, a taxa de aprendizagem explícita $\eta$ nos permite controlar o comprimento do passo para tratar de questões de convergência.

## Implementação

Implementar Adam do zero não é muito assustador. Por conveniência, armazenamos o contador de intervalos de tempo $t$ no dicionário de `hiperparâmetros`. Além disso, tudo é simples.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Estamos prontos para usar Adam para treinar o modelo. Usamos uma taxa de aprendizado de $\eta = 0,01$.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

Uma implementação mais concisa é direta, pois `adam` é um dos algoritmos fornecidos como parte da biblioteca de otimização `trainer` Gluon. Portanto, só precisamos passar os parâmetros de configuração para uma implementação no Gluon.

```{.python .input}
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

Um dos problemas de Adam é que ele pode falhar em convergir mesmo em configurações convexas quando a estimativa do segundo momento em $\mathbf{s}_t$ explode. Como uma correção :cite:`Zaheer.Reddi.Sachan.ea.2018` propôs uma atualização refinada (e inicialização) para $\mathbf{s}_t$. Para entender o que está acontecendo, vamos reescrever a atualização do Adam da seguinte maneira:

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

Whenever $\mathbf{g}_t^2$ has high variance or updates are sparse, $\mathbf{s}_t$ might forget past values too quickly. A possible fix for this is to replace $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ by $\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$. Now the magnitude of the update no longer depends on the amount of deviation. This yields the Yogi updates

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

The authors furthermore advise to initialize the momentum on a larger initial batch rather than just initial pointwise estimate. We omit the details since they are not material to the discussion and since even without this convergence remains pretty good.

```{.python .input}
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## Summary

* Adam combines features of many optimization algorithms into a fairly robust update rule. 
* Created on the basis of RMSProp, Adam also uses EWMA on the minibatch stochastic gradient.
* Adam uses bias correction to adjust for a slow startup when estimating momentum and a second moment. 
* For gradients with significant variance we may encounter issues with convergence. They can be amended by using larger minibatches or by switching to an improved estimate for $\mathbf{s}_t$. Yogi offers such an alternative. 

## Exercises

1. Adjust the learning rate and observe and analyze the experimental results.
1. Can you rewrite momentum and second moment updates such that it does not require bias correction?
1. Why do you need to reduce the learning rate $\eta$ as we converge?
1. Try to construct a case for which Adam diverges and Yogi converges?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1078)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1079)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjEzODExMzUxXX0=
-->
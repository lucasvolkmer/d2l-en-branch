# Computação Assíncrona
:label:`sec_async`


Os computadores de hoje são sistemas altamente paralelos, consistindo em vários núcleos de CPU (geralmente várias *threads* por núcleo), vários elementos de processamento por GPU e, muitas vezes, várias GPUs por dispositivo. Resumindo, podemos processar muitas coisas diferentes ao mesmo tempo, geralmente em dispositivos diferentes. Infelizmente, Python não é uma ótima maneira de escrever código paralelo e assíncrono, pelo menos não com alguma ajuda extra. Afinal, o Python é de thread único e é improvável que isso mude no futuro. Estruturas de aprendizado profundo, como MXNet e TensorFlow, utilizam um modelo de programação assíncrona para melhorar o desempenho (o PyTorch usa o próprio programador do Python, levando a uma compensação de desempenho diferente).
Para PyTorch, por padrão, as operações de GPU são assíncronas. Quando você chama uma função que usa a GPU, as operações são enfileiradas no dispositivo específico, mas não necessariamente executadas até mais tarde. Isso nos permite executar mais cálculos em paralelo, incluindo operações na CPU ou outras GPUs.

Portanto, entender como a programação assíncrona funciona nos ajuda a desenvolver programas mais eficientes, reduzindo proativamente os requisitos computacionais e as dependências mútuas. Isso nos permite reduzir a sobrecarga de memória e aumentar a utilização do processador. Começamos importando as bibliotecas necessárias.

```{.python .input  n=1}
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=1}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
import numpy
```

## Assincronismo via *Back-end*

:begin_tab:`mxnet`
Para um aquecimento, considere o seguinte problema brinquedo - queremos gerar uma matriz aleatória e multiplicá-la. Vamos fazer isso no NumPy e no MXNet NP para ver a diferença.
:end_tab:

:begin_tab: `pytorch`
Para um aquecimento, considere o seguinte problema brinquedo - queremos gerar uma matriz aleatória e multiplicá-la. Vamos fazer isso tanto no NumPy quanto no tensor PyTorch para ver a diferença.
Observe que o `tensor` do PyTorch é definido em uma gpu.
:end_tab:

```{.python .input  n=2}
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input  n=2}
#@tab pytorch
# warmup for gpu computation
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

:begin_tab:`mxnet`
Isso é ordens de magnitude mais rápido. Pelo menos parece que sim. Uma vez que ambos são executados no mesmo processador, algo mais deve estar acontecendo. Forçar o MXNet a terminar toda a computação antes de retornar mostra o que aconteceu anteriormente: a computação está sendo executada pelo *back-end* enquanto o *front-end* retorna o controle ao Python.
:end_tab:

:begin_tab:`pytorch`
Isso é ordens de magnitude mais rápido. Pelo menos parece que sim.
O produto de ponto Numpy é executado no processador cpu enquanto
A multiplicação da matriz de Pytorch é executada no gpu e, portanto, o último
espera-se que seja muito mais rápida. Mas a enorme diferença de tempo sugere que algo mais deve estar acontecendo.
Por padrão, as operações da GPU são assíncronas no PyTorch.
Forçando PyTorch a terminar todos os cálculos antes de retornar os programas,
o que aconteceu anteriormente: o cálculo está sendo executado pelo backend
enquanto o front-end retorna o controle para Python.
:end_tab:

```{.python .input  n=3}
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input  n=3}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

:begin_tab:`mxnet`
De um modo geral, o MXNet possui um front-end para interação direta com os usuários, por exemplo, via Python, bem como um *back-end* usado pelo sistema para realizar a computação.
Conforme mostrado em: numref: `fig_frontends`, os usuários podem escrever programas MXNet em várias linguagens de front-end, como Python, R, Scala e C ++. Independentemente da linguagem de programação de front-end usada, a execução de programas MXNet ocorre principalmente no *back-end* de implementações C ++. As operações emitidas pela linguagem do front-end são passadas para o back-end para execução.
O back-end gerencia seus próprios threads que continuamente coletam e executam tarefas enfileiradas. Observe que, para que isso funcione, o *back-end* deve ser capaz de controlar as dependências entre as várias etapas do gráfico computacional. Portanto, não é possível paralelizar operações que dependem umas das outras.
:end_tab:

:begin_tab:`pytorch`
Em termos gerais, o PyTorch tem um *front-end* para interação direta com os usuários, por exemplo, via Python, bem como um *back-end* usado pelo sistema para realizar a computação.
Conforme mostrado em: numref: `fig_frontends`, os usuários podem escrever programas PyTorch em várias linguagens de *front-end*, como Python e C ++. Independentemente da linguagem de programação de frontend usada, a execução de programas PyTorch ocorre principalmente no backend de implementações C ++. As operações emitidas pela linguagem do *front-end* são passadas para o *back-end* para execução.
O *back-end* gerencia suas próprias threads que continuamente coletam e executam tarefas enfileiradas.
Observe que para que isso funcione, o *back-end* deve ser capaz de rastrear as
dependências entre várias etapas no gráfico computacional.
Portanto, não é possível paralelizar operações que dependem umas das outras.
:end_tab:


![Programação *Frontend*.](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

Vejamos outro exemplo brinquedo para entender um pouco melhor o grafo de dependência.

```{.python .input  n=4}
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input  n=4}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

![Dependências.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`

O trecho de código acima também é ilustrado em :numref:`fig_asyncgraph`.
Sempre que a *thread* de *front-end* do Python executa uma das três primeiras instruções, ela simplesmente retorna a tarefa para a fila de *back-end*. Quando os resultados da última instrução precisam ser impressos, a *thread* de *front-end* do Python irá esperar que a*thread* de *back-end* do C ++ termine de calcular o resultado da variável `z`. Um benefício desse *design* é que a *thread* de *front-end* do Python não precisa realizar cálculos reais. Portanto, há pouco impacto no desempenho geral do programa, independentemente do desempenho do Python. :numref:`fig_threading`  ilustra como *front-end* e *back-end* interagem.

![Frontend and Backend.](../img/threading.svg)
:label:`fig_threading`


## Barreiras e Bloqueadores


Existem várias operações que forçam o Python a aguardar a conclusão:
* Obviamente, `npx.waitall ()` espera até que todo o cálculo seja concluído, independentemente de quando as instruções de cálculo foram emitidas. Na prática, é uma má ideia usar este operador, a menos que seja absolutamente necessário, pois pode levar a um desempenho insatisfatório.
* Se quisermos apenas esperar até que uma variável específica esteja disponível, podemos chamar `z.wait_to_read ()`. Nesse caso, os blocos MXNet retornam ao Python até que a variável `z` seja calculada. Outros cálculos podem continuar depois.

Vamos ver como isso funciona na prática:

```{.python .input  n=5}
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

Both operations take approximately the same time to complete. Besides the obvious blocking operations we recommend that the reader is aware of *implicit* blockers. Printing a variable clearly requires the variable to be available and is thus a blocker. Lastly, conversions to NumPy via `z.asnumpy()` and conversions to scalars via `z.item()` are blocking, since NumPy has no notion of asynchrony. It needs access to the values just like the `print` function. Copying small amounts of data frequently from MXNet's scope to NumPy and back can destroy performance of an otherwise efficient code, since each such operation requires the computational graph to evaluate all intermediate results needed to get the relevant term *before* anything else can be done.

```{.python .input  n=7}
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

## Improving Computation

On a heavily multithreaded system (even regular laptops have 4 threads or more and on multi-socket servers this number can exceed 256) the overhead of scheduling operations can become significant. This is why it is highly desirable to have computation and scheduling occur asynchronously and in parallel. To illustrate the benefit of doing this let us see what happens if we increment a variable by 1 multiple times, both in sequence or asynchronously. We simulate synchronous execution by inserting a `wait_to_read()` barrier in between each addition.

```{.python .input  n=9}
with d2l.Benchmark('synchronous'):
    for _ in range(1000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(1000):
        y = x + 1
    y.wait_to_read()
```

A slightly simplified interaction between the Python frontend thread and the C++ backend thread can be summarized as follows:

1. The frontend orders the backend to insert the calculation task `y = x + 1` into the queue.
1. The backend then receives the computation tasks from the queue and performs the actual computations.
1. The backend then returns the computation results to the frontend.

Assume that the durations of these three stages are $t_1, t_2$ and $t_3$, respectively. If we do not use asynchronous programming, the total time taken to perform 1000 computations is approximately $1000 (t_1+ t_2 + t_3)$. If asynchronous programming is used, the total time taken to perform 1000 computations can be reduced to $t_1 + 1000 t_2 + t_3$ (assuming $1000 t_2 > 999t_1$), since the frontend does not have to wait for the backend to return computation results for each loop.

## Improving Memory Footprint

Imagine a situation where we keep on inserting operations into the backend by executing Python code on the frontend. For instance, the frontend might insert a large number of minibatch tasks within a very short time. After all, if no meaningful computation happens in Python this can be done quite quickly. If each of these tasks can be launched quickly at the same time this may cause a spike in memory usage. Given a finite amount of memory available on GPUs (and even on CPUs) this can lead to resource contention or even program crashes. Some readers might have noticed that previous training routines made use of synchronization methods such as `item` or even `asnumpy`.

We recommend to use these operations carefully, e.g., for each minibatch, such as to balance computational efficiency and memory footprint. To illustrate what happens let us implement a simple training loop for a deep network and measure its memory consumption and timing. Below is the mock data generator and deep network.

```{.python .input  n=10}
def data_iter():
    timer = d2l.Timer()
    num_batches, batch_size = 150, 1024
    for i in range(num_batches):
        X = np.random.normal(size=(batch_size, 512))
        y = np.ones((batch_size,))
        yield X, y
        if (i + 1) % 50 == 0:
            print(f'batch {i + 1}, time {timer.stop():.4f} sec')

net = nn.Sequential()
net.add(nn.Dense(2048, activation='relu'),
        nn.Dense(512, activation='relu'), nn.Dense(1))
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd')
loss = gluon.loss.L2Loss()
```

Next we need a tool to measure the memory footprint of our code. We use a relatively primitive `ps` call to accomplish this (note that the latter only works on Linux and MacOS). For a much more detailed analysis of what is going on here use e.g., Nvidia's [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) or Intel's [vTune](https://software.intel.com/en-us/vtune).

```{.python .input  n=12}
def get_mem():
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15]) / 1e3
```

Before we can begin testing we need to initialize the parameters of the network and process one batch. Otherwise it would be tricky to see what the additional memory consumption is. See :numref:`sec_deferred_init` for further details related to initialization.

```{.python .input  n=13}
for X, y in data_iter():
    break
loss(y, net(X)).wait_to_read()
```

To ensure that we do not overflow the task buffer on the backend we insert a `wait_to_read` call for the loss function at the end of each loop. This forces the forward propagation to complete before a new forward propagation is commenced. Note that a (possibly more elegant) alternative would have been to track the loss in a scalar variable and to force a barrier via the `item` call.

```{.python .input  n=14}
mem = get_mem()
with d2l.Benchmark('time per epoch'):
    for X, y in data_iter():
        with autograd.record():
            l = loss(y, net(X))
        l.backward()
        trainer.step(X.shape[0])
        l.wait_to_read()  # Barrier before a new batch
    npx.waitall()
print(f'increased memory: {get_mem() - mem:f} MB')
```

As we see, the timing of the minibatches lines up quite nicely with the overall runtime of the optimization code. Moreover, memory footprint only increases slightly. Now let us see what happens if we drop the barrier at the end of each minibatch.

```{.python .input  n=14}
mem = get_mem()
with d2l.Benchmark('time per epoch'):
    for X, y in data_iter():
        with autograd.record():
            l = loss(y, net(X))
        l.backward()
        trainer.step(X.shape[0])
    npx.waitall()
print(f'increased memory: {get_mem() - mem:f} MB')
```

Even though the time to issue instructions for the backend is an order of magnitude smaller, we still need to perform computation. Consequently a large amount of intermediate results cannot be released and may pile up in memory. While this didn't cause any issues in the toy example above, it might well have resulted in out of memory situations when left unchecked in real world scenarios.

## Summary

* MXNet decouples the Python frontend from an execution backend. This allows for fast asynchronous insertion of commands into the backend and associated parallelism.
* Asynchrony leads to a rather responsive frontend. However, use caution not to overfill the task queue since it may lead to excessive memory consumption.
* It is recommended to synchronize for each minibatch to keep frontend and backend approximately synchronized.
* Be aware of the fact that conversions from MXNet's memory management to Python will force the backend to wait until  the specific variable is ready. `print`, `asnumpy` and `item` all have this effect. This can be desirable but a carless use of synchronization can ruin performance.
* Chip vendors offer sophisticated performance analysis tools to obtain a much more fine-grained insight into the efficiency of deep learning.


## Exercises

1. We mentioned above that using asynchronous computation can reduce the total amount of time needed to perform $1000$ computations to $t_1 + 1000 t_2 + t_3$. Why do we have to assume $1000 t_2 > 999 t_1$ here?
1. How would you need to modify the training loop if you wanted to have an overlap of one minibatch each? I.e., if you wanted to ensure that batch $b_t$ finishes before batch $b_{t+2}$ commences?
1. What might happen if we want to execute code on CPUs and GPUs simultaneously? Should you still insist on synchronizing after every minibatch has been issued?
1. Measure the difference between `waitall` and `wait_to_read`. Hint: perform a number of instructions and synchronize for an intermediate result.


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/361)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI0NjExODQ1MywxMzk2MTIwMTk3LDE3Mz
Q5MTYxNjldfQ==
-->
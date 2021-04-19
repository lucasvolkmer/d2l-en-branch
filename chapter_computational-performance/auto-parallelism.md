# Paralelismo Automático
:label:`sec_auto_para`

:begin_tab:`mxnet`
O MXNet constrói automatically constructmente gráficos computatcional graphs at the is no *back-end*. Using a
ando um
gráfico computatcional graph, the, o syistem is aware of all thea está ciente de todas as dependeêncieas,
and cane pode executar selectively execute multiple non-amente várias tarefas não interdependent tasks ines em parallel to
improve speedo para
melhorar a velocidade. FPor instanceexemplo, :numref:`fig_asyncgraph` inem :numref:`sec_async` initcializes twoa duas variableáveis independentlyemente. Consequently theemente, o syistem can choose to execute them ina pode optar por executá-las em parallelo.
:end_tab:

:begin_tab:`pytorch`
O PyTorch constrói automaticamente gráficos computacionais no *back-end*. Usando um gráfico computacional, o sistema está ciente de todas as dependências e pode executar seletivamente várias tarefas não interdependentes em paralelo para melhorar a velocidade. Por exemplo, :numref:`fig_asyncgraph` em :numref:`sec_async` inicializa duas variáveis independentemente. Consequentemente, o sistema pode optar por executá-las em paralelo.
:end_tab:

Normalmente, um único operador usará todos os recursos computacionais em todas as CPUs ou em uma única GPU. Por exemplo, o operador `dot` usará todos os núcleos (e threads) em todas as CPUs, mesmo se houver vários processadores de CPU em uma única máquina. O mesmo se aplica a uma única GPU. Consequentemente, a paralelização não é tão útil em computadores de dispositivo único. Com vários dispositivos, as coisas são mais importantes. Embora a paralelização seja normalmente mais relevante entre várias GPUs, adicionar a CPU local aumentará um pouco o desempenho. Veja, por exemplo, :cite:`Hadjis.Zhang.Mitliagkas.ea.2016` para um artigo que se concentra no treinamento de modelos de visão computacional combinando uma GPU e uma CPU. Com a conveniência de uma estrutura de paralelização automática, podemos atingir o mesmo objetivo em algumas linhas de código Python. De forma mais ampla, nossa discussão sobre computação paralela automática concentra-se na computação paralela usando CPUs e GPUs, bem como a paralelização de computação e comunicação.
Começamos importando os pacotes e módulos necessários. Observe que precisamos de pelo menos duas GPUs para executar os experimentos nesta seção.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## Computação Paralela em GPUs

Vamos começar definindo uma carga de trabalho de referência para testar - a função `run` abaixo realiza 10 multiplicações matriz-matriz no dispositivo de nossa escolha usando dados alocados em duas variáveis,` x_gpu1` e `x_gpu2`.

```{.python .input}
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```


:begin_tab:`mxnet`
Agora aplicamos a função aos dados. Para garantir que o cache não desempenhe um papel nos resultados, aquecemos os dispositivos realizando uma única passagem em cada um deles antes da medição.
:end_tab:

:begin_tab:`pytorch`
Agora aplicamos a função aos dados. Para garantir que o cache não desempenhe um papel nos resultados, aquecemos os dispositivos realizando uma única passagem em cada um deles antes da medição. `torch.cuda.synchronize ()` espera que todos os kernels em todos os streams em um dispositivo CUDA sejam concluídos. Ele recebe um argumento `device`, o dispositivo para o qual precisamos sincronizar. Ele usa o dispositivo atual, fornecido por `current_device ()`, se o argumento do dispositivo for `None` (padrão).
:end_tab:


```{.python .input}
run(x_gpu1)  # Warm-up both devices
run(x_gpu2)
npx.waitall()  

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Warm-up all devices
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU 1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU 2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
Se removermos o `waitall ()` entre as duas tarefas, o sistema fica livre para paralelizar a computação em ambos os dispositivos automaticamente.
:end_tab:

:begin_tab:`pytorch`
Se removermos `torch.cuda.synchronize ()` entre as duas tarefas, o sistema fica livre para paralelizar a computação em ambos os dispositivos automaticamente.
:end_tab:

```{.python .input}
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
No caso acima, o tempo total de execução é menor que a soma de suas partes, uma vez que o MXNet programa automaticamente a computação em ambos os dispositivos GPU sem a necessidade de um código sofisticado em nome do usuário.
:end_tab:

:begin_tab:`pytorch`
No caso acima, o tempo total de execução é menor que a soma de suas partes, uma vez que o PyTorch programa automaticamente a computação em ambos os dispositivos GPU sem a necessidade de um código sofisticado em nome do usuário.
:end_tab:

## Computação Paralela e Comunicação

Em muitos casos, precisamos mover dados entre diferentes dispositivos, digamos, entre CPU e GPU, ou entre diferentes GPUs. Isso ocorre, por exemplo, quando queremos realizar a otimização distribuída onde precisamos agregar os gradientes em vários cartões aceleradores. Vamos simular isso computando na GPU e copiando os resultados de volta para a CPU.

```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
Isso é um tanto ineficiente. Observe que já podemos começar a copiar partes de `y` para a CPU enquanto o restante da lista ainda está sendo calculado. Essa situação ocorre, por exemplo, quando calculamos o gradiente (*backprop*) em um minibatch. Os gradientes de alguns dos parâmetros estarão disponíveis antes dos outros. Portanto, é vantajoso começar a usar a largura de banda do barramento PCI-Express enquanto a GPU ainda está em execução. Remover `waitall` entre as duas partes nos permite simular este cenário.
:end_tab:

:begin_tab:`pytorch`
This is somewhat inefficient. Note that we could already start copying parts of `y` to the CPU while the remainder of the list is still being computed. This situation occurs, e.g., when we compute the (backprop) gradient on a minibatch. The gradients of some of the parameters will be available earlier than that of others. Hence it works to our advantage to start using PCI-Express bus bandwidth while the GPU is still running. In PyTorch, several functions such as `to()` and `copy_()` admit an explicit `non_blocking` argument, which lets the caller bypass synchronization when it is unnecessary. Setting `non_blocking=True` allows us to simulate this scenario.
:end_tab:


```{.python .input}
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

The total time required for both operations is (as expected) significantly less than the sum of their parts. Note that this task is different from parallel computation as it uses a different resource: the bus between CPU and GPUs. In fact, we could compute on both devices and communicate, all at the same time. As noted above, there is a dependency between computation and communication: `y[i]` must be computed before it can be copied to the CPU. Fortunately, the system can copy `y[i-1]` while computing `y[i]` to reduce the total running time.

We conclude with an illustration of the computational graph and its dependencies for a simple two-layer MLP when training on a CPU and two GPUs, as depicted in :numref:`fig_twogpu`. It would be quite painful to schedule the parallel program resulting from this manually. This is where it is advantageous to have a graph based compute backend for optimization.

![Two layer MLP on a CPU and 2 GPUs.](../img/twogpu.svg)
:label:`fig_twogpu`


## Summary

* Modern systems have a variety of devices, such as multiple GPUs and CPUs. They can be used in parallel, asynchronously. 
* Modern systems also have a variety of resources for communication, such as PCI Express, storage (typically SSD or via network), and network bandwidth. They can be used in parallel for peak efficiency. 
* The backend can improve performance through through automatic parallel computation and communication. 

## Exercises

1. 10 operations were performed in the `run` function defined in this section. There are no dependencies between them. Design an experiment to see if MXNet will automatically execute them in parallel.
1. When the workload of an individual operator is sufficiently small, parallelization can help even on a single CPU or GPU. Design an experiment to verify this. 
1. Design an experiment that uses parallel computation on CPU, GPU and communication between both devices.
1. Use a debugger such as NVIDIA's Nsight to verify that your code is efficient. 
1. Designing computation tasks that include more complex data dependencies, and run experiments to see if you can obtain the correct results while improving performance.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1681)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyMjQ5NzYwMTksMTg0NDUwMzcxM119
-->
# Servidores de Parâmetros
:label:`sec_parameterserver`


À medida que mudamos de GPUs únicas para várias GPUs e depois para vários servidores contendo várias GPUs, possivelmente todos espalhadas por vários racks e switches de rede, nossos algoritmos para treinamento distribuído e paralelo precisam se tornar muito mais sofisticados. Os detalhes são importantes, já que diferentes interconexões têm larguras de banda muito diferentes (por exemplo, NVLink pode oferecer até 100 GB/s em 6 links em uma configuração apropriada, PCIe 3.0 16x lanes oferecem 16 GB/s, enquanto mesmo Ethernet de 100 GbE de alta velocidade atinge apenas 10 GB/s) . Ao mesmo tempo, não é razoável esperar que um modelador estatístico seja um especialista em redes e sistemas.

A ideia central do servidor de parâmetros foi introduzida em :cite:`Smola.Narayanamurthy.2010` no contexto de modelos de variáveis ​​latentes distribuídas. Uma descrição da semântica push e pull seguida em :cite:`Ahmed.Aly.Gonzalez.ea.2012` e uma descrição do sistema e uma biblioteca de código aberto seguida em :cite:`Li.Andersen.Park.ea.2014`. A seguir, iremos motivar os componentes necessários para a eficiência.

## Treinamento Paralelo de Dados

Vamos revisar a abordagem de treinamento paralelo de dados para treinamento distribuído. Usaremos isso com a exclusão de todos os outros nesta seção, uma vez que é significativamente mais simples de implementar na prática. Praticamente não há casos de uso (além do aprendizado profundo em gráficos) onde qualquer outra estratégia de paralelismo é preferida, já que as GPUs têm muita memória hoje em dia. :numref:`fig_parameterserver` descreve a variante de paralelismo de dados que implementamos na seção anterior. O aspecto principal nisso é que a agregação de gradientes ocorre na GPU0 antes que os parâmetros atualizados sejam retransmitidos para todas as GPUs.

![Na esquerda: treinamento de GPU único; Na direita: uma variante do treinamento multi-GPU. Ele procede da seguinte maneira. (1) calculamos a perda e o gradiente, (2) todos os gradientes são agregados em uma GPU, (3) a atualização dos parâmetros acontece e os parâmetros são redistribuídos para todas as GPUs.](../img/ps.svg)
:label:`fig_parameterserver`

Em retrospecto, a decisão de agregar na GPU0 parece bastante ad-hoc. Afinal, podemos muito bem agregar na CPU. Na verdade, poderíamos até decidir agregar alguns dos parâmetros em uma GPU e alguns outros em outra. Desde que o algoritmo de otimização suporte isso, não há razão real para que não o possamos fazer. Por exemplo, se tivermos quatro vetores de parâmetro $\mathbf{v}_1, \ldots, \mathbf{v}_4$ com gradientes associados $\mathbf{g}_1, \ldots, \mathbf{g}_4$ poderíamos agreguar os gradientes em uma GPU cada.

$$\mathbf{g}_{i} = \sum_{j \in \mathrm{GPUs}} \mathbf{g}_{ij}$$

Esse raciocínio parece arbitrário e frívolo. Afinal, a matemática é a mesma do começo ao fim. No entanto, estamos lidando com *hardware* físico real onde diferentes barramentos têm diferentes larguras de banda, conforme discutido em :numref:`sec_hardware`. Considere um servidor GPU real de 4 vias conforme descrito em :numref:`fig_bw_hierarchy`. Se estiver bem conectado, pode ter uma placa de rede 100 GbE. Os números mais comuns estão na faixa de 1-10 GbE com uma largura de banda efetiva de 100 MB/s a 1 GB/s. Uma vez que as CPUs têm poucas pistas PCIe para se conectar a todas as GPUs diretamente (por exemplo, CPUs da Intel para consumidores têm 24 pistas), precisamos de um [multiplexador](https://www.broadcom.com/products/pcie-switches-bridges/ chaves pcie). A largura de banda da CPU em um link Gen3 16x é de 16 GB/s. Esta também é a velocidade na qual *cada* uma das GPUs é conectada ao *switch*. Isso significa que é mais eficaz a comunicação entre os dispositivos.

![Um servidor GPU de 4 vias.](../img/bw-hierarchy.svg)
:label:`fig_bw_hierarchy`

Para o efeito do argumento, vamos supor que os gradientes 'pesam' 160 MB. Nesse caso, leva 30ms para enviar os gradientes de todas as 3 GPUs restantes para a quarta (cada transferência leva 10 ms = 160 MB / 16 GB/s). Adicione mais 30ms para transmitir os vetores de peso de volta, chegamos a um total de 60ms.
Se enviarmos todos os dados para a CPU, incorremos em uma penalidade de 40ms, pois *cada* uma das quatro GPUs precisa enviar os dados para a CPU, resultando em um total de 80ms. Por último, suponha que somos capazes de dividir os gradientes em 4 partes de 40 MB cada. Agora podemos agregar cada uma das partes em uma GPU *diferente simultaneamente*, já que o switch PCIe oferece uma operação de largura de banda total entre todos os links. Em vez de 30 ms, isso leva 7,5 ms, resultando em um total de 15 ms para uma operação de sincronização. Resumindo, dependendo de como sincronizamos os parâmetros, a mesma operação pode levar de 15ms a 80ms. :numref:`fig_ps_distributed` descreve as diferentes estratégias para a troca de parâmetros.

![Estratégias de sincronização.](../img/ps-distributed.svg)
:label:`fig_ps_distributed`

Observe que temos mais uma ferramenta à nossa disposição quando se trata de melhorar o desempenho: em uma rede profunda, leva algum tempo para calcular todos os gradientes de cima para baixo. Podemos começar a sincronizar gradientes para alguns grupos de parâmetros, mesmo enquanto ainda estamos ocupados computando-os para outros (os detalhes técnicos disso estão um tanto envolvidos). Veja, por exemplo, :cite:`Sergeev.Del-Balso.2018` para obter detalhes sobre como fazer isso em [Horovod](https://github.com/horovod/horovod).

## Sincronização em Anel

Quando se trata de sincronização em *hardware* de *deep learning* moderno, frequentemente encontramos conectividade de rede significativamente personalizada. Por exemplo, as instâncias AWS P3.16xlarge e NVIDIA DGX-2 compartilham a estrutura de conectividade de :numref:`fig_nvlink`. Cada GPU se conecta a uma CPU host por meio de um link PCIe que opera no máximo a 16 GB/s. Além disso, cada GPU também possui 6 conexões NVLink, cada uma das quais é capaz de transferir 300 Gbit/s bidirecionalmente. Isso equivale a cerca de 18 GB/s por link por direção. Resumindo, a largura de banda NVLink agregada é significativamente maior do que a largura de banda PCIe. A questão é como usá-lo de forma mais eficiente.

![Conectividade NVLink em servidores 8GPU V100 (imagem cortesia da NVIDIA).](../img/nvlink.svg)
:label:`fig_nvlink`

Acontece que :cite:`Wang.Li.Liberty.ea.2018` a estratégia de sincronização ideal é decompor a rede em dois anéis e usá-los para sincronizar os dados diretamente. :numref:`fig_nvlink_twoloop` ilustra que a rede pode ser decomposta em um anel (1-2-3-4-5-6-7-8-1) com largura de banda NVLink dupla e em um (1-4-6-3 -5-8-2-7-1) com largura de banda regular. Projetar um protocolo de sincronização eficiente nesse caso não é trivial.

![Decomposição da rede NVLink em dois anéis.](../img/nvlink-twoloop.svg)
:label:`fig_nvlink_twoloop`

Considere o seguinte experimento de pensamento: dado um anel de $n$  nós de computação (ou GPUs), podemos enviar gradientes do primeiro para o segundo nó. Lá, ele é adicionado ao gradiente local e enviado para o terceiro nó e assim por diante. Após $n-1$ passos, o gradiente agregado pode ser encontrado no último nó visitado. Ou seja, o tempo para agregar gradientes cresce linearmente com o número de nós. Mas, se fizermos isso, o algoritmo será bastante ineficiente. Afinal, a qualquer momento, há apenas um dos nós se comunicando. E se quebrássemos os gradientes em $n$ pedaços e começássemos a sincronizar o pedaço $i$ começando no nó $i$. Como cada pedaço tem o tamanho $1/n$, o tempo total agora é $(n-1)/n \approx 1$ Em outras palavras, o tempo gasto para agregar gradientes *não aumenta* à medida que aumentamos o tamanho do anel. Este é um resultado surpreendente. :numref:`fig_ringsync` ilustra a sequência de etapas em $n=4$ nodes.

![Sincronização de anel em 4 nós. Cada nó começa a transmitir partes de gradientes para seu vizinho esquerdo até que o gradiente montado possa ser encontrado em seu vizinho direito.](../img/ringsync.svg)
:label:`fig_ringsync`

Se usarmos o mesmo exemplo de sincronização de 160 MB em 8 GPUs V100, chegaremos a aproximadamente $2 \cdot 160 \mathrm{MB} / (3 \cdot 18 \mathrm{GB/s}) \approx 6 \mathrm{ms}$. Isto é um pouco melhor do que usar o barramento PCIe, embora agora estejamos usando 8 GPUs. Observe que, na prática, esses números são um pouco piores, uma vez que os *frameworks* de aprendizado profundo geralmente falham em reunir a comunicação em grandes transferências de burst. Além disso, o tempo é crítico.
Observe que há um equívoco comum de que a sincronização em anel é fundamentalmente diferente de outros algoritmos de sincronização. A única diferença é que o caminho de sincronização é um pouco mais elaborado quando comparado a uma árvore simples.

## Treinamento Multi-Máquina

Distributed training on multiple machines adds a further challenge: we need to communicate with servers that are only connected across a comparatively lower bandwidth fabric which can be over an order of magnitude slower in some cases. Synchronization across devices is tricky. After all, different machines running training code will have subtly different speed. Hence we need to *synchronize* them if we want to use synchronous distributed optimization. :numref:`fig_ps_multimachine` illustrates how distributed parallel training occurs.

1. A (different) batch of data is read on each machine, split across multiple GPUs and transferred to GPU memory. There predictions and gradients are computed on each GPU batch separately.
2. The gradients from all local GPUs are aggregated on one GPU (or alternatively parts of it are aggregated over different GPUs.
3. The gradients are sent to the CPU.
4. The CPU sends the gradients to a central parameter server which aggregates all the gradients.
5. The aggregate gradients are then used to update the weight vectors and the updated weight vectors are broadcast back to the individual CPUs.
6. The information is sent to one (or multiple) GPUs.
7. The updated weight vectors are spread across all GPUs.

![Multi-machine multi-GPU distributed parallel training.](../img/ps-multimachine.svg)
:label:`fig_ps_multimachine`

Each of these operations seems rather straightforward. And, indeed, they can be carried out efficiently *within* a single machine. Once we look at multiple machines, though, we can see that the central parameter server becomes the bottleneck. After all, the bandwidth per server is limited, hence for $m$ workers the time it takes to send all gradients to the server is $O(m)$. We can break through this barrier by increasing the number of servers to $n$. At this point each server only needs to store $O(1/n)$ of the parameters, hence the total time for updates and optimization becomes $O(m/n)$. Matching both numbers yields constant scaling regardless of how many workers we are dealing with. In practice we use the *same* machines both as workers and as servers. :numref:`fig_ps_multips` illustrates the design. See also :cite:`Li.Andersen.Park.ea.2014` for details. In particular, ensuring that multiple machines work without unreasonable delays is nontrivial. We omit details on barriers and will only briefly touch on synchronous and asynchronous updates below.

![Top - a single parameter server is a bottleneck since its bandwidth is finite. Bottom - multiple parameter servers store parts of the parameters with aggregate bandwidth.](../img/ps-multips.svg)
:label:`fig_ps_multips`

## (key,value) Stores

Implementing the steps required for distributed multi-GPU training in practice is nontrivial. In particular, given the many different choices that we might encounter. This is why it pays to use a common abstraction, namely that of a (key,value) store with redefined update semantics. Across many servers and many GPUs the gradient computation can be defined as

$$\mathbf{g}_{i} = \sum_{k \in \mathrm{workers}} \sum_{j \in \mathrm{GPUs}} \mathbf{g}_{ijk}.$$

The key aspect in this operation is that it is a *commutative reduction*, that is, it turns many vectors into one and the order in which the operation is applied does not matter. This is great for our purposes since we do not (need to) have fine grained control over when which gradient is received. Note that it is possible for us to perform the reduction stagewise. Furthermore, note that this operation is independent between blocks $i$ pertaining to different parameters (and gradients).

This allows us to define the following two operations: push, which accumulates gradients, and pull, which retrieves aggregate gradients. Since we have many different sets of gradients (after all, we have many layers), we need to index the gradients with a key $i$. This similarity to (key,value) stores, such as the one introduced in Dynamo
:cite:`DeCandia.Hastorun.Jampani.ea.2007` is not by coincidence. They, too, satisfy many similar characteristics, in particular when it comes to distributing the parameters across multiple servers.

* **push(key, value)** sends a particular gradient (the value) from a worker to a common storage. There the parameter is aggregated, e.g., by summing it up.
* **pull(key, value)** retrieves an aggregate parameter from common storage, e.g., after combining the gradients from all workers.

By hiding all the complexity about synchronization behind a simple push and pull operation we can decouple the concerns of the statistical modeler who wants to be able to express optimization in simple terms and the systems engineer who needs to deal with the complexity inherent in distributed synchronization. In the next section we will experiment with such a (key,value) store in practice.

## Summary

* Synchronization needs to be highly adaptive to specific network infrastructure and connectivity within a server. This can make a significant difference to the time it takes to synchronize.
* Ring-synchronization can be optimal for P3 and DGX-2 servers. For others possibly not so much.
* A hierarchical synchronization strategy works well when adding multiple parameter servers for increased bandwidth.
* Asynchronous communication (while computation is still ongoing) can improve performance.

## Exercises

1. Can you increase the ring synchronization even further? Hint: you can send messages in both directions.
1. Fully asynchronous. Some delays permitted?
1. Fault tolerance. How? What if we lose a server? Is this a problem?
1. Checkpointing
1. Tree aggregation. Can you do it faster?
1. Other reductions (commutative semiring).

[Discussions](https://discuss.d2l.ai/t/366)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI2MjI5MjI3MywtOTYzMjY4NDM5LC0xOD
UwMjgxMjQ1LC0xODc1NzQ5NDkwXX0=
-->
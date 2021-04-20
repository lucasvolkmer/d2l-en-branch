# *Hardware*
:label:`sec_hardware`

Construir sistemas com ótimo desempenho requer um bom entendimento dos algoritmos e modelos para capturar os aspectos estatísticos do problema. Ao mesmo tempo, também é indispensável ter pelo menos um mínimo de conhecimento do *hardware* subjacente. A seção atual não substitui um curso apropriado sobre design de *hardware* e sistemas. Em vez disso, pode servir como um ponto de partida para entender por que alguns algoritmos são mais eficientes do que outros e como obter um bom rendimento. Um bom design pode facilmente fazer uma diferença de ordem de magnitude e, por sua vez, pode fazer a diferença entre ser capaz de treinar uma rede (por exemplo, em uma semana) ou não (em 3 meses, perdendo o prazo) . Começaremos examinando os computadores. Em seguida, daremos zoom para observar com mais cuidado as CPUs e GPUs. Por último, diminuímos o zoom para revisar como vários computadores estão conectados em um centro de servidores ou na nuvem. Este não é um guia de compra de GPU. Para isto, veja :numref:`sec_buy_gpu`. Uma introdução à computação em nuvem com AWS pode ser encontrada em :numref:`sec_aws`.

Leitores impacientes podem se virar com :numref:`fig_latencynumbers`. Ele foi retirado da [postagem interativa](https://people.eecs.berkeley.edu/~rcs/research/interactive_latency.html) de Colin Scott , que oferece uma boa visão geral do progresso na última década. Os números originais são devidos à  [palestra de Stanford de 2010](https://static.googleusercontent.com/media/research.google.com/en//people/jeff/Stanford-DL-Nov-2010.pdf) de Jeff Dean. A discussão abaixo explica algumas das razões para esses números e como eles podem nos guiar no projeto de algoritmos. A discussão abaixo é de alto nível e superficial. Claramente, *não substitui* um curso adequado, mas apenas fornece informações suficientes para que um modelador estatístico tome decisões de projeto adequadas. Para uma visão geral aprofundada da arquitetura de computadores, recomendamos ao leitor :cite:`Hennessy.Patterson.2011` ou um curso recente sobre o assunto, como o de [Arste Asanovic](http://inst.eecs.berkeley.edu/~cs152/sp19/).

![Números de latência que todo programador deve conhecer.](../img/latencynumbers.png)
:label:`fig_latencynumbers`

## Computadores


A maioria dos pesquisadores de aprendizagem profunda tem acesso a um computador com uma boa quantidade de memória, computação, alguma forma de acelerador, como uma GPU, ou múltiplos deles. Consiste em vários componentes principais:

* Um processador, também conhecido como CPU, que é capaz de executar os programas que fornecemos (além de executar um sistema operacional e muitas outras coisas), normalmente consistindo de 8 ou mais núcleos.
* Memória (RAM) para armazenar e recuperar os resultados da computação, como vetores de peso, ativações, geralmente dados de treinamento.
* Uma conexão de rede Ethernet (às vezes múltipla) com velocidades que variam de 1 Gbit / s a ​​100 Gbit / s (em servidores de ponta podem ser encontradas interconexões mais avançadas).
* Um barramento de expansão de alta velocidade (PCIe) para conectar o sistema a uma ou mais GPUs. Os servidores têm até 8 aceleradores, geralmente conectados em uma topologia avançada, os sistemas desktop têm 1-2, dependendo do orçamento do usuário e do tamanho da fonte de alimentação.
* O armazenamento durável, como um disco rígido magnético (HDD), estado sólido (SSD), em muitos casos conectado usando o barramento PCIe, fornece transferência eficiente de dados de treinamento para o sistema e armazenamento de pontos de verificação intermediários conforme necessário.

![Conectividade de componentes](../img/mobo-symbol.svg)
:label:`fig_mobo-symbol`


Como :numref:`fig_mobo-symbol` indica, a maioria dos componentes (rede, GPU, armazenamento) são conectados à CPU através do barramento PCI Express. Ele consiste em várias pistas diretamente conectadas à CPU. Por exemplo, o Threadripper 3 da AMD tem 64 pistas PCIe 4.0, cada uma das quais é capaz de transferência de dados de 16 Gbit/s em ambas as direções. A memória é conectada diretamente à CPU com uma largura de banda total de até 100 GB/s.

Quando executamos o código em um computador, precisamos embaralhar os dados para os processadores (CPU ou GPU), realizarem cálculos e, em seguida, mover os resultados do processador de volta para a RAM e armazenamento durável. Portanto, para obter um bom desempenho, precisamos nos certificar de que isso funcione perfeitamente, sem que nenhum dos sistemas se torne um grande gargalo. Por exemplo, se não conseguirmos carregar as imagens com rapidez suficiente, o processador não terá nenhum trabalho a fazer. Da mesma forma, se não conseguirmos mover as matrizes com rapidez suficiente para a CPU (ou GPU), seus elementos de processamento morrerão de fome. Finalmente, se quisermos sincronizar vários computadores na rede, o último não deve retardar a computação. Uma opção é intercalar comunicação e computação. Vamos dar uma olhada nos vários componentes com mais detalhes.

## Memória


Em sua forma mais básica, a memória é usada para armazenar dados que precisam estar prontamente acessíveis. Atualmente, a CPU RAM é tipicamente da variedade [DDR4](https://en.wikipedia.org/wiki/DDR4_SDRAM), oferecendo largura de banda de 20-25GB/s por módulo. Cada módulo possui um barramento de 64 bits. Normalmente, pares de módulos de memória são usados ​​para permitir vários canais. As CPUs têm entre 2 e 4 canais de memória, ou seja, têm entre 40 GB/s e 100 GB/s de largura de banda de memória de pico. Freqüentemente, existem dois bancos por canal. Por exemplo, o Zen 3 *Threadripper* da AMD tem 8 slots.

Embora esses números sejam impressionantes, na verdade, eles contam apenas parte da história. Quando queremos ler uma parte da memória, primeiro precisamos dizer ao módulo de memória onde a informação pode ser encontrada. Ou seja, primeiro precisamos enviar o *endereço* para a RAM. Feito isso, podemos escolher ler apenas um único registro de 64 bits ou uma longa sequência de registros. Este último é chamado de *leitura intermitente*. Resumindo, enviar um endereço para a memória e configurar a transferência leva aproximadamente 100 ns (os detalhes dependem dos coeficientes de tempo específicos dos chips de memória usados), cada transferência subsequente leva apenas 0,2 ns. Resumindo, a primeira leitura é 500 vezes mais cara que as subsequentes! Poderíamos realizar até $10.000.000$ leituras aleatórias por segundo. Isso sugere que evitamos o acesso aleatório à memória o máximo possível e, em vez disso, usamos leituras (e gravações) intermitentes.

As coisas ficam um pouco mais complexas quando levamos em consideração que temos vários bancos. Cada banco pode ler a memória independentemente. Isso significa duas coisas: o número efetivo de leituras aleatórias é até 4x maior, desde que sejam distribuídas uniformemente pela memória. Isso também significa que ainda é uma má ideia realizar leituras aleatórias, uma vez que as leituras intermitentes também são 4x mais rápidas. Em segundo lugar, devido ao alinhamento da memória aos limites de 64 bits, é uma boa ideia alinhar quaisquer estruturas de dados com os mesmos limites. Compiladores fazem isso [automaticamente](https://en.wikipedia.org/wiki/Data_structure_alignment) quando os sinalizadores apropriados são definidos. Os leitores curiosos são incentivados a revisar uma palestra sobre DRAMs, como a de [Zeshan Chishti](http://web.cecs.pdx.edu/~zeshan/ece585_lec5.pdf).

A memória da GPU está sujeita a requisitos de largura de banda ainda maiores, uma vez que têm muito mais elementos de processamento do que CPUs. Em geral, existem duas opções para resolvê-los. Uma é tornar o barramento de memória significativamente mais amplo. Por exemplo, o RTX 2080 Ti da NVIDIA tem um barramento de 352 bits. Isso permite que muito mais informações sejam transferidas ao mesmo tempo. Em segundo lugar, as GPUs usam memória específica de alto desempenho. Dispositivos de consumo, como as séries RTX e Titan da NVIDIA, geralmente usam chips [GDDR6](https://en.wikipedia.org/wiki/GDDR6_SDRAM) com largura de banda agregada superior a 500 GB / s. Uma alternativa é usar módulos HBM (memória de alta largura de banda). Eles usam uma interface muito diferente e se conectam diretamente com GPUs em um *wafer* de silício dedicado. Isso os torna muito caros e seu uso é normalmente limitado a chips de servidor de ponta, como a série de aceleradores NVIDIA Volta V100. Sem surpresa, a memória da GPU é *muito* menor do que a memória da CPU devido ao seu custo mais alto. Para nossos propósitos, em geral, suas características de desempenho são semelhantes, apenas muito mais rápidas. Podemos ignorar com segurança os detalhes para o propósito deste livro. Eles só importam ao ajustar os kernels da GPU para alto rendimento.

## Armazenamento

Vimos que algumas das principais características da RAM eram *largura de banda* e *latência*. O mesmo é verdade para dispositivos de armazenamento, só que as diferenças podem ser ainda mais extremas.


**Discos rígidos** são usados ​​há mais de meio século. Em poucas palavras, eles contêm uma série de pratos giratórios com cabeças que podem ser posicionadas para ler / escrever em qualquer faixa. Discos de última geração suportam até 16 TB em 9 pratos. Um dos principais benefícios dos HDDs é que eles são relativamente baratos. Uma de suas muitas desvantagens são seus modos de falha tipicamente catastróficos e sua latência de leitura relativamente alta.

Para entender o último, considere o fato de que os HDDs giram em torno de 7.200 RPM. Se fossem muito mais rápidos, estilhaçariam devido à força centrífuga exercida nas travessas. Isso tem uma grande desvantagem quando se trata de acessar um setor específico no disco: precisamos esperar até que o prato tenha girado na posição (podemos mover as cabeças, mas não acelerar os discos reais). Portanto, pode demorar mais de 8 ms até que os dados solicitados estejam disponíveis. Uma maneira comum de expressar isso é dizer que os HDDs podem operar a aproximadamente 100 IOPs. Este número permaneceu essencialmente inalterado nas últimas duas décadas. Pior ainda, é igualmente difícil aumentar a largura de banda (é da ordem de 100-200 MB/s). Afinal, cada cabeçote lê uma trilha de bits, portanto, a taxa de bits é dimensionada apenas com a raiz quadrada da densidade da informação. Como resultado, os HDDs estão rapidamente se tornando relegados ao armazenamento de arquivos e armazenamento de baixo nível para conjuntos de dados muito grandes.

**Solid State Drives** use Flash memory to store information persistently. This allows for *much faster* access to stored records. Modern SSDs can operate at 100,000 to 500,000 IOPs, i.e., up to 3 orders of magnitude faster than HDDs. Furthermore, their bandwidth can reach 1-3GB/s, i.e., one order of magnitude faster than HDDs. These improvements sound almost too good to be true. Indeed, they come with a number of caveats, due to the way SSDs are designed.

* SSDs store information in blocks (256 KB or larger). They can only be written as a whole, which takes significant time. Consequently bit-wise random writes on SSD have very poor performance. Likewise, writing data in general takes significant time since the block has to be read, erased and then rewritten with new information. By now SSD controllers and firmware have developed algorithms to mitigate this. Nonetheless writes can be much slower, in particular for QLC (quad level cell) SSDs. The key for improved performance is to maintain a *queue* of operations, to prefer reads and to write in large blocks if possible.
* The memory cells in SSDs wear out relatively quickly (often already after a few thousand writes). Wear-level protection algorithms are able to spread the degradation over many cells. That said, it is not recommended to use SSDs for swap files or for large aggregations of log-files.
* Lastly, the massive increase in bandwidth has forced computer designers to attach SSDs directly to the PCIe bus. The drives capable of handling this, referred to as NVMe (Non Volatile Memory enhanced), can use up to 4 PCIe lanes. This amounts to up to 8GB/s on PCIe 4.0.

**Cloud Storage** provides a configurable range of performance. That is, the assignment of storage to virtual machines is dynamic, both in terms of quantity and in terms speed, as chosen by the user. We recommend that the user increase the provisioned number of IOPs whenever latency is too high, e.g., during training with many small records.

## CPUs

Central Processing Units (CPUs) are the centerpiece of any computer (as before we give a very high level description focusing primarily on what matters for efficient deep learning models). They consist of a number of key components: processor cores which are able to execute machine code, a bus connecting them (the specific topology differs significantly between processor models, generations and vendors), and caches to allow for higher bandwidth and lower latency memory access than what is possible by reads from main memory. Lastly, almost all modern CPUs contain vector processing units to aid with high performance linear algebra and convolutions, as they are common in media processing and machine learning.

![Intel Skylake consumer quad-core CPU](../img/skylake.svg)
:label:`fig_skylake`

:numref:`fig_skylake` depicts an Intel Skylake consumer grade quad-core CPU. It has an integrated GPU, caches, and a ringbus connecting the four cores. Peripherals (Ethernet, WiFi, Bluetooth, SSD controller, USB, etc.) are either part of the chipset or directly attached (PCIe) to the CPU.

### Microarchitecture

Each of the processor cores consists of a rather sophisticated set of components. While details differ between generations and vendors, the basic functionality is pretty much standard. The front end loads instructions and tries to predict which path will be taken (e.g., for control flow). Instructions are then decoded from assembly code to microinstructions. Assembly code is often not the lowest level code that a processor executes. Instead, complex instructions may be decoded into a set of more lower level operations. These are then processed by the actual execution core. Often the latter is capable of performing many operations simultaneously. For instance, the ARM Cortex A77 core of :numref:`fig_cortexa77` is able to perform up to 8 operations simultaneously.

![ARM Cortex A77 Microarchitecture Overview](../img/a77.svg)
:label:`fig_cortexa77`

This means that efficient programs might be able to perform more than one instruction per clock cycle, *provided that* they can be carried out independently. Not all units are created equal. Some specialize in integer instructions whereas others are optimized for floating point performance. To increase throughput, the processor might also follow  multiple codepaths simultaneously in a branching instruction and then discard the results of the branches not taken. This is why branch prediction units matter (on the frontend) such that only the most promising paths are pursued.

### Vectorization

Deep learning is extremely compute hungry. Hence, to make CPUs suitable for machine learning, one needs to perform many operations in one clock cycle. This is achieved via vector units. They have different names: on ARM they are called NEON, on x86 the latest generation is referred to as [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) units. A common aspect is that they are able to perform SIMD (single instruction multiple data) operations. :numref:`fig_neon128` shows how 8 short integers can be added in one clock cycle on ARM.

![128 bit NEON vectorization](../img/neon128.svg)
:label:`fig_neon128`

Depending on architecture choices, such registers are up to 512 bit long, allowing for the combination of up to 64 pairs of numbers. For instance, we might be multiplying two numbers and adding them to a third, which is also known as a fused multiply-add. Intel's [OpenVino](https://01.org/openvinotoolkit) uses these to achieve respectable throughput for deep learning on server grade CPUs. Note, though, that this number is entirely dwarved by what GPUs are capable of achieving. For instance, NVIDIA's RTX 2080 Ti has 4,352 CUDA cores, each of which is capable of processing such an operation at any time.

### Cache

Consider the following situation: we have a modest CPU core with 4 cores as depicted in :numref:`fig_skylake` above, running at 2GHz frequency. Moreover, let us assume that we have an IPC (instructions per clock) count of 1 and that the units have AVX2 with 256bit width enabled. Let us furthermore assume that at least one of the registers used for AVX2 operations needs to be retrieved from memory. This means that the CPU consumes 4x256bit = 1kbit of data per clock cycle. Unless we are able to transfer $2 \cdot 10^9 \cdot 128 = 256 \cdot 10^9$ bytes to the processor per second the processing elements are going to starve. Unfortunately the memory interface of such a chip only supports 20-40 GB/s data transfer, i.e., one order of magnitude less. The fix is to avoid loading *new* data from memory as far as possible and rather to cache it locally on the CPU. This is where caches come in handy (see this [Wikipedia article](https://en.wikipedia.org/wiki/Cache_hierarchy) for a primer). Commonly the following names / concepts are used:

* **Registers** are strictly speaking not part of the cache. They help stage instructions. That said, CPU registers are memory locations that a CPU can access at clock speed without any delay penalty. CPUs have tens of registers. It is up to the compiler (or programmer) to use registers efficiently. For instance the C programming language has a `register` keyword.
* **L1** caches are the first line of defense against high memory bandwidth requirements. L1 caches are tiny (typical sizes might be 32-64kB) and often split into data and instructions caches. When data is found in the L1 cache access is very fast. If it cannot be found there, the search progresses down the cache hierarchy.
* **L2** caches are the next stop. Depending on architecture design and processor size they might be exclusive. They might be accessible only by a given core or shared between multiple cores. L2 caches are larger (typically 256-512kB per core) and slower than L1. Furthermore, to access something in L2 we first need to check to realize that the data is not in L1, which adds a small amount of extra latency.
* **L3** caches are shared between multiple cores and can be quite large. AMD's Epyc 3 server CPUs have a whopping 256MB of cache spread across multiple chiplets. More typical numbers are in the 4-8MB range.

Predicting which memory elements will be needed next is one of the key optimization parameters in chip design. For instance, it is advisable to traverse memory in a *forward* direction since most caching algorithms will try to *read ahead* rather than backwards. Likewise, keeping memory access patterns local is a good way of improving performance.
Adding caches is a double-edge sword. On one hand they ensure that the processor cores do not starve of data. At the same time they increase chip size, using up area that otherwise could have been spent on increasing processing power. Moreover, *cache misses* can be expensive. Consider the worst case scenario, depicted in :numref:`fig_falsesharing`. A memory location is cached on processor 0 when a thread on processor 1 requests the data. To obtain it, processor 0 needs to stop what it is doing, write the information back to main memory and then let processor 1 read it from memory. During this operation both processors wait. Quite potentially such code runs *more slowly* on multiple processors when compared to an efficient single-processor implementation. This is one more reason for why there is a practical limit to cache sizes (besides their physical size).

![False sharing (image courtesy of Intel)](../img/falsesharing.svg)
:label:`fig_falsesharing`

## GPUs and other Accelerators

It is not an exaggeration to claim that deep learning would not have been successful without GPUs. By the same token, it is quite reasonable to argue that GPU manufacturers' fortunes have been increased significantly due to deep learning. This co-evolution of hardware and algorithms has led to a situation where for better or worse deep learning is the preferable statistical modeling paradigm. Hence it pays to understand the specific benefits that GPUs and related accelerators such as the TPU :cite:`Jouppi.Young.Patil.ea.2017` offer.

Of note is a distinction that is often made in practice: accelerators are optimized either for training or inference. For the latter we only need to compute the forward propagation in a network. No storage of intermediate data is needed for backpropagation. Moreover, we may not need very precise computation (FP16 or INT8 typically suffice). On the other hand, during training all intermediate results need storing to compute gradients. Moreover, accumulating gradients requires higher precision to avoid numerical underflow (or overflow). This means that FP16 (or mixed precision with FP32) is the minimum required. All of this necessitates faster and larger memory (HBM2 vs. GDDR6) and more processing power. For instance, NVIDIA's [Turing](https://devblogs.nvidia.com/nvidia-turing-architecture-in-depth/) T4 GPUs are optimized for inference whereas the V100 GPUs are preferable for training.

Recall :numref:`fig_neon128`. Adding vector units to a processor core allowed us to increase throughput significantly (in the example in the figure we were able to perform 16 operations simultaneously). What if we added operations that optimized not just operations between vectors but also between matrices? This strategy led to Tensor Cores (more on this shortly). Secondly, what if we added many more cores? In a nutshell, these two strategies summarize the design decisions in GPUs. :numref:`fig_turing_processing_block` gives an overview over a basic processing block. It contains 16 integer and 16 floating point units. In addition to that, two Tensor Cores accelerate a narrow subset of additional operations relevant for deep learning. Each Streaming Multiprocessor (SM) consists of four such blocks.

![NVIDIA Turing Processing Block (image courtesy of NVIDIA)](../img/turing-processing-block.png)
:width:`150px`
:label:`fig_turing_processing_block`

12 streaming multiprocessors are then grouped into graphics processing clusters which make up the high-end TU102 processors. Ample memory channels and an L2 cache complement the setup. :numref:`fig_turing` has the relevant details. One of the reasons for designing such a device is that individual blocks can be added or removed as needed to allow for more compact chips and to deal with yield issues (faulty modules might not be activated). Fortunately programming such devices is well hidden from the casual deep learning researcher beneath layers of CUDA and framework code. In particular, more than one of the programs might well be executed simultaneously on the GPU, provided that there are available resources. Nonetheless it pays to be aware of the limitations of the devices to avoid picking models that do not fit into device memory.

![NVIDIA Turing Architecture (image courtesy of NVIDIA)](../img/turing.png)
:width:`350px`
:label:`fig_turing`

A last aspect that is worth mentioning in more detail are TensorCores. They are an example of a recent trend of adding more optimized circuits that are specifically effective for deep learning. For instance, the TPU added a systolic array :cite:`Kung.1988` for fast matrix multiplication. There the design was to support a very small number (one for the first generation of TPUs) of large operations. TensorCores are at the other end. They are optimized for small operations involving between 4x4 and 16x16 matrices, depending on their numerical precision. :numref:`fig_tensorcore` gives an overview of the optimizations.

![NVIDIA TensorCores in Turing (image courtesy of NVIDIA)](../img/tensorcore.jpg)
:width:`400px`
:label:`fig_tensorcore`

Obviously when optimizing for computation we end up making certain compromises. One of them is that GPUs are not very good at handling interrupts and sparse data. While there are notable exceptions, such as [Gunrock](https://github.com/gunrock/gunrock) :cite:`Wang.Davidson.Pan.ea.2016`, the access pattern of sparse matrices and vectors do not go well with the high bandwidth burst read operations where GPUs excel. Matching both goals is an area of active research. See e.g., [DGL](http://dgl.ai), a library tuned for deep learning on graphs.

## Networks and Buses

Whenever a single device is insufficient for optimization we need to transfer data to and from it to synchronize processing. This is where networks and buses come in handy. We have a number of design parameters: bandwidth, cost, distance and flexibility. On one end we have WiFi which has a pretty good range, is very easy to use (no wires, after all), cheap but it offers comparatively mediocre bandwidth and latency. No machine learning researcher within their right mind would use it to build a cluster of servers. In what follows we focus on interconnects that are suitable for deep learning.

* **PCIe** is a dedicated bus for very high bandwidth point to point connections (up to 16 Gbs on PCIe 4.0) per lane. Latency is in the order of single-digit microseconds (5 μs). PCIe links are precious. Processors only have a limited number of them: AMD's EPYC 3 has 128 lanes, Intel's Xeon has up to 48 lanes per chip; on desktop grade CPUs the numbers are 20 (Ryzen 9) and 16 (Core i9) respectively. Since GPUs have typically 16 lanes this limits the number of GPUs that can connect to the CPU at full bandwidth. After all, they need to share the links with other high bandwidth peripherals such as storage and Ethernet. Just like with RAM access, large bulk transfers are preferable due to reduced packet overhead.
* **Ethernet** is the most commonly used way of connecting computers. While it is significantly slower than PCIe, it is very cheap and resilient to install and covers much longer distances. Typical bandwidth for low-grade servers is 1 GBit/s. Higher end devices (e.g., [C5 instances](https://aws.amazon.com/ec2/instance-types/c5/) in the cloud) offer between 10 and 100 GBit/s bandwidth. As in all previous cases data transmission has significant overheads. Note that we almost never use raw Ethernet directly but rather a protocol that is executed on top of the physical interconnect (such as UDP or TCP/IP). This adds further overhead. Like PCIe, Ethernet is designed to connect two devices, e.g., a computer and a switch.
* **Switches** allow us to connect multiple devices in a manner where any pair of them can carry out a (typically full bandwidth) point to point connection simultaneously. For instance, Ethernet switches might connect 40 servers at high cross-sectional bandwidth. Note that switches are not unique to traditional computer networks. Even PCIe lanes can be [switched](https://www.broadcom.com/products/pcie-switches-bridges/pcie-switches). This occurs e.g., to connect a large number of GPUs to a host processor, as is the case for the [P2 instances](https://aws.amazon.com/ec2/instance-types/p2/).
* **NVLink** is an alternative to PCIe when it comes to very high bandwidth interconnects. It offers up to 300 Gbit/s data transfer rate per link. Server GPUs (Volta V100) have 6 links whereas consumer grade GPUs (RTX 2080 Ti) have only one link, operating at a reduced 100 Gbit/s rate. We recommend to use [NCCL](https://github.com/NVIDIA/nccl) to achieve high data transfer between GPUs.

## Summary

* Devices have overheads for operations. Hence it is important to aim for a small number of large transfers rather than many small ones. This applies to RAM, SSDs, Networks and GPUs.
* Vectorization is key for performance. Make sure you are aware of the specific abilities of your accelerator. E.g., some Intel Xeon CPUs are particularly good for INT8 operations, NVIDIA Volta GPUs excel at FP16 matrix-matrix operations and NVIDIA Turing shines at FP16, INT8 and INT4 operations.
* Numerical overflow due to small datatypes can be a problem during training (and to a lesser extent during inference).
* Aliasing can significantly degrade performance. For instance, memory alignment on 64 bit CPUs should be done with respect to 64 bit boundaries. On GPUs it is a good idea to keep convolution sizes aligned e.g., to TensorCores.
* Match your algorithms to the hardware (memory footprint, bandwidth, etc.). Great speedup (orders of magnitude) can be achieved when fitting the parameters into caches.
* We recommend that you sketch out the performance of a novel algorithm on paper before verifying the experimental results. Discrepancies of an order-of-magnitude or more are reasons for concern.
* Use profilers to debug performance bottlenecks.
* Training and inference hardware have different sweet spots in terms of price / performance.

## More Latency Numbers

The summary in :numref:`table_latency_numbers` and :numref:`table_latency_numbers_tesla` are due to [Eliot Eshelman](https://gist.github.com/eshelman) who maintains an updated version of the numbers as a [GitHub Gist](https://gist.github.com/eshelman/343a1c46cb3fba142c1afdcdeec17646).

:Common Latency Numbers.

| Action | Time | Notes |
| :----------------------------------------- | -----: | :---------------------------------------------- |
| L1 cache reference/hit                     | 1.5 ns | 4 cycles                                        |
| Floating-point add/mult/FMA                | 1.5 ns | 4 cycles                                        |
| L2 cache reference/hit                     |   5 ns | 12 ~ 17 cycles                                  |
| Branch mispredict                          |   6 ns | 15 ~ 20 cycles                                  |
| L3 cache hit (unshared cache)              |  16 ns | 42 cycles                                       |
| L3 cache hit (shared in another core)      |  25 ns | 65 cycles                                       |
| Mutex lock/unlock                          |  25 ns |                                                 |
| L3 cache hit (modified in another core)    |  29 ns | 75 cycles                                       |
| L3 cache hit (on a remote CPU socket)      |  40 ns | 100 ~ 300 cycles (40 ~ 116 ns)                  |
| QPI hop to a another CPU (per hop)         |  40 ns |                                                 |
| 64MB memory ref. (local CPU)          |  46 ns | TinyMemBench on Broadwell E5-2690v4             |
| 64MB memory ref. (remote CPU)         |  70 ns | TinyMemBench on Broadwell E5-2690v4             |
| 256MB memory ref. (local CPU)         |  75 ns | TinyMemBench on Broadwell E5-2690v4             |
| Intel Optane random write                  |  94 ns | UCSD Non-Volatile Systems Lab                   |
| 256MB memory ref. (remote CPU)        | 120 ns | TinyMemBench on Broadwell E5-2690v4             |
| Intel Optane random read                   | 305 ns | UCSD Non-Volatile Systems Lab                   |
| Send 4KB over 100 Gbps HPC fabric          |   1 μs | MVAPICH2 over Intel Omni-Path                   |
| Compress 1KB with Google Snappy            |   3 μs |                                                 |
| Send 4KB over 10 Gbps ethernet             |  10 μs |                                                 |
| Write 4KB randomly to NVMe SSD             |  30 μs | DC P3608 NVMe SSD (QOS 99% is 500μs)            |
| Transfer 1MB to/from NVLink GPU            |  30 μs | ~33GB/s on NVIDIA 40GB NVLink                 |
| Transfer 1MB to/from PCI-E GPU             |  80 μs | ~12GB/s on PCIe 3.0 x16 link                  |
| Read 4KB randomly from NVMe SSD            | 120 μs | DC P3608 NVMe SSD (QOS 99%)                     |
| Read 1MB sequentially from NVMe SSD        | 208 μs | ~4.8GB/s DC P3608 NVMe SSD                    |
| Write 4KB randomly to SATA SSD             | 500 μs | DC S3510 SATA SSD (QOS 99.9%)                   |
| Read 4KB randomly from SATA SSD            | 500 μs | DC S3510 SATA SSD (QOS 99.9%)                   |
| Round trip within same datacenter          | 500 μs | One-way ping is ~250μs                          |
| Read 1MB sequentially from SATA SSD        |   2 ms | ~550MB/s DC S3510 SATA SSD                    |
| Read 1MB sequentially from disk            |   5 ms | ~200MB/s server HDD                           |
| Random Disk Access (seek+rotation)         |  10 ms |                                                 |
| Send packet CA->Netherlands->CA            | 150 ms |                                                 |
:label:`table_latency_numbers`

:Latency Numbers for NVIDIA Tesla GPUs.

| Action | Time | Notes |
| :------------------------------ | -----: | :---------------------------------------- |
| GPU Shared Memory access        |  30 ns | 30~90 cycles (bank conflicts add latency) |
| GPU Global Memory access        | 200 ns | 200~800 cycles                            |
| Launch CUDA kernel on GPU       |  10 μs | Host CPU instructs GPU to start kernel    |
| Transfer 1MB to/from NVLink GPU |  30 μs | ~33GB/s on NVIDIA 40GB NVLink           |
| Transfer 1MB to/from PCI-E GPU  |  80 μs | ~12GB/s on PCI-Express x16 link         |
:label:`table_latency_numbers_tesla`

## Exercises

1. Write C code to test whether there is any difference in speed between accessing memory aligned or misaligned relative to the external memory interface. Hint: be careful of caching effects.
1. Test the difference in speed between accessing memory in sequence or with a given stride.
1. How could you measure the cache sizes on a CPU?
1. How would you lay out data across multiple memory channels for maximum bandwidth? How would you lay it out if you had many small threads?
1. An enterprise class HDD is spinning at 10,000 rpm. What is the absolutely minimum time an HDD needs to spend worst case before it can read data (you can assume that heads move almost instantaneously)? Why are 2.5" HDDs becoming popular for commercial servers (relative to 3.5" and 5.25" drives)?
1. Assume that an HDD manufacturer increases the storage density from 1 Tbit per square inch to 5 Tbit per square inch. How much information can you store on a ring on a 2.5" HDD? Is there a difference between the inner and outer tracks?
1. The AWS P2 instances have 16 K80 Kepler GPUs. Use `lspci` on a p2.16xlarge and a p2.8xlarge instance to understand how the GPUs are connected to the CPUs. Hint: keep your eye out for PCI PLX bridges.
1. Going from 8 bit to 16 bit datatypes increases the amount of silicon approximately by 4x. Why? Why might NVIDIA have added INT4 operations to their Turing GPUs.
1. Given 6 high speed links between GPUs (such as for the Volta V100 GPUs), how would you connect 8 of them? Look up the connectivity used in the P3.16xlarge servers.
1. How much faster is it to read forward through memory vs. reading backwards? Does this number differ between different computers and CPU vendors? Why? Write C code and experiment with it.
1. Can you measure the cache size of your disk? What is it for a typical HDD? Do SSDs need a cache?
1. Measure the packet overhead when sending messages across the Ethernet. Look up the difference between UDP and TCP/IP connections.
1. Direct Memory Access allows devices other than the CPU to write (and read) directly to (from) memory. Why is this a good idea?
1. Look at the performance numbers for the Turing T4 GPU. Why does the performance 'only' double as you go from FP16 to INT8 and INT4?
1. What is the shortest time it should take for a packet on a roundtrip between San Francisco and Amsterdam? Hint: you can assume that the distance is 10,000km.


[Discussions](https://discuss.d2l.ai/t/363)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU2NjA3Nzk3OSwxNjkwMTMxNDIsLTcyMD
czOTcxNiwtNTY0NjgwMDQ4LC01NzIxOTA1NTYsMjQyOTQ5MzEy
XX0=
-->
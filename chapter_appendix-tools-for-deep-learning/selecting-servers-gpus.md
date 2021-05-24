# Seleção de servidores e GPUs
:label:`sec_buy_gpu`

O treinamento de aprendizado profundo geralmente requer grande quantidade de computação. Atualmente, as GPUs são os aceleradores de hardware mais econômicos para aprendizado profundo. Em particular, em comparação com CPUs, as GPUs são mais baratas e oferecem melhor desempenho, muitas vezes em mais de uma ordem de magnitude. Além disso, um único servidor pode suportar várias GPUs, até 8 para servidores de ponta. Os números mais comuns são até 4 GPUs para uma estação de trabalho de engenharia, uma vez que os requisitos de aquecimento, refrigeração e energia aumentam rapidamente além do que um prédio de escritórios pode suportar. Para implantações maiores de computação em nuvem, como Amazon [P3](https://aws.amazon.com/ec2/instance-types/p3/) e [G4](https://aws.amazon.com/blogs/aws As instâncias de / in-the-works-ec2-instances-g4-with-nvidia-t4-gpus /) são uma solução muito mais prática.

## Selecionando Servidores


Normalmente, não há necessidade de comprar CPUs de última geração com muitos threads, pois grande parte da computação ocorre nas GPUs. Dito isso, devido ao Global Interpreter Lock (GIL) no Python, o desempenho de thread único de uma CPU pode ser importante em situações em que temos de 4 a 8 GPUs. Tudo igual, isso sugere que CPUs com um número menor de núcleos, mas uma frequência de clock maior, pode ser uma escolha mais econômica. Por exemplo, ao escolher entre uma CPU de 4 GHz de 6 núcleos e uma CPU de 3,5 GHz de 8 núcleos, a primeira é muito preferível, embora sua velocidade agregada seja menor.
Uma consideração importante é que as GPUs usam muita energia e, portanto, dissipam muito calor. Isso requer um resfriamento muito bom e um chassi grande o suficiente para usar as GPUs. Siga as diretrizes abaixo, se possível:

1. **Fonte de alimentação**. As GPUs usam uma quantidade significativa de energia. Faça um orçamento de até 350 W por dispositivo (verifique o *pico de demanda* da placa de vídeo ao invés da demanda típica, já que um código eficiente pode consumir muita energia). Se sua fonte de alimentação não atender à demanda, você verá que o sistema se tornará instável.
1. **Tamanho do chassi**. As GPUs são grandes e os conectores de alimentação auxiliares geralmente precisam de espaço extra. Além disso, chassis grandes são mais fáceis de resfriar.
1. **Resfriamento da GPU**. Se você tiver um grande número de GPUs, talvez queira investir em refrigeração líquida. Além disso, busque *designs de referência* mesmo que tenham menos ventiladores, já que são finos o suficiente para permitir a entrada de ar entre os dispositivos. Se você comprar uma GPU com vários ventiladores, ela pode ser muito grossa para obter ar suficiente ao instalar várias GPUs e você terá um estrangulamento térmico.
1. **Slots PCIe**. Mover dados de e para a GPU (e trocá-los entre as GPUs) requer muita largura de banda. Recomendamos slots PCIe 3.0 com 16 pistas. Se você montar várias GPUs, certifique-se de ler cuidadosamente a descrição da placa-mãe para garantir que a largura de banda 16x ainda esteja disponível quando várias GPUs são usadas ao mesmo tempo e que você está obtendo PCIe 3.0 em vez de PCIe 2.0 para os slots adicionais. Algumas placas-mãe fazem downgrade para largura de banda de 8x ou até 4x com várias GPUs instaladas. Isso se deve em parte ao número de pistas PCIe que a CPU oferece.

Resumindo, aqui estão algumas recomendações para construir um servidor de aprendizado profundo:

* **Principiante**. Compre uma GPU de baixo custo com baixo consumo de energia (GPUs de jogos baratos adequados para uso de aprendizado profundo 150-200W). Se você tiver sorte, seu computador atual irá suportá-lo.
* ** 1 GPU **. Uma CPU de baixo custo com 4 núcleos será suficiente e a maioria das placas-mãe será suficiente. Procure ter pelo menos 32 GB de DRAM e invista em um SSD para acesso local aos dados. Uma fonte de alimentação com 600W deve ser suficiente. Compre uma GPU com muitos fãs.
* ** 2 GPUs **. Uma CPU low-end com 4-6 núcleos será suficiente. Procure por DRAM de 64 GB e invista em um SSD. Você precisará da ordem de 1000 W para duas GPUs de última geração. Em termos de placas-mãe, certifique-se de que elas tenham * dois * slots PCIe 3.0 x16. Se puder, compre uma placa-mãe com dois espaços livres (espaçamento de 60 mm) entre os slots PCIe 3.0 x16 para ar extra. Nesse caso, compre duas GPUs com muitos ventiladores.
* ** 4 GPUs **. Certifique-se de comprar uma CPU com velocidade de thread único relativamente rápida (ou seja, alta frequência de clock). Você provavelmente precisará de uma CPU com um número maior de pistas PCIe, como um AMD Threadripper. Você provavelmente precisará de placas-mãe relativamente caras para obter 4 slots PCIe 3.0 x16, pois eles provavelmente precisam de um PLX para multiplexar as pistas PCIe. Compre GPUs com design de referência que sejam estreitas e deixe o ar entrar entre as GPUs. Você precisa de uma fonte de alimentação de 1600-2000W e a tomada em seu escritório pode não suportar isso. Este servidor provavelmente irá rodar *alto e alto *. Você não o quer debaixo de sua mesa. 128 GB de DRAM é recomendado. Obtenha um SSD (1-2 TB NVMe) para armazenamento local e vários discos rígidos em configuração RAID para armazenar seus dados.
* ** 8 GPUs **. Você precisa comprar um chassi de servidor multi-GPU dedicado com várias fontes de alimentação redundantes (por exemplo, 2 + 1 para 1600 W por fonte de alimentação). Isso exigirá CPUs de servidor de soquete duplo, 256 GB ECC DRAM, uma placa de rede rápida (10 GBE recomendado) e você precisará verificar se os servidores suportam o * formato físico * das GPUs. O fluxo de ar e a colocação da fiação diferem significativamente entre as GPUs do consumidor e do servidor (por exemplo, RTX 2080 vs. Tesla V100). Isso significa que você pode não conseguir instalar a GPU do consumidor em um servidor devido à folga insuficiente para o cabo de alimentação ou à falta de um chicote de fiação adequado (como um dos co-autores descobriu dolorosamente).
## Selecting GPUs

At present, AMD and NVIDIA are the two main manufacturers of dedicated GPUs. NVIDIA was the first to enter the deep learning field and provides better support for deep learning frameworks via CUDA. Therefore, most buyers choose NVIDIA GPUs.

NVIDIA provides two types of GPUs, targeting individual users (e.g., via the GTX and RTX series) and enterprise users (via its Tesla series). The two types of GPUs provide comparable compute power. However, the enterprise user GPUs generally use (passive) forced cooling, more memory, and ECC (error correcting) memory. These GPUs are more suitable for data centers and usually cost ten times more than consumer GPUs.

If you are a large company with 100+ servers you should consider the NVIDIA Tesla series or alternatively use GPU servers in the cloud. For a lab or a small to medium company with 10+ servers the NVIDIA RTX series is likely most cost effective. You can buy preconfigured servers with Supermicro or Asus chassis that hold 4-8 GPUs efficiently.

GPU vendors typically release a new generation every 1-2 years, such as the GTX 1000 (Pascal) series released in 2017 and the RTX 2000 (Turing) series released in 2019. Each series offers several different models that provide different performance levels. GPU performance is primarily a combination of the following three parameters:

1. **Compute power**. Generally we look for 32-bit floating-point compute power. 16-bit floating point training (FP16) is also entering the mainstream. If you are only interested in prediction, you can also use 8-bit integer. The latest generation of Turing GPUs offers 4-bit acceleration. Unfortunately at present the algorithms to train low-precision networks are not widespread yet.
1. **Memory size**. As your models become larger or the batches used during training grow bigger, you will need more GPU memory. Check for HBM2 (High Bandwidth Memory) vs. GDDR6 (Graphics DDR) memory. HBM2 is faster but much more expensive.
1. **Memory bandwidth**. You can only get the most out of your compute power when you have sufficient memory bandwidth. Look for wide memory buses if using GDDR6.

For most users, it is enough to look at compute power. Note that many GPUs offer different types of acceleration. E.g., NVIDIA's TensorCores accelerate a subset of operators by 5x. Ensure that your libraries support this. The GPU memory should be no less than 4 GB (8 GB is much better). Try to avoid using the GPU also for displaying a GUI (use the built-in graphics instead). If you cannot avoid it, add an extra 2 GB of RAM for safety.

:numref:`fig_flopsvsprice` compares the 32-bit floating-point compute power and price of the various GTX 900, GTX 1000 and RTX 2000 series models. The prices are the suggested prices found on Wikipedia.

![Floating-point compute power and price comparison. ](../img/flopsvsprice.svg)
:label:`fig_flopsvsprice`

We can see a number of things:

1. Within each series, price and performance are roughly proportional. Titan models command a significant premium for the benefit of larger amounts of GPU memory. However, the newer models offer better cost effectiveness, as can be seen by comparing the 980 Ti and 1080 Ti. The price does not appear to improve much for the RTX 2000 series. However, this is due to the fact that they offer far superior low precision performance (FP16, INT8 and INT4).
2. The performance-to-cost ratio of the GTX 1000 series is about two times greater than the 900 series.
3. For the RTX 2000 series the price is an *affine* function of the price.

![Floating-point compute power and energy consumption. ](../img/wattvsprice.svg)
:label:`fig_wattvsprice`


:numref:`fig_wattvsprice` shows how energy consumption scales mostly linearly with the amount of computation. Second, later generations are more efficient. This seems to be contradicted by the graph corresponding to the RTX 2000 series. However, this is a consequence of the TensorCores which draw disproportionately much energy.


## Summary

* Watch out for power, PCIe bus lanes, CPU single thread speed and cooling when building a server.
* You should purchase the latest GPU generation if possible.
* Use the cloud for large deployments.
* High density servers may not be compatible with all GPUs. Check the mechanical and cooling specifications before you buy.
* Use FP16 or lower precision for high efficiency.


[Discussions](https://discuss.d2l.ai/t/425)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTM1NjM2MjM2Ml19
-->
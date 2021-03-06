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
* ** 4 GPUs **. Certifique-se de comprar uma CPU com velocidade de thread único relativamente rápida (ou seja, alta frequência de clock). Você provavelmente precisará de uma CPU com um número maior de pistas PCIe, como um AMD Threadripper. Você provavelmente precisará de placas-mãe relativamente caras para obter 4 slots PCIe 3.0 x16, pois eles provavelmente precisam de um PLX para multiplexar as pistas PCIe. Compre GPUs com design de referência que sejam estreitas e deixe o ar entrar entre as GPUs. Você precisa de uma fonte de alimentação de 1600-2000W e a tomada em seu escritório pode não suportar isso. Este servidor provavelmente irá rodar *alto e alto*. Você não o quer debaixo de sua mesa. 128 GB de DRAM é recomendado. Obtenha um SSD (1-2 TB NVMe) para armazenamento local e vários discos rígidos em configuração RAID para armazenar seus dados.
* ** 8 GPUs **. Você precisa comprar um chassi de servidor multi-GPU dedicado com várias fontes de alimentação redundantes (por exemplo, 2 + 1 para 1600 W por fonte de alimentação). Isso exigirá CPUs de servidor de soquete duplo, 256 GB ECC DRAM, uma placa de rede rápida (10 GBE recomendado) e você precisará verificar se os servidores suportam o *formato físico* das GPUs. O fluxo de ar e a colocação da fiação diferem significativamente entre as GPUs do consumidor e do servidor (por exemplo, RTX 2080 vs. Tesla V100). Isso significa que você pode não conseguir instalar a GPU do consumidor em um servidor devido à folga insuficiente para o cabo de alimentação ou à falta de um chicote de fiação adequado (como um dos co-autores descobriu dolorosamente).
* 
## Selecionando GPUs


Atualmente, a AMD e a NVIDIA são os dois principais fabricantes de GPUs dedicadas. A NVIDIA foi a primeira a entrar no campo de aprendizado profundo e oferece melhor suporte para frameworks de aprendizado profundo via CUDA. Portanto, a maioria dos compradores escolhe GPUs NVIDIA.

A NVIDIA oferece dois tipos de GPUs, direcionados a usuários individuais (por exemplo, por meio das séries GTX e RTX) e usuários corporativos (por meio de sua série Tesla). Os dois tipos de GPUs fornecem potência de computação comparável. No entanto, as GPUs de usuário corporativo geralmente usam resfriamento forçado (passivo), mais memória e memória ECC (correção de erros). Essas GPUs são mais adequadas para data centers e geralmente custam dez vezes mais do que as GPUs de consumidor.

Se você é uma grande empresa com mais de 100 servidores, deve considerar a série NVIDIA Tesla ou, alternativamente, usar servidores GPU na nuvem. Para um laboratório ou uma empresa de pequeno a médio porte com mais de 10 servidores, a série NVIDIA RTX é provavelmente a mais econômica. Você pode comprar servidores pré-configurados com chassis Supermicro ou Asus que comportam de 4 a 8 GPUs com eficiência.

Os fornecedores de GPU normalmente lançam uma nova geração a cada 1-2 anos, como a série GTX 1000 (Pascal) lançada em 2017 e a série RTX 2000 (Turing) lançada em 2019. Cada série oferece vários modelos diferentes que fornecem níveis de desempenho diferentes. O desempenho da GPU é principalmente uma combinação dos três parâmetros a seguir:

1. **Potência de computação**. Geralmente procuramos poder de computação de ponto flutuante de 32 bits. O treinamento de ponto flutuante de 16 bits (FP16) também está entrando no mercado. Se você está interessado apenas em previsões, também pode usar números inteiros de 8 bits. A última geração de GPUs Turing oferece aceleração de 4 bits. Infelizmente, no momento, os algoritmos para treinar redes de baixa precisão ainda não estão muito difundidos.
1. **Tamanho da memória**. À medida que seus modelos ficam maiores ou os lotes usados ​​durante o treinamento ficam maiores, você precisará de mais memória de GPU. Verifique a existência de memória HBM2(High Bandwidth Memory) vs. GDDR6 (Graphics DDR). HBM2 é mais rápido, mas muito mais caro.
1. **Largura de banda da memória**. Você só pode obter o máximo do seu poder de computação quando tiver largura de banda de memória suficiente. Procure por barramentos de memória ampla se estiver usando GDDR6.

Para a maioria dos usuários, basta olhar para o poder de computação. Observe que muitas GPUs oferecem diferentes tipos de aceleração. Por exemplo, os TensorCores da NVIDIA aceleram um subconjunto de operadoras em 5x. Certifique-se de que suas bibliotecas suportem isso. A memória da GPU não deve ser inferior a 4 GB(8 GB é muito melhor). Tente evitar o uso da GPU também para exibir uma GUI (em vez disso, use os gráficos integrados). Se você não puder evitá-lo, adicione 2 GB extras de RAM para segurança.

:numref:`fig_flopsvsprice` compara o poder de computação de ponto flutuante de 32 bits e o preço dos vários modelos das séries GTX 900, GTX 1000 e RTX 2000. Os preços são os preços sugeridos encontrados na Wikipedia.

![Poder de computação de ponto flutuante e comparação de preços. ](../img/flopsvsprice.svg)
:label:`fig_flopsvsprice`

Podemos ver várias coisas:

1. Dentro de cada série, o preço e o desempenho são aproximadamente proporcionais. Os modelos Titan oferecem um prêmio significativo para o benefício de grandes quantidades de memória GPU. No entanto, os modelos mais novos oferecem melhor relação custo-benefício, como pode ser visto ao comparar o 980 Ti e o 1080 Ti. O preço não parece melhorar muito para a série RTX 2000. No entanto, isso se deve ao fato de que eles oferecem desempenho de baixa precisão muito superior (FP16, INT8 e INT4).
2. A relação desempenho-custo da série GTX 1000 é cerca de duas vezes maior do que a série 900.
3. Para a série RTX 2000, o preço é uma função *afim* do preço.

![Potência de computação de ponto flutuante e consumo de energia.](../img/wattvsprice.svg)
:label:`fig_wattvsprice`


:numref:`fig_wattvsprice` mostra como o consumo de energia aumenta linearmente com a quantidade de computação. Em segundo lugar, as gerações posteriores são mais eficientes. Isso parece ser contradito pelo gráfico correspondente à série RTX 2000. No entanto, isso é uma consequência dos TensorCores que consomem energia desproporcionalmente.


## Sumário

* Cuidado com a energia, faixas de barramento PCIe, velocidade de thread único da CPU e resfriamento ao construir um servidor.
* Você deve comprar a geração de GPU mais recente, se possível.
* Use a nuvem para grandes implantações.
* Os servidores de alta densidade podem não ser compatíveis com todas as GPUs. Verifique as especificações mecânicas e de resfriamento antes de comprar.
* Use FP16 ou precisão inferior para alta eficiência.


[Discussão](https://discuss.d2l.ai/t/425)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTUzMDk2MDY5NF19
-->
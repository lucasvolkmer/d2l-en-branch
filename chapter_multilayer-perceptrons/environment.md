# Mudança de Ambiente e Distribuição


Nas seções anteriores, trabalhamos
uma série de aplicações práticas de *machine learning*,
ajustando modelos a uma variedade de conjuntos de dados.
E, no entanto, nunca paramos para contemplar
de onde vêm os dados em primeiro lugar
ou o que planejamos fazer
com as saídas de nossos modelos.
Muitas vezes, desenvolvedores de *machine learning*
na posse de pressa de dados para desenvolver modelos,
não param para considerar essas questões fundamentais.

Muitas implantações de *machine learning* com falha
podem ser rastreadas até este padrão.
Às vezes, os modelos parecem ter um desempenho maravilhoso
conforme medido pela precisão do conjunto de teste
mas falham catastroficamente na implantação
quando a distribuição de dados muda repentinamente.
Mais insidiosamente, às vezes a própria implantação de um modelo
pode ser o catalisador que perturba a distribuição de dados.
Digamos, por exemplo, que treinamos um modelo
para prever quem vai pagar em comparação com o inadimplemento de um empréstimo,
descobrindo que a escolha de calçado de um candidato
foi associado ao risco de inadimplência
(Oxfords indicam reembolso, tênis indicam inadimplência).
Podemos estar inclinados a, a partir daí, conceder empréstimos
a todos os candidatos vestindo Oxfords
e negar a todos os candidatos o uso de tênis.

Neste caso, nosso salto imprudente de
reconhecimento de padrões para a tomada de decisão
e nossa falha em considerar criticamente o ambiente
pode ter consequências desastrosas.
Para começar, assim que começamos
tomar decisões com base em calçados,
os clientes perceberiam e mudariam seu comportamento.
Em pouco tempo, todos os candidatos estariam usando Oxfords,
sem qualquer melhoria coincidente na capacidade de crédito.
Dedique um minuto para digerir isso, porque há muitos problemas semelhantes
em muitas aplicações de *machine learning*:
introduzindo nossas decisões baseadas em modelos para o ambiente,
podemos quebrar o modelo.

Embora não possamos dar a esses tópicos
um tratamento completo em uma seção,
pretendemos aqui expor algumas preocupações comuns,
e estimular o pensamento crítico
necessário para detectar essas situações precocemente,
mitigar os danos e usar o *machine learning* com responsabilidade.
Algumas das soluções são simples
(peça os dados "corretos"),
alguns são tecnicamente difíceis
(implementar um sistema de aprendizagem por reforço),
e outros exigem que saiamos do reino de
previsão estatística em conjunto e
lidemos com difíceis questões filosóficas
relativas à aplicação ética de algoritmos.

## Tipos de Turno de Distribuição


Para começar, ficamos com a configuração de predição passiva
considerando as várias maneiras que as distribuições de dados podem mudar
e o que pode ser feito para salvar o desempenho do modelo.
Em uma configuração clássica, assumimos que nossos dados de treinamento
foram amostrados de alguma distribuição $p_S(\mathbf{x},y)$
mas que nossos dados de teste consistirão
de exemplos não rotulados retirados de
alguma distribuição diferente $p_T(\mathbf{x},y)$.
Já, devemos enfrentar uma realidade preocupante.
Ausentes quaisquer suposições sobre como $p_S$
e $p_T$ se relacionam entre si,
aprender um classificador robusto é impossível.

Considere um problema de classificação binária,
onde desejamos distinguir entre cães e gatos.
Se a distribuição pode mudar de forma arbitrária,
então nossa configuração permite o caso patológico
em que a distribuição sobre os insumos permanece
constante: $p_S(\mathbf{x}) = p_T(\mathbf{x})$,
mas os *labels* estão todos invertidos:
$p_S(y | \mathbf{x}) = 1 - p_T(y | \mathbf{x})$.
Em outras palavras, se Deus pode decidir de repente
que no futuro todos os "gatos" agora são cachorros
e o que anteriormente chamamos de "cães" agora são gatos --- sem
qualquer mudança na distribuição de entradas $p(\mathbf{x})$,
então não podemos distinguir essa configuração
de um em que a distribuição não mudou nada.

Felizmente, sob algumas suposições restritas
sobre como nossos dados podem mudar no futuro,
algoritmos de princípios podem detectar mudanças
e às vezes até se adaptam na hora,
melhorando a precisão do classificador original.

### Mudança Covariável


Entre as categorias de mudança de distribuição,
o deslocamento covariável pode ser o mais amplamente estudado.
Aqui, assumimos que, embora a distribuição de entradas
pode mudar com o tempo, a função de rotulagem,
ou seja, a distribuição condicional
$P(y \mid \mathbf{x})$ não muda.
Os estatísticos chamam isso de *mudança covariável*
porque o problema surge devido a um
mudança na distribuição das covariáveis (*features*).
Embora às vezes possamos raciocinar sobre a mudança de distribuição
sem invocar causalidade, notamos que a mudança da covariável
é a suposição natural para invocar nas configurações
onde acreditamos que $\mathbf{x}$ causa $y$.

Considere o desafio de distinguir cães e gatos.
Nossos dados de treinamento podem consistir em imagens do tipo em :numref:`fig_cat-dog-train`.

![Dados de treinamento para distinguir cães e gatos.](../img/cat-dog-train.svg)
:label:`fig_cat-dog-train`


No momento do teste, somos solicitados a classificar as imagens em :numref:`fig_cat-dog-test`.

![Dados de teste para distinguir cães e gatos.](../img/cat-dog-test.svg)
:label:`fig_cat-dog-test`

O conjunto de treinamento consiste em fotos,
enquanto o conjunto de teste contém apenas desenhos animados.
Treinamento em um conjunto de dados com
características do conjunto de teste
pode significar problemas na ausência de um plano coerente
para saber como se adaptar ao novo domínio.

### Mudança de *Label*

A *Mudança de Label* descreve o problema inverso.
Aqui, assumimos que o rótulo marginal $P(y)$
pode mudar
mas a distribuição condicional de classe
$P(\mathbf{x} \mid y)$ permanece fixa nos domínios.
A mudança de *label* é uma suposição razoável a fazer
quando acreditamos que $y$ causa $\mathbf{x}$.
Por exemplo, podemos querer prever diagnósticos
dados seus sintomas (ou outras manifestações),
mesmo enquanto a prevalência relativa de diagnósticos
esteja mudando com o tempo.
A mudança de rótulo é a suposição apropriada aqui
porque as doenças causam sintomas.
Em alguns casos degenerados, a mudança de rótulo
e as suposições de mudança de covariável podem ser mantidas simultaneamente.
Por exemplo, quando o rótulo é determinístico,
a suposição de mudança da covariável será satisfeita,
mesmo quando $y$ causa $\mathbf{x}$.
Curiosamente, nesses casos,
muitas vezes é vantajoso trabalhar com métodos
que fluem da suposição de mudança de rótulo.
Isso ocorre porque esses métodos tendem
envolver a manipulação de objetos que se parecem com rótulos (muitas vezes de baixa dimensão),
ao contrário de objetos que parecem entradas,
que tendem a ser altamente dimensionais no *deep learning*.

### Mudança de Conceito

Também podemos encontrar o problema relacionado de *mudança de conceito*,
que surge quando as próprias definições de rótulos podem mudar.
Isso soa estranho --- um *gato* é um *gato*, não?
No entanto, outras categorias estão sujeitas a mudanças no uso ao longo do tempo.
Critérios de diagnóstico para doença mental,
o que passa por moda e cargos,
estão todos sujeitos a consideráveis
quantidades de mudança de conceito.
Acontece que se navegarmos pelos Estados Unidos,
mudando a fonte de nossos dados por geografia,
encontraremos uma mudança considerável de conceito em relação
a distribuição de nomes para *refrigerantes*
como mostrado em :numref:`fig_popvssoda`.

![Mudança de conceito em nomes de refrigerantes nos Estados Unidos.](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

Se fossemos construir um sistema de tradução automática,
a distribuição $P(y \mid \mathbf{x})$ pode ser diferente
dependendo de nossa localização.
Esse problema pode ser difícil de detectar.
Podemos ter esperança de explorar o conhecimento
cuja mudança só ocorre gradualmente
seja em um sentido temporal ou geográfico.

## Exemplos de Mudança de Distribuição

Antes de mergulhar no formalismo e algoritmos,
podemos discutir algumas situações concretas
onde a covariável ou mudança de conceito pode não ser óbvia.

### Diagnóstico Médico


Imagine que você deseja criar um algoritmo para detectar o câncer.
Você coleta dados de pessoas saudáveis ​​e doentes
e você treina seu algoritmo.
Funciona bem, oferecendo alta precisão
e você conclui que está pronto
para uma carreira de sucesso em diagnósticos médicos.
*Não tão rápido.*

As distribuições que deram origem aos dados de treinamento
e aqueles que você encontrará na natureza podem diferir consideravelmente.
Isso aconteceu com uma inicialização infeliz
que alguns de nós (autores) trabalhamos anos atrás.
Eles estavam desenvolvendo um exame de sangue para uma doença
que afeta predominantemente homens mais velhos
e esperava estudá-lo usando amostras de sangue
que eles haviam coletado de pacientes.
No entanto, é consideravelmente mais difícil
obter amostras de sangue de homens saudáveis
do que pacientes doentes já no sistema.
Para compensar, a *startup* solicitou
doações de sangue de estudantes em um campus universitário
para servir como controles saudáveis ​​no desenvolvimento de seu teste.
Então eles perguntaram se poderíamos ajudá-los a
construir um classificador para detecção da doença.

Como explicamos a eles,
seria realmente fácil distinguir
entre as coortes saudáveis ​​e doentes
com precisão quase perfeita.
No entanto, isso ocorre porque os assuntos de teste
diferiam em idade, níveis hormonais,
atividade física, dieta, consumo de álcool,
e muitos outros fatores não relacionados à doença.
Era improvável que fosse o caso com pacientes reais.
Devido ao seu procedimento de amostragem,
poderíamos esperar encontrar mudanças extremas covariadas.
Além disso, este caso provavelmente não seria
corrigível por meio de métodos convencionais.
Resumindo, eles desperdiçaram uma quantia significativa de dinheiro.



### Carros Autônomos


Digamos que uma empresa queira aproveitar o *machine learning*
para o desenvolvimento de carros autônomos.
Um componente chave aqui é um detector de beira de estrada.
Uma vez que dados anotados reais são caros de se obter,
eles tiveram a ideia (inteligente e questionável)
de usar dados sintéticos de um motor de renderização de jogo
como dados de treinamento adicionais.
Isso funcionou muito bem em "dados de teste"
extraídos do mecanismo de renderização.
Infelizmente, dentro de um carro de verdade foi um desastre.
Como se viu, a beira da estrada havia sido renderizada
com uma textura muito simplista.
Mais importante, *todo* o acostamento havia sido renderizado
com a *mesma* textura e o detector de beira de estrada
aprendeu sobre essa "característica" muito rapidamente.

Algo semelhante aconteceu com o Exército dos EUA
quando eles tentaram detectar tanques na floresta pela primeira vez.
Eles tiraram fotos aéreas da floresta sem tanques,
em seguida, dirigiram os tanques para a floresta
e tiraram outro conjunto de fotos.
O classificador pareceu funcionar *perfeitamente*.
Infelizmente, ele apenas aprendeu
como distinguir árvores com sombras
de árvores sem sombras --- o primeiro conjunto
de fotos foi tirado no início da manhã,
o segundo conjunto ao meio-dia.

### Distribuições Não-estacionárias


Surge uma situação muito mais sutil
quando a distribuição muda lentamente
(também conhecido como *distribuição não-estacionária*)
e o modelo não é atualizado de forma adequada.
Abaixo estão alguns casos típicos.

* Treinamos um modelo de publicidade computacional e deixamos de atualizá-lo com frequência (por exemplo, esquecemos de incorporar que um novo dispositivo obscuro chamado iPad acabou de ser lançado).
* Construímos um filtro de spam. Ele funciona bem na detecção de todos os spams que vimos até agora. Mas então os spammers se tornaram mais inteligentes e criaram novas mensagens que se parecem com tudo o que vimos antes.
* Construímos um sistema de recomendação de produtos. Ele funciona durante todo o inverno, mas continua a recomendar chapéus de Papai Noel muito depois do Natal.

### Mais Anedotas

* Construímos um detector de rosto. Funciona bem em todos os *benchmarks*. Infelizmente, ele falha nos dados de teste --- os exemplos ofensivos são closes em que o rosto preenche a imagem inteira (nenhum dado desse tipo estava no conjunto de treinamento).
* Construímos um mecanismo de busca na Web para o mercado dos EUA e queremos implantá-lo no Reino Unido.
* Treinamos um classificador de imagens compilando um grande conjunto de dados onde cada um entre um grande conjunto de classes é igualmente representado no conjunto de dados, digamos 1000 categorias, representadas por 1000 imagens cada. Em seguida, implantamos o sistema no mundo real, onde a distribuição real do rótulo das fotos é decididamente não uniforme.






## Correção de Mudança de Distribuição

Como já discutimos, existem muitos casos
onde distribuições de treinamento e teste
$P(\mathbf{x}, y)$ são diferentes.
Em alguns casos, temos sorte e os modelos funcionam
apesar da covariável, rótulo ou mudança de conceito.
Em outros casos, podemos fazer melhor empregando
estratégias baseadas em princípios para lidar com a mudança.
O restante desta seção torna-se consideravelmente mais técnico.
O leitor impaciente pode continuar na próxima seção
já que este material não é pré-requisito para conceitos subsequentes.

### Risco Empírico e Risco
:label:`subsec_empirical-risk-and-risk`

Vamos primeiro refletir sobre o que exatamente
está acontecendo durante o treinamento do modelo:
nós iteramos sobre recursos e rótulos associados
de dados de treinamento
$\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$
e atualizamos os parâmetros de um modelo $f$ após cada *minibatch*.
Para simplificar, não consideramos regularização,
portanto, minimizamos amplamente a perda no treinamento:

$$\mathop{\mathrm{minimizar}}_f \frac{1}{n} \sum_{i=1}^n l(f(\mathbf{x}_i), y_i),$$
:eqlabel:`eq_empirical-risk-min`

onde $l$ é a função de perda
medir "quão ruim" a previsão $f(\mathbf{x}_i)$ recebe o rótulo associado $y_i$.
Os estatísticos chamam o termo em :eqref:`eq_empirical-risk-min` *risco empírico*.
O *risco empírico* é uma perda média sobre os dados de treinamento
para aproximar o *risco*,
que é a
expectativa de perda sobre toda a população de dados extraídos de sua verdadeira distribuição
$p(\mathbf{x},y)$:

$$E_{p(\mathbf{x}, y)} [l(f(\mathbf{x}), y)] = \int\int l(f(\mathbf{x}), y) p(\mathbf{x}, y) \;d\mathbf{x}dy.$$
:eqlabel:`eq_true-risk`

No entanto, na prática, normalmente não podemos obter toda a população de dados.
Assim, a *minimização de risco empírico*,
que está minimizando o risco empírico em :eqref:`eq_empirical-risk-min`,
é uma estratégia prática para *machine learning*,
com a esperança de aproximar
minimizando o risco.



### Covariate Shift Correction
:label:`subsec_covariate-shift-correction`

Suponha que queremos estimar
algumas dependências $P(y \mid \mathbf{x})$
para as quais rotulamos os dados $(\mathbf{x}_i, y_i)$.
Infelizmente, as observações $\mathbf{x}_i$ são desenhadas
de alguma *distribuição de origem* q(\mathbf{x})$
em vez da *distribuição de destino* $p(\mathbf{x})$.
Felizmente,
a suposição de dependência significa
que a distribuição condicional não muda: $p(y \mid \mathbf{x}) = q(y \mid \mathbf{x})$.
Se a distribuição de origem $q(\mathbf{x})$ está "errada",
podemos corrigir isso usando a seguinte identidade simples no risco:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(y \mid \mathbf{x})p(\mathbf{x}) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(y \mid \mathbf{x})q(\mathbf{x})\frac{p(\mathbf{x})}{q(\mathbf{x})} \;d\mathbf{x}dy.
\end{aligned}
$$

Em outras palavras, precisamos pesar novamente cada exemplo de dados
pela proporção do
probabilidade
que teria sido extraída da distribuição correta para a errada:

$$\beta_i \stackrel{\mathrm{def}}{=} \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}.$$

Conectando o peso $\beta_i$ para
cada exemplo de dados $(\mathbf{x}_i, y_i)$
podemos treinar nosso modelo usando
*minimização de risco empírico ponderado*:

$$\mathop{\mathrm{minimizar}}_f \frac{1}{n} \sum_{i=1}^n \beta_i l(f(\mathbf{x}_i), y_i).$$
:eqlabel:`eq_weighted-empirical-risk-min`



Infelizmente, não sabemos essa proporção,
portanto, antes de fazermos qualquer coisa útil, precisamos estimá-la.
Muitos métodos estão disponíveis,
incluindo algumas abordagens teóricas de operador extravagantes
que tentam recalibrar o operador de expectativa diretamente
usando uma norma mínima ou um princípio de entropia máxima.
Observe que, para qualquer abordagem desse tipo, precisamos de amostras
extraídas de ambas as distribuições --- o "verdadeiro" $p$, por exemplo,
por acesso aos dados de teste, e aquele usado
para gerar o conjunto de treinamento $q$ (o último está trivialmente disponível).
Observe, entretanto, que só precisamos dos recursos $\mathbf{x} \sim p(\mathbf{x})$;
não precisamos acessar os rótulos $y \sim p(y)$.


Neste caso, existe uma abordagem muito eficaz
que dará resultados quase tão bons quanto a original: regressão logística,
que é um caso especial de regressão *softmax* (ver :numref:`sec_softmax`)
para classificação binária.
Isso é tudo o que é necessário para calcular as razões de probabilidade estimadas.
Aprendemos um classificador para distinguir
entre os dados extraídos de $p(\mathbf{x})$
e dados extraídos de $q(\mathbf{x})$.
Se é impossível distinguir
entre as duas distribuições
então isso significa que as instâncias associadas
são igualmente prováveis ​​de virem de
qualquer uma das duas distribuições.
Por outro lado, quaisquer instâncias
que podem ser bem discriminadas
devem ser significativamente sobreponderadas
ou subponderadas em conformidade.

Para simplificar, suponha que temos
um número igual de instâncias de ambas as distribuições
$p(\mathbf{x})$
e $q(\mathbf{x})$,, respectivamente.
Agora denote por $z$ rótulos que são $1$
para dados extraídos de $p$ e $-1$ para dados extraídos de $q$.
Então, a probabilidade em um conjunto de dados misto é dada por

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \text{ e portanto } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

Assim, se usarmos uma abordagem de regressão logística,
onde $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-h(\mathbf{x}))}$  ($h$ é uma função parametrizada),
segue que

$$
\beta_i = \frac{1/(1 + \exp(-h(\mathbf{x}_i)))}{\exp(-h(\mathbf{x}_i))/(1 + \exp(-h(\mathbf{x}_i)))} = \exp(h(\mathbf{x}_i)).
$$


Como resultado, precisamos resolver dois problemas:
primeiro a distinguir entre
dados extraídos de ambas as distribuições,
e, em seguida, um problema de minimização de risco empírico ponderado
em :eqref:`eq_weighted-empirical-risk-min`
onde pesamos os termos em $\beta_i$..

Agora estamos prontos para descrever um algoritmo de correção.
Suponha que temos um conjunto de treinamento $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ e um conjunto de teste não rotulado $\{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$.
Para mudança de covariável,
assumimos que $\mathbf{x}_i$ para todos os $1 \leq i \leq n$ são retirados de alguma distribuição de origem
e $\mathbf{u}_i$ for all $1 \leq i \leq m$
são retirados da distribuição de destino.
Aqui está um algoritmo prototípico
para corrigir a mudança da covariável:


1. Gere um conjunto de treinamento de classificação binária: $\{(\mathbf{x}_1, -1), \ldots, (\mathbf{x}_n, -1), (\mathbf{u}_1, 1), \ldots, (\mathbf{u}_m, 1)\}$.
1. Treine um classificador binário usando regressão logística para obter a função $h$.
1. Pese os dados de treinamento usando $\beta_i = \exp(h(\mathbf{x}_i))$ ou melhor $$\beta_i = \min(\exp(h(\mathbf{x}_i)), c)$ para alguma constante $c$.
1. Use pesos $\beta_i$ para treinar em $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ em :eqref:`eq_weighted-empirical-risk-min`.

Observe que o algoritmo acima se baseia em uma suposição crucial.
Para que este esquema funcione, precisamos que cada exemplo de dados
na distribuição de destino (por exemplo, tempo de teste)
tenha probabilidade diferente de zero de ocorrer no momento do treinamento.
Se encontrarmos um ponto onde $p(\mathbf{x}) > 0$ mas $q(\mathbf{x}) = 0$,
então, o peso de importância correspondente deve ser infinito.






### Correção de Mudança de *Label*

Suponha que estamos lidando com um
tarefa de classificação com $k$ categorias.
Usando a mesma notação em :numref:`subsec_covariate-shift-correction`,
$q$ e $p$ são as distribuições de origem (por exemplo, tempo de treinamento) e a distribuição de destino (por exemplo, tempo de teste), respectivamente.
Suponha que a distribuição dos rótulos mude ao longo do tempo:
$q(y) \neq p(y)$, mas a distribuição condicional de classe
permanece a mesma: $q(\mathbf{x} \mid y)=p(\mathbf{x} \mid y)$.
Se a distribuição de origem $q(y)$ estiver "errada",
nós podemos corrigir isso
de acordo com
a seguinte identidade no risco
conforme definido em
 :eqref: `eq_true-risk`:
 
$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(\mathbf{x} \mid y)p(y) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(\mathbf{x} \mid y)q(y)\frac{p(y)}{q(y)} \;d\mathbf{x}dy.
\end{aligned}
$$



Aqui, nossos pesos de importância corresponderão às
taxas de probabilidade de rótulo

$$\beta_i \stackrel{\mathrm{def}}{=} \frac{p(y_i)}{q(y_i)}.$$


Uma coisa boa sobre a mudança de rótulo é que
se tivermos um modelo razoavelmente bom
na distribuição de origem,
então podemos obter estimativas consistentes desses pesos
sem nunca ter que lidar com a dimensão ambiental.
No aprendizado profundo, as entradas tendem
a ser objetos de alta dimensão, como imagens,
enquanto os rótulos são frequentemente objetos mais simples, como categorias.

Para estimar a distribuição de rótulos de destino,
primeiro pegamos nosso classificador de prateleira razoavelmente bom
(normalmente treinado nos dados de treinamento)
e calculamos sua matriz de confusão usando o conjunto de validação
(também da distribuição de treinamento).
A *matriz de confusão*, $\mathbf{C}$, é simplesmente uma matriz $k \times k$,
onde cada coluna corresponde à categoria do rótulo (informações básicas)
e cada linha corresponde à categoria prevista do nosso modelo.
O valor de cada célula $c_{ij}$ é a fração do total de previsões no conjunto de validação
onde o verdadeiro rótulo era $j$ e nosso modelo previu $i$.


Agora, não podemos calcular a matriz de confusão
nos dados de destino diretamente,
porque não conseguimos ver os rótulos dos exemplos
que vemos na natureza,
a menos que invistamos em um pipeline de anotação em tempo real complexo.
O que podemos fazer, no entanto, é calcular a média de todas as nossas previsões de modelos
no momento do teste juntas, produzindo os resultados médios do modelo $\mu(\hat{\mathbf{y}}) \in \mathbb{R}^k$,
cujo $i^\mathrm{th}$ elemento $\mu(\hat{y}_i)$
é a fração das previsões totais no conjunto de teste
onde nosso modelo previu $i$.

Acontece que sob algumas condições amenas --- se
nosso classificador era razoavelmente preciso em primeiro lugar,
e se os dados alvo contiverem apenas categorias
que vimos antes,
e se a suposição de mudança de rótulo se mantém em primeiro lugar
(a suposição mais forte aqui),
então podemos estimar a distribuição do rótulo do conjunto de teste
resolvendo um sistema linear simples

$$\mathbf{C} p(\mathbf{y}) = \mu(\hat{\mathbf{y}}),$$


porque como uma estimativa $\sum_{j=1}^k c_{ij} p(y_j) = \mu(\hat{y}_i)$ vale para todos $1 \leq i \leq k$,
onde $p(y_j)$ é o elemento $j^\mathrm{th}$ do vetor de distribuição de rótulo $k$-dimensional $p(\mathbf{y})$.
Se nosso classificador é suficientemente preciso para começar,
então a matriz de confusão $\mathbf{C}$ será invertível,
e obtemos uma solução $p(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$.

Porque observamos os rótulos nos dados de origem,
é fácil estimar a distribuição $q(y)$.
Então, para qualquer exemplo de treinamento $i$ com rótulo $y_i$,
podemos pegar a razão de nossa estimativa de $p(y_i)/q(y_i)$
para calcular o peso $\beta_i$,
e conecter isso à minimização de risco empírico ponderado
em :eqref:`eq_weighted-empirical-risk-min`.


### Correção da Mudança de Conceito

Concept shift is much harder to fix in a principled manner.
For instance, in a situation where suddenly the problem changes
from distinguishing cats from dogs to one of
distinguishing white from black animals,
it will be unreasonable to assume
that we can do much better than just collecting new labels
and training from scratch.
Fortunately, in practice, such extreme shifts are rare.
Instead, what usually happens is that the task keeps on changing slowly.
To make things more concrete, here are some examples:

* In computational advertising, new products are launched,
old products become less popular. This means that the distribution over ads and their popularity changes gradually and any click-through rate predictor needs to change gradually with it.
* Traffic camera lenses degrade gradually due to environmental wear, affecting image quality progressively.
* News content changes gradually (i.e., most of the news remains unchanged but new stories appear).

In such cases, we can use the same approach that we used for training networks to make them adapt to the change in the data. In other words, we use the existing network weights and simply perform a few update steps with the new data rather than training from scratch.


## A Taxonomy of Learning Problems

Armed with knowledge about how to deal with changes in distributions, we can now consider some other aspects of machine learning problem formulation.


### Batch Learning

In *batch learning*, we have access to training features and labels $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$, which we use to train a model $f(\mathbf{x})$. Later on, we deploy this model to score new data $(\mathbf{x}, y)$ drawn from the same distribution. This is the default assumption for any of the problems that we discuss here. For instance, we might train a cat detector based on lots of pictures of cats and dogs. Once we trained it, we ship it as part of a smart catdoor computer vision system that lets only cats in. This is then installed in a customer's home and is never updated again (barring extreme circumstances).


### Online Learning

Now imagine that the data $(\mathbf{x}_i, y_i)$ arrives one sample at a time. More specifically, assume that we first observe $\mathbf{x}_i$, then we need to come up with an estimate $f(\mathbf{x}_i)$ and only once we have done this, we observe $y_i$ and with it, we receive a reward or incur a loss, given our decision.
Many real problems fall into this category. For example, we need to predict tomorrow's stock price, this allows us to trade based on that estimate and at the end of the day we find out whether our estimate allowed us to make a profit. In other words, in *online learning*, we have the following cycle where we are continuously improving our model given new observations.

$$
\mathrm{model} ~ f_t \longrightarrow
\mathrm{data} ~ \mathbf{x}_t \longrightarrow
\mathrm{estimate} ~ f_t(\mathbf{x}_t) \longrightarrow
\mathrm{observation} ~ y_t \longrightarrow
\mathrm{loss} ~ l(y_t, f_t(\mathbf{x}_t)) \longrightarrow
\mathrm{model} ~ f_{t+1}
$$

### Bandits

*Bandits* are a special case of the problem above. While in most learning problems we have a continuously parametrized function $f$ where we want to learn its parameters (e.g., a deep network), in a *bandit* problem we only have a finite number of arms that we can pull, i.e., a finite number of actions that we can take. It is not very surprising that for this simpler problem stronger theoretical guarantees in terms of optimality can be obtained. We list it mainly since this problem is often (confusingly) treated as if it were a distinct learning setting.


### Control

In many cases the environment remembers what we did. Not necessarily in an adversarial manner but it will just remember and the response will depend on what happened before. For instance, a coffee boiler controller will observe different temperatures depending on whether it was heating the boiler previously. PID (proportional-integral-derivative) controller algorithms are a popular choice there.
Likewise, a user's behavior on a news site will depend on what we showed him previously (e.g., he will read most news only once). Many such algorithms form a model of the environment in which they act such as to make their decisions appear less random.
Recently,
control theory (e.g., PID variants) has also been used
to automatically tune hyperparameters
to achive better disentangling and reconstruction quality,
and improve the diversity of generated text and the reconstruction quality of generated images :cite:`Shao.Yao.Sun.ea.2020`.




### Reinforcement Learning

In the more general case of an environment with memory, we may encounter situations where the environment is trying to cooperate with us (cooperative games, in particular for non-zero-sum games), or others where the environment will try to win. Chess, Go, Backgammon, or StarCraft are some of the cases in *reinforcement learning*. Likewise, we might want to build a good controller for autonomous cars. The other cars are likely to respond to the autonomous car's driving style in nontrivial ways, e.g., trying to avoid it, trying to cause an accident, and trying to cooperate with it.

### Considering the Environment

One key distinction between the different situations above is that the same strategy that might have worked throughout in the case of a stationary environment, might not work throughout when the environment can adapt. For instance, an arbitrage opportunity discovered by a trader is likely to disappear once he starts exploiting it. The speed and manner at which the environment changes determines to a large extent the type of algorithms that we can bring to bear. For instance, if we know that things may only change slowly, we can force any estimate to change only slowly, too. If we know that the environment might change instantaneously, but only very infrequently, we can make allowances for that. These types of knowledge are crucial for the aspiring data scientist to deal with concept shift, i.e., when the problem that he is trying to solve changes over time.




## Fairness, Accountability, and Transparency in Machine Learning

Finally, it is important to remember
that when you deploy machine learning systems
you are not merely optimizing a predictive model---you
are typically providing a tool that will
be used to (partially or fully) automate decisions.
These technical systems can impact the lives
of individuals subject to the resulting decisions.
The leap from considering predictions to decisions
raises not only new technical questions,
but also a slew of ethical questions
that must be carefully considered.
If we are deploying a medical diagnostic system,
we need to know for which populations
it may work and which it may not.
Overlooking foreseeable risks to the welfare of
a subpopulation could cause us to administer inferior care.
Moreover, once we contemplate decision-making systems,
we must step back and reconsider how we evaluate our technology.
Among other consequences of this change of scope,
we will find that *accuracy* is seldom the right measure.
For instance, when translating predictions into actions,
we will often want to take into account
the potential cost sensitivity of erring in various ways.
If one way of misclassifying an image
could be perceived as a racial sleight of hand,
while misclassification to a different category
would be harmless, then we might want to adjust
our thresholds accordingly, accounting for societal values
in designing the decision-making protocol.
We also want to be careful about
how prediction systems can lead to feedback loops.
For example, consider predictive policing systems,
which allocate patrol officers
to areas with high forecasted crime.
It is easy to see how a worrying pattern can emerge:

 1. Neighborhoods with more crime get more patrols.
 1. Consequently, more crimes are discovered in these neighborhoods, entering the training data available for future iterations.
 1. Exposed to more positives, the model predicts yet more crime in these neighborhoods.
 1. In the next iteration, the updated model targets the same neighborhood even more heavily leading to yet more crimes discovered, etc.

Often, the various mechanisms by which
a model's predictions become coupled to its training data
are unaccounted for in the modeling process.
This can lead to what researchers call *runaway feedback loops*.
Additionally, we want to be careful about
whether we are addressing the right problem in the first place.
Predictive algorithms now play an outsize role
in mediating the dissemination of information.
Should the news that an individual encounters
be determined by the set of Facebook pages they have *Liked*?
These are just a few among the many pressing ethical dilemmas
that you might encounter in a career in machine learning.



## Summary

* In many cases training and test sets do not come from the same distribution. This is called distribution shift.
* The risk is the expectation of the loss over the entire population of data drawn from their true distribution. However, this entire population is usually unavailable. Empirical risk is an average loss over the training data to approximate the risk. In practice, we perform empirical risk minimization.
* Under the corresponding assumptions, covariate and label shift can be detected and corrected for at test time. Failure to account for this bias can become problematic at test time.
* In some cases, the environment may remember automated actions and respond in surprising ways. We must account for this possibility when building models and continue to monitor live systems, open to the possibility that our models and the environment will become entangled in unanticipated ways.

## Exercises

1. What could happen when we change the behavior of a search engine? What might the users do? What about the advertisers?
1. Implement a covariate shift detector. Hint: build a classifier.
1. Implement a covariate shift corrector.
1. Besides distribution shift, what else could affect how the empirical risk approximates the risk?


[Discussions](https://discuss.d2l.ai/t/105)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwNjI3NjQ4NDEsMzQ2NjMyNDI5LDMzNT
MzNDEwMywtMTUwNDU2MTI2OCwyMDkwMTU5NjgsLTYwOTYxMzg5
OSwxOTc1MTYyMjM4LDIwNjI5OTE5OCwtMjQyNzQwOTQsMTg3MD
QzNTMxNiwzMzc1NDc3NDksODQyNzc1ODEyXX0=
-->
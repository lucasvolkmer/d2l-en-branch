# Introdução
:label:`chap_introduction`


Até recentemente, quase todos os programas de computador com os quais interagimos diariamente
eram codificados por desenvolvedores de software desde os primeiros princípios.
Digamos que quiséssemos escrever um aplicativo para gerenciar uma plataforma de *e-commerce*.
Depois de se amontoar em um quadro branco por algumas horas para refletir sobre o problema,
iríamos apresentar os traços gerais de uma solução de trabalho que provavelmente se pareceria com isto:
(i) os usuários interagem com o aplicativo por meio de uma interface
executando em um navegador da *web* ou aplicativo móvel;
(ii) nosso aplicativo interage com um mecanismo de banco de dados de nível comercial
para acompanhar o estado de cada usuário e manter registros
de histórico de transações;
e (iii) no cerne de nossa aplicação,
a *lógica de negócios* (você pode dizer, os *cérebros*) de nosso aplicativo
descreve em detalhes metódicos a ação apropriada
que nosso programa deve levar em todas as circunstâncias concebíveis.

Para construir o cérebro de nosso aplicativo,
teríamos que percorrer todos os casos esquivos possíveis
que antecipamos encontrar, criando regras apropriadas.
Cada vez que um cliente clica para adicionar um item ao carrinho de compras,
adicionamos uma entrada à tabela de banco de dados do carrinho de compras,
associando o ID desse usuário ao ID do produto solicitado.
Embora poucos desenvolvedores acertem completamente na primeira vez
(podem ser necessários alguns testes para resolver os problemas),
na maior parte, poderíamos escrever esse programa a partir dos primeiros princípios
e lançá-lo com confiança
*antes* de ver um cliente real.
Nossa capacidade de projetar sistemas automatizados a partir dos primeiros princípios
que impulsionam o funcionamento de produtos e sistemas,
frequentemente em novas situações,
é um feito cognitivo notável.
E quando você é capaz de conceber soluções que funcionam $100\%$ do tempo,
você não deveria usar o *machine learning*.


Felizmente para a crescente comunidade de cientistas de *machine learning*,
muitas tarefas que gostaríamos de automatizar
não se curvam tão facilmente à habilidade humana.
Imagine se amontoar em volta do quadro branco com as mentes mais inteligentes que você conhece,
mas desta vez você está lidando com um dos seguintes problemas:

* Escreva um programa que preveja o clima de amanhã com base em informações geográficas, imagens de satélite e uma janela de rastreamento do tempo passado.
* Escreva um programa que aceite uma pergunta, expressa em texto de forma livre, e a responda corretamente.
* Escreva um programa que, dada uma imagem, possa identificar todas as pessoas que ela contém, desenhando contornos em torno de cada uma.
* Escreva um programa que apresente aos usuários produtos que eles provavelmente irão gostar, mas que provavelmente não encontrarão no curso natural da navegação.

Em cada um desses casos, mesmo programadores de elite
são incapazes de codificar soluções do zero.
As razões para isso podem variar. Às vezes, o programa
que procuramos segue um padrão que muda com o tempo,
e precisamos que nossos programas se adaptem.
Em outros casos, a relação (digamos, entre pixels,
e categorias abstratas) podem ser muito complicadas,
exigindo milhares ou milhões de cálculos
que estão além da nossa compreensão consciente
mesmo que nossos olhos administrem a tarefa sem esforço.
*Machine learning* é o estudo de poderosas
técnicas que podem aprender com a experiência.
À medida que um algoritmo de *machine learning* acumula mais experiência,
normalmente na forma de dados observacionais ou
interações com um ambiente, seu desempenho melhora.
Compare isso com nossa plataforma de comércio eletrônico determinística,
que funciona de acordo com a mesma lógica de negócios,
não importa quanta experiência acumule,
até que os próprios desenvolvedores aprendam e decidam
que é hora de atualizar o *software*.
Neste livro, ensinaremos os fundamentos do *machine learning*,
e foco em particular no *deep learning *,
um poderoso conjunto de técnicas
impulsionando inovações em áreas tão diversas como a visão computacional,
processamento de linguagem natural, saúde e genômica.

## Um exemplo motivador


Antes de começar a escrever, os autores deste livro,
como grande parte da força de trabalho, tiveram que se tornar cafeinados.
Entramos no carro e começamos a dirigir.
Usando um iPhone, Alex gritou "Ei, Siri",
despertando o sistema de reconhecimento de voz do telefone.
Então Mu comandou "rota para a cafeteria *Blue Bottle*".
O telefone rapidamente exibiu a transcrição de seu comando.
Ele também reconheceu que estávamos pedindo direções
e abriu o aplicativo *Maps* (app)
para cumprir nosso pedido.
Depois de laberto, o aplicativo *Maps* identificou várias rotas.
Ao lado de cada rota, o telefone exibia um tempo de trânsito previsto.
Enquanto fabricamos esta história por conveniência pedagógica,
isso demonstra que no intervalo de apenas alguns segundos,
nossas interações diárias com um telefone inteligente
podem envolver vários modelos de *machine learning*.


Imagine apenas escrever um programa para responder a uma *palavra de alerta*
como "Alexa", "OK Google" e "Hey Siri".
Tente codificar em uma sala sozinho
com nada além de um computador e um editor de código,
conforme ilustrado em: numref: `fig_wake_word`.
Como você escreveria tal programa a partir dos primeiros princípios?
Pense nisso ... o problema é difícil.
A cada segundo, o microfone irá coletar aproximadamente
44.000 amostras.
Cada amostra é uma medida da amplitude da onda sonora.
Que regra poderia mapear de forma confiável, de um trecho de áudio bruto a previsões confiáveis
$\{\text{yes},\text{no}\}$
sobre se o trecho de áudio contém a palavra de ativação?
Se você estiver travado, não se preocupe.
Também não sabemos escrever tal programa do zero.
É por isso que usamos o *machine learning*.

![Identificar uma palavra de ativação.](../img/wake-word.svg)
:label:`fig_wake_word`



Aqui está o truque.
Muitas vezes, mesmo quando não sabemos como dizer a um computador
explicitamente como mapear de entradas para saídas,
ainda assim, somos capazes de realizar a façanha cognitiva por nós mesmos.
Em outras palavras, mesmo que você não saiba
como programar um computador para reconhecer a palavra "Alexa",
você mesmo é capaz de reconhecê-lo.
Armados com essa habilidade, podemos coletar um enorme *dataset*
contendo exemplos de áudio
e rotular aqueles que contêm
e que não contêm a palavra de ativação.
Na abordagem de *machine learning*,
não tentamos projetar um sistema
*explicitamente* para reconhecer palavras de ativação.
Em vez disso, definimos um programa flexível
cujo comportamento é determinado por vários *parâmetros*.
Em seguida, usamos o conjunto de dados para determinar o melhor conjunto possível de parâmetros,
aqueles que melhoram o desempenho do nosso programa
com respeito a alguma medida de desempenho na tarefa de interesse.

Você pode pensar nos parâmetros como botões que podemos girar,
manipulando o comportamento do programa.
Fixando os parâmetros, chamamos o programa de *modelo*.
O conjunto de todos os programas distintos (mapeamentos de entrada-saída)
que podemos produzir apenas manipulando os parâmetros
é chamada de *família* de modelos.
E o meta-programa que usa nosso conjunto de dados
para escolher os parâmetros é chamado de *algoritmo de aprendizagem*.


Antes de prosseguirmos e envolvermos o algoritmo de aprendizagem,
temos que definir o problema com precisão,
identificando a natureza exata das entradas e saídas,
e escolher uma família modelo apropriada.
Nesse caso,
nosso modelo recebe um trecho de áudio como *entrada*,
e o modelo
gera uma seleção entre
$\{\text{yes},\text{no}\}$ como *saída*.
Se tudo correr de acordo com o plano
as suposições da modelo vão
normalmente estar corretas quanto a
se o áudio contém a palavra de ativação.

Se escolhermos a família certa de modelos,
deve haver uma configuração dos botões
de forma que o modelo dispara "sim" toda vez que ouve a palavra "Alexa".
Como a escolha exata da palavra de ativação é arbitrária,
provavelmente precisaremos de uma família modelo suficientemente rica que,
por meio de outra configuração dos botões, ele poderia disparar "sim"
somente ao ouvir a palavra "Damasco".
Esperamos que a mesma família de modelo seja adequada
para reconhecimento "Alexa" e reconhecimento "Damasco"
porque parecem, intuitivamente, tarefas semelhantes.
No entanto, podemos precisar de uma família totalmente diferente de modelos
se quisermos lidar com entradas ou saídas fundamentalmente diferentes,
digamos que se quiséssemos mapear de imagens para legendas,
ou de frases em inglês para frases em chinês.

Como você pode imaginar, se apenas definirmos todos os botões aleatoriamente,
é improvável que nosso modelo reconheça "Alexa",
"Apricot", ou qualquer outra palavra em inglês.
No *machine learning*,
o *aprendizado* (*learning*) é o processo
pelo qual descobrimos a configuração certa dos botões
coagindo o comportamento desejado de nosso modelo.
Em outras palavras,
nós *treinamos* nosso modelo com dados.
Conforme mostrado em: numref:`fig_ml_loop`, o processo de treinamento geralmente se parece com o seguinte:

1. Comece com um modelo inicializado aleatoriamente que não pode fazer nada útil.
2. Pegue alguns de seus dados (por exemplo, trechos de áudio e *labels* $\{\text{yes},\text{no}\}$ correspondentes).
3. Ajuste os botões para que o modelo seja menos ruim em relação a esses exemplos.
4. Repita as etapas 2 e 3 até que o modelo esteja incrível.

![Um processo de treinamento típico.](../img/ml-loop.svg)
:label:`fig_ml_loop`

Para resumir, em vez de codificar um reconhecedor de palavra de acionamento,
nós codificamos um programa que pode *aprender* a reconhecê-las
se o apresentarmos com um grande *dataset* rotulado.
Você pode pensar neste ato de determinar o comportamento de um programa
apresentando-o com um *dataset* como *programação com dados*.
Quer dizer,
podemos "programar" um detector de gatos, fornecendo nosso sistema de aprendizado de máquina
com muitos exemplos de cães e gatos.
Dessa forma, o detector aprenderá a emitir um número positivo muito grande se for um gato, um número negativo muito grande se for um cachorro,
e algo mais próximo de zero se não houver certeza,
e isso é apenas a ponta do *iceberg* do que o *machine learning*  pode fazer.
*Deep learning*,
que iremos explicar em maiores detalhes posteriormente,
é apenas um entre muitos métodos populares
para resolver problemas de *machine learning*.

## Componentes chave


Em nosso exemplo de palavra de ativação, descrevemos um *dataset*
consistindo em trechos de áudio e *labels* binários,
e nós
demos uma sensação ondulante de como podemos treinar
um modelo para aproximar um mapeamento de áudios para classificações.
Esse tipo de problema,
onde tentamos prever um *label* desconhecido designado
com base em entradas conhecidas
dado um conjunto de dados que consiste em exemplos
para os quais os rótulos são conhecidos,
é chamado de *aprendizagem supervisionada*.
Esse é apenas um entre muitos tipos de problemas de *machine learning*.
Posteriormente, mergulharemos profundamente em diferentes problemas de *machine learning*.
Primeiro, gostaríamos de lançar mais luz sobre alguns componentes principais
que nos acompanharão, independentemente do tipo de problema de *machine learning* que enfrentarmos:

1. Os *dados* com os quais podemos aprender.
1. Um *modelo* de como transformar os dados.
1. Uma *função objetivo* que quantifica o quão bem (ou mal) o modelo está indo.
1. Um *algoritmo* para ajustar os parâmetros do modelo para otimizar a função objetivo.

### Dados


Nem é preciso dizer que você não pode fazer ciência de dados sem dados.
Podemos perder centenas de páginas pensando no que exatamente constitui os dados,
mas por agora, vamos errar no lado prático
e focar nas principais propriedades com as quais se preocupar.
Geralmente, estamos preocupados com uma coleção de exemplos.
Para trabalhar com dados de maneira útil,
nós tipicamente
precisamos chegar a uma representação numérica adequada.
Cada *exemplo* (ou *ponto de dados*, *instância de dados*, *amostra*) normalmente consiste em um conjunto
de atributos chamados *recursos* (ou *covariáveis ​*),
a partir do qual o modelo deve fazer suas previsões.
Nos problemas de aprendizagem supervisionada acima,
a coisa a prever
é um atributo especial
que é designado como
o *rótulo* (*label*) (ou *alvo*).


Se estivéssemos trabalhando com dados de imagem,
cada fotografia individual pode constituir um exemplo,
cada um representado por uma lista ordenada de valores numéricos
correspondendo ao brilho de cada pixel.
Uma fotografia colorida de $200\times 200$ consistiria em $200\times200 \times3=120000$
valores numéricos, correspondentes ao brilho
dos canais vermelho, verde e azul para cada pixel.
Em outra tarefa tradicional, podemos tentar prever
se um paciente vai sobreviver ou não,
dado um conjunto padrão de recursos, como
idade, sinais vitais e diagnósticos.


Quando cada exemplo é caracterizado pelo mesmo número de valores numéricos,
dizemos que os dados consistem em vetores de comprimento fixo
e descrevemos o comprimento constante dos vetores
como a *dimensionalidade* dos dados.
Como você pode imaginar, o comprimento fixo pode ser uma propriedade conveniente.
Se quiséssemos treinar um modelo para reconhecer o câncer em imagens microscópicas,
entradas de comprimento fixo significam que temos uma coisa a menos com que nos preocupar.

No entanto, nem todos os dados podem ser facilmente representados como
vetores de *comprimento fixo*.
Embora possamos esperar que as imagens do microscópio venham de equipamentos padrão,
não podemos esperar imagens extraídas da Internet
aparecerem todas com a mesma resolução ou formato.
Para imagens, podemos considerar cortá-los todas em um tamanho padrão,
mas essa estratégia só nos leva até certo ponto.
Corremos o risco de perder informações nas partes cortadas.
Além disso, os dados de texto resistem a representações de comprimento fixo ainda mais obstinadamente.
Considere os comentários de clientes deixados em sites de comércio eletrônico
como Amazon, IMDB e TripAdvisor.
Alguns são curtos: "é uma porcaria!".
Outros vagam por páginas.
Uma das principais vantagens do *deep learning* sobre os métodos tradicionais
é a graça comparativa com a qual os modelos modernos
podem lidar com dados de *comprimento variável*.

Geralmente, quanto mais dados temos, mais fácil se torna nosso trabalho.
Quando temos mais dados, podemos treinar modelos mais poderosos
e dependem menos de suposições pré-concebidas.
A mudança de regime de (comparativamente) pequeno para *big data*
é um dos principais contribuintes para o sucesso do *deep learning* moderno.
Para esclarecer, muitos dos modelos mais interessantes de *deep learning* não funcionam sem grandes *datasets*.
Alguns outros trabalham no regime de pequenos dados,
mas não são melhores do que as abordagens tradicionais.

Por fim, não basta ter muitos dados e processá-los com inteligência.
Precisamos dos dados *certos*.
Se os dados estiverem cheios de erros,
ou se os recursos escolhidos não são preditivos
da quantidade alvo de interesse,
o aprendizado vai falhar.
A situação é bem capturada pelo clichê:
*entra lixo, sai lixo*.
Além disso, o desempenho preditivo ruim não é a única consequência potencial.
Em aplicativos sensíveis de *machine learning*,
como policiamento preditivo, triagem de currículo e modelos de risco usados ​​para empréstimos,
devemos estar especialmente alertas para as consequências de dados inúteis.
Um modo de falha comum ocorre em conjuntos de dados onde alguns grupos de pessoas
não são representados nos dados de treinamento.
Imagine aplicar um sistema de reconhecimento de câncer de pele na natureza
que nunca tinha visto pele negra antes.
A falha também pode ocorrer quando os dados
não apenas sub-representem alguns grupos
mas refletem preconceitos sociais.
Por exemplo,
se as decisões de contratação anteriores forem usadas para treinar um modelo preditivo
que será usado para selecionar currículos,
então os modelos de aprendizado de máquina poderiam inadvertidamente
capturar e automatizar injustiças históricas.
Observe que tudo isso pode acontecer sem o cientista de dados
conspirar ativamente, ou mesmo estar ciente.


### Modelos

A maior parte do *machine learning* envolve transformar os dados de alguma forma.
Talvez queiramos construir um sistema que ingere fotos e preveja sorrisos.
Alternativamente,
podemos querer ingerir um conjunto de leituras de sensor
e prever quão normais ou anômalas são as leituras.
Por *modelo*, denotamos a maquinaria computacional para ingestão de dados
de um tipo,
e cuspir previsões de um tipo possivelmente diferente.
Em particular, estamos interessados ​​em modelos estatísticos
que podem ser estimados a partir de dados.
Embora os modelos simples sejam perfeitamente capazes de abordar
problemas apropriadamente simples,
os problemas
nos quais nos concentramos neste livro, ampliam os limites dos métodos clássicos.
O *deep learning* é diferenciado das abordagens clássicas
principalmente pelo conjunto de modelos poderosos em que se concentra.
Esses modelos consistem em muitas transformações sucessivas dos dados
que são encadeados de cima para baixo, daí o nome *deep learning*.
No caminho para discutir modelos profundos,
também discutiremos alguns métodos mais tradicionais.


### Funções Objetivo


Anteriormente, apresentamos o *machine learning* como aprendizado com a experiência.
Por *aprender* aqui,
queremos dizer melhorar em alguma tarefa ao longo do tempo.
Mas quem pode dizer o que constitui uma melhoria?
Você pode imaginar que poderíamos propor a atualização do nosso modelo,
e algumas pessoas podem discordar sobre se a atualização proposta
constituiu uma melhoria ou um declínio.

A fim de desenvolver um sistema matemático formal de máquinas de aprendizagem,
precisamos ter medidas formais de quão bons (ou ruins) nossos modelos são.
No *machine learning*, e na otimização em geral,
chamamos elas de *funções objetivo*.
Por convenção, geralmente definimos funções objetivo
de modo que quanto menor, melhor.
Esta é apenas uma convenção.
Você pode assumir qualquer função
para a qual mais alto é melhor, e transformá-la em uma nova função
que é qualitativamente idêntica, mas para a qual menor é melhor,
invertendo o sinal.
Porque quanto menor é melhor, essas funções às vezes são chamadas
*funções de perda* (*loss functions*).

Ao tentar prever valores numéricos,
a função de perda mais comum é *erro quadrático*,
ou seja, o quadrado da diferença entre a previsão e a verdade fundamental.
Para classificação, o objetivo mais comum é minimizar a taxa de erro,
ou seja, a fração de exemplos em que
nossas previsões discordam da verdade fundamental.
Alguns objetivos (por exemplo, erro quadrático) são fáceis de otimizar.
Outros (por exemplo, taxa de erro) são difíceis de otimizar diretamente,
devido à indiferenciabilidade ou outras complicações.
Nesses casos, é comum otimizar um *objetivo substituto*.

Normalmente, a função de perda é definida
no que diz respeito aos parâmetros do modelo
e depende do conjunto de dados.
Nós aprendemos
os melhores valores dos parâmetros do nosso modelo
minimizando a perda incorrida em um conjunto
consistindo em alguns exemplos coletados para treinamento.
No entanto, indo bem nos dados de treinamento
não garante que teremos um bom desempenho com dados não vistos.
Portanto, normalmente queremos dividir os dados disponíveis em duas partições:
o *dataset de treinamento* (ou *conjunto de treinamento*, para ajustar os parâmetros do modelo)
e o *dataset de teste* (ou *conjunto de teste*, que é apresentado para avaliação),
relatando o desempenho do modelo em ambos.
Você pode pensar no desempenho do treinamento como sendo
as pontuações de um aluno em exames práticos
usado para se preparar para algum exame final real.
Mesmo que os resultados sejam encorajadores,
isso não garante sucesso no exame final.
Em outras palavras,
o desempenho do teste pode divergir significativamente do desempenho do treinamento.
Quando um modelo tem um bom desempenho no conjunto de treinamento
mas falha em generalizar para dados invisíveis,
dizemos que está fazendo *overfitting*.
Em termos da vida real, é como ser reprovado no exame real
apesar de ir bem nos exames práticos.

### Algoritmos de Otimização

Assim que tivermos alguma fonte de dados e representação,
um modelo e uma função objetivo bem definida,
precisamos de um algoritmo capaz de pesquisar
para obter os melhores parâmetros possíveis para minimizar a função de perda.
Algoritmos de otimização populares para aprendizagem profunda
baseiam-se em uma abordagem chamada *gradiente descendente*.
Em suma, em cada etapa, este método
verifica, para cada parâmetro,
para que lado a perda do conjunto de treinamento se moveria
se você perturbou esse parâmetro apenas um pouco.
Em seguida, atualiza
o parâmetro na direção que pode reduzir a perda.

## Tipos de Problemas de *Machine Learning*

O problema da palavra de ativação em nosso exemplo motivador
é apenas um entre
muitos problemas que o *machine learning* pode resolver.
Para motivar ainda mais o leitor
e nos fornecer uma linguagem comum quando falarmos sobre mais problemas ao longo do livro,
a seguir nós
listamos uma amostra dos problemas de *machine learning*.
Estaremos constantemente nos referindo a
nossos conceitos acima mencionados
como dados, modelos e técnicas de treinamento.

### Aprendizagem Supervisionada


A aprendizagem supervisionada (*supervised learning*) aborda a tarefa de
prever *labels* com recursos de entrada.
Cada par recurso-rótulo é chamado de exemplo.
Às vezes, quando o contexto é claro, podemos usar o termo *exemplos*
para se referir a uma coleção de entradas,
mesmo quando os *labels* correspondentes são desconhecidos.
Nosso objetivo é produzir um modelo
que mapeia qualquer entrada para uma previsão de *label*.


Para fundamentar esta descrição em um exemplo concreto,
se estivéssemos trabalhando na área de saúde,
então podemos querer prever se um paciente teria um ataque cardíaco ou não.
Esta observação, "ataque cardíaco" ou "sem ataque cardíaco",
seria nosso *label*.
Os recursos de entrada podem ser sinais vitais
como frequência cardíaca, pressão arterial diastólica,
e pressão arterial sistólica.

A supervisão entra em jogo porque para a escolha dos parâmetros, nós (os supervisores) fornecemos ao modelo um conjunto de dados
consistindo em exemplos rotulados,
onde cada exemplo é correspondido com o *label* da verdade fundamental.
Em termos probabilísticos, normalmente estamos interessados ​​em estimar
a probabilidade condicional de determinados recursos de entrada de um *label*.
Embora seja apenas um entre vários paradigmas no *machine learning*,
a aprendizagem supervisionada é responsável pela maioria das bem-sucedidas
aplicações de *machine learning* na indústria.
Em parte, isso ocorre porque muitas tarefas importantes
podem ser descritas nitidamente como estimar a probabilidade
de algo desconhecido dado um determinado *dataset* disponível:

* Prever câncer versus não câncer, dada uma imagem de tomografia computadorizada.
* Prever a tradução correta em francês, dada uma frase em inglês.
* Prever o preço de uma ação no próximo mês com base nos dados de relatórios financeiros deste mês.


Mesmo com a descrição simples
"previsão de *labels* com recursos de entrada"
a aprendizagem supervisionada pode assumir muitas formas
e exigem muitas decisões de modelagem,
dependendo (entre outras considerações) do tipo, tamanho,
e o número de entradas e saídas.
Por exemplo, usamos diferentes modelos para processar sequências de comprimentos arbitrários
e para processar representações de vetores de comprimento fixo.
Visitaremos muitos desses problemas em profundidade
ao longo deste livro.

Informalmente, o processo de aprendizagem se parece com o seguinte.
Primeiro, pegue uma grande coleção de exemplos para os quais os recursos são conhecidos
e selecione deles um subconjunto aleatório,
adquirindo os *labels* da verdade fundamental para cada um.
Às vezes, esses *labels* podem ser dados disponíveis que já foram coletados
(por exemplo, um paciente morreu no ano seguinte?)
e outras vezes, podemos precisar empregar anotadores humanos para rotular os dados,
(por exemplo, atribuição de imagens a categorias).
Juntas, essas entradas e os *labels* correspondentes constituem o conjunto de treinamento.
Alimentamos o *dataset* de treinamento em um algoritmo de aprendizado supervisionado,
uma função que recebe como entrada um conjunto de dados
e produz outra função: o modelo aprendido.
Finalmente, podemos alimentar entradas não vistas anteriormente para o modelo aprendido,
usando suas saídas como previsões do rótulo correspondente.
O processo completo é desenhado em: numref: `fig_supervised_learning`.

![Aprendizagem supervisionada.](../img/supervised-learning.svg)
:label:`fig_supervised_learning`

#### Regressão


Talvez a tarefa de aprendizagem supervisionada mais simples
para entender é *regressão*.
Considere, por exemplo, um conjunto de dados coletados
de um banco de dados de vendas de casas.
Podemos construir uma mesa,
onde cada linha corresponde a uma casa diferente,
e cada coluna corresponde a algum atributo relevante,
como a metragem quadrada de uma casa,
o número de quartos, o número de banheiros e o número de minutos (caminhando) até o centro da cidade.
Neste conjunto de dados, cada exemplo seria uma casa específica,
e o vetor de recurso correspondente seria uma linha na tabela.
Se você mora em Nova York ou São Francisco,
e você não é o CEO da Amazon, Google, Microsoft ou Facebook,
o vetor de recursos (metragem quadrada, nº de quartos, nº de banheiros, distância a pé)
para sua casa pode ser algo como: $ [56, 1, 1, 60] $.
No entanto, se você mora em Pittsburgh, pode ser parecido com $ [279, 4, 3, 10] $.
Vetores de recursos como este são essenciais
para a maioria dos algoritmos clássicos de *machine learning*.

O que torna um problema uma regressão é, na verdade, o resultado.
Digamos que você esteja em busca de uma nova casa.
Você pode querer estimar o valor justo de mercado de uma casa,
dados alguns recursos como acima.
O *label*, o preço de venda, é um valor numérico.
Quando os *labels* assumem valores numéricos arbitrários,
chamamos isso de problema de *regressão*.
Nosso objetivo é produzir um modelo cujas previsões
aproximar os valores reais do *label*.



Muitos problemas práticos são problemas de regressão bem descritos.
Prever a avaliação que um usuário atribuirá a um filme
pode ser pensado como um problema de regressão
e se você projetou um ótimo algoritmo para realizar essa façanha em 2009,
você pode ter ganho o [prêmio de 1 milhão de dólares da Netflix] (https://en.wikipedia.org/wiki/Netflix_Prize).
Previsão do tempo de permanência de pacientes no hospital
também é um problema de regressão.
Uma boa regra é: qualquer problema de *quanto?* Ou *quantos?*
deve sugerir regressão,
tal como:

* Quantas horas durará esta cirurgia?
* Quanta chuva esta cidade terá nas próximas seis horas?


Mesmo que você nunca tenha trabalhado com *machine learning* antes,
você provavelmente já trabalhou em um problema de regressão informalmente.
Imagine, por exemplo, que você mandou consertar seus ralos
e que seu contratante gastou 3 horas
removendo sujeira de seus canos de esgoto.
Então ele lhe enviou uma conta de 350 dólares.
Agora imagine que seu amigo contratou o mesmo empreiteiro por 2 horas
e que ele recebeu uma nota de 250 dólares.
Se alguém lhe perguntasse quanto esperar
em sua próxima fatura de remoção de sujeira
você pode fazer algumas suposições razoáveis,
como mais horas trabalhadas custam mais dólares.
Você também pode presumir que há alguma carga básica
e que o contratante cobra por hora.
Se essas suposições forem verdadeiras, dados esses dois exemplos de dados,
você já pode identificar a estrutura de preços do contratante:
100 dólares por hora mais 50 dólares para aparecer em sua casa.
Se você acompanhou esse exemplo, então você já entendeu
a ideia de alto nível por trás da regressão linear.

Neste caso, poderíamos produzir os parâmetros
que correspondem exatamente aos preços do contratante.
Às vezes isso não é possível,
por exemplo, se alguma variação se deve a algum fator
além dos dois citados.
Nestes casos, tentaremos aprender modelos
que minimizam a distância entre nossas previsões e os valores observados.
Na maioria de nossos capítulos, vamos nos concentrar em
minimizar a função de perda de erro quadrático.
Como veremos mais adiante, essa perda corresponde ao pressuposto
que nossos dados foram corrompidos pelo ruído gaussiano.


#### Classificação


Embora os modelos de regressão sejam ótimos para responder às questões *quantos?*,
muitos problemas não se adaptam confortavelmente a este modelo.
Por exemplo,
um banco deseja adicionar a digitalização de cheques ao seu aplicativo móvel.
Isso envolveria o cliente tirando uma foto de um cheque
com a câmera do smartphone deles
e o aplicativo precisaria ser capaz
de entender automaticamente o texto visto na imagem.
Especificamente,
também precisaria entender o texto manuscrito para ser ainda mais robusto,
como mapear um caractere escrito à mão
a um dos personagens conhecidos.
Este tipo de problema de *qual?* É chamado de *classificação*.
É tratado com um conjunto diferente de algoritmos
do que aqueles usados ​​para regressão, embora muitas técnicas sejam transportadas.

Na *classificação*, queremos que nosso modelo analise os recursos,
por exemplo, os valores de pixel em uma imagem,
e, em seguida, prever qual *categoria* (formalmente chamada de *classe*),
entre alguns conjuntos discretos de opções, um exemplo pertence.
Para dígitos manuscritos, podemos ter dez classes,
correspondendo aos dígitos de 0 a 9.
A forma mais simples de classificação é quando existem apenas duas classes,
um problema que chamamos de *classificação binária*.
Por exemplo, nosso conjunto de dados pode consistir em imagens de animais
e nossos rótulos podem ser as classes $\mathrm{\{cat, dog\}}$.
Durante a regressão, buscamos um regressor para produzir um valor numérico,
na classificação, buscamos um classificador, cuja saída é a atribuição de classe prevista.

Por razões que abordaremos à medida que o livro se torna mais técnico,
pode ser difícil otimizar um modelo que só pode produzir
uma tarefa categórica difícil,
por exemplo, "gato" ou "cachorro".
Nesses casos, geralmente é muito mais fácil expressar
nosso modelo na linguagem das probabilidades.
Dados os recursos de um exemplo,
nosso modelo atribui uma probabilidade
para cada classe possível.
Voltando ao nosso exemplo de classificação animal
onde as classes são $\mathrm{\{gato, cachorro\}}$,
um classificador pode ver uma imagem e gerar a probabilidade
que a imagem é um gato como 0,9.
Podemos interpretar esse número dizendo que o classificador
tem 90\% de certeza de que a imagem representa um gato.
A magnitude da probabilidade para a classe prevista
transmite uma noção de incerteza.
Esta não é a única noção de incerteza
e discutiremos outros em capítulos mais avançados.


Quando temos mais de duas classes possíveis,
chamamos o problema de *classificação multiclasse*.
Exemplos comuns incluem reconhecimento de caracteres escritos à mão
$\mathrm{\{0, 1, 2, ... 9, a, b, c, ...\}}$.
Enquanto atacamos problemas de regressão tentando
minimizar a função de perda de erro quadrático,
a função de perda comum para problemas de classificação é chamada de *entropia cruzada* (*cross-entropy*),
cujo nome pode ser desmistificado
por meio de uma introdução à teoria da informação nos capítulos subsequentes.

Observe que a classe mais provável não é necessariamente
aquela que você usará para sua decisão.
Suponha que você encontre um lindo cogumelo em seu quintal
como mostrado em: numref: `fig_death_cap`.

![Cicuta verde [^-1] --- não coma!](../img/death-cap.jpg)
:width:`200px`
:label:`fig_death_cap`

[^-1]: O cogumelo *Amanita phalloides*, cujo nome popular em inglês é *Death cap*, foi traduzido para o nome popular em português, Cicuta verde.

Agora, suponha que você construiu um classificador e o treinou
para prever se um cogumelo é venenoso com base em uma fotografia.
Digamos que nossos resultados do classificador de detecção de veneno
que a probabilidade de que
: numref: `fig_death_cap` contém um Cicuta verde de 0,2.
Em outras palavras, o classificador tem 80\% de certeza
que nosso cogumelo não é um Cicuta verde.
Ainda assim, você teria que ser um tolo para comê-lo.
Isso porque o certo benefício de um jantar delicioso
não vale a pena um risco de 20\% de morrer por causa disso.
Em outras palavras, o efeito do risco incerto
supera o benefício de longe.
Assim, precisamos calcular o risco esperado que incorremos como a função de perda,
ou seja, precisamos multiplicar a probabilidade do resultado
com o benefício (ou dano) associado a ele.
Nesse caso,
a perda incorrida ao comer o cogumelo
pode ser $0,2\times\infty + 0,8 \times 0 = \infty$,
Considerando que a perda de descarte é
$0,2\times 0 + 0,8 \times 1 = 0,8$.
Nossa cautela foi justificada:
como qualquer micologista nos diria,
o cogumelo em: numref: `fig_death_cap` na verdade
é um Cicuta verde.

A classificação pode ser muito mais complicada do que apenas
classificação binária, multi-classe ou mesmo com vários rótulos.
Por exemplo, existem algumas variantes de classificação
para abordar hierarquias.
As hierarquias assumem que existem alguns relacionamentos entre as muitas classes.
Portanto, nem todos os erros são iguais --- se devemos errar, preferiríamos
classificar incorretamente para uma classe parecida em vez de uma classe distante.
Normalmente, isso é conhecido como *classificação hierárquica*.
Um exemplo inicial é devido a [Linnaeus] (https://en.wikipedia.org/wiki/Carl_Linnaeus), que organizou os animais em uma hierarquia.

No caso da classificação animal,
pode não ser tão ruim confundir um poodle (uma raça de cachorro) com um schnauzer (outra raça de cachorro),
mas nosso modelo pagaria uma grande penalidade
se confundisse um poodle com um dinossauro.
Qual hierarquia é relevante pode depender
sobre como você planeja usar o modelo.
Por exemplo, cascavéis e cobras-liga
podem estar perto da árvore filogenética,
mas confundir uma cascavel com uma cobra-liga pode ser mortal.

#### *Tags*

Alguns problemas de classificação se encaixam perfeitamente
nas configurações de classificação binária ou multiclasse.
Por exemplo, podemos treinar um classificador binário normal
para distinguir gatos de cães.
Dado o estado atual da visão computacional,
podemos fazer isso facilmente, com ferramentas disponíveis no mercado.
No entanto, não importa o quão preciso seja o nosso modelo,
podemos ter problemas quando o classificador
encontra uma imagem dos *Músicos da Cidade de Bremen*,
um conto de fadas alemão popular com quatro animais
in: numref: `fig_stackedanimals`.

![Um burro, um cachorro, um gato e um galo.](../img/stackedanimals.png)
:width:`300px`
:label:`fig_stackedanimals`


Como você pode ver, há um gato em: numref: `fig_stackedanimals`,
e um galo, um cachorro e um burro,
com algumas árvores ao fundo.
Dependendo do que queremos fazer com nosso modelo
em última análise, tratando isso como um problema de classificação binária
pode não fazer muito sentido.
Em vez disso, podemos dar ao modelo a opção de
dizer que a imagem retrata um gato, um cachorro, um burro,
*e* um galo.

O problema de aprender a prever classes que são
não mutuamente exclusivas é chamado de *classificação multi-rótulo*.
Os problemas de *tags* automáticas são geralmente mais bem descritos
como problemas de classificação multi-rótulo.
Pense nas *tags* que as pessoas podem aplicar a postagens em um blog técnico,
por exemplo, "*machine learning*", "tecnologia", "*gadgets*",
"linguagens de programação", "Linux", "computação em nuvem", "AWS".
Um artigo típico pode ter de 5 a 10 *tags* aplicadas
porque esses conceitos estão correlacionados.
Postagens sobre "computação em nuvem" provavelmente mencionarão "AWS"
e postagens sobre "*machine learning*" também podem tratar
de "linguagens de programação".

Também temos que lidar com esse tipo de problema ao lidar
com a literatura biomédica, onde etiquetar corretamente os artigos é importante
porque permite que os pesquisadores façam revisões exaustivas da literatura.
Na *National Library of Medicine*, vários anotadores profissionais
revisam cada artigo que é indexado no *PubMed*
para associá-lo aos termos relevantes do MeSH,
uma coleção de aproximadamente 28000 *tags*.
Este é um processo demorado e o
os anotadores normalmente têm um atraso de um ano entre o arquivamento e a definição das *tags*.
O *machine learning* pode ser usado aqui para fornecer *tags* provisórias
até que cada artigo possa ter uma revisão manual adequada.
Na verdade, por vários anos, a organização BioASQ
tem [sediado concursos] (http://bioasq.org/) para fazer exatamente isso.

#### Busca


Às vezes, não queremos apenas atribuir cada exemplo a um valor real. No campo da recuperação de informações,
queremos impor uma classificação a um conjunto de itens.
Tome como exemplo a pesquisa na web.
O objetivo é menos determinar se
uma página específica é relevante para uma consulta, mas, em vez disso,
qual dentre a infinidade de resultados de pesquisa é
mais relevante
para um determinado usuário.
Nós realmente nos preocupamos com a ordem dos resultados de pesquisa relevantes
e nosso algoritmo de aprendizagem precisa produzir subconjuntos ordenados
de elementos de um conjunto maior.
Em outras palavras, se formos solicitados a produzir as primeiras 5 letras do alfabeto, há uma diferença
entre retornar "A B C D E" e "C A B E D".
Mesmo que o conjunto de resultados seja o mesmo,
a ordenação dentro do conjunto importa.

Uma possível solução para este problema é primeiro atribuir
para cada elemento no conjunto uma pontuação de relevância correspondente
e, em seguida, para recuperar os elementos com melhor classificação.
[PageRank] (https://en.wikipedia.org/wiki/PageRank),
o molho secreto original por trás do mecanismo de pesquisa do Google
foi um dos primeiros exemplos de tal sistema de pontuação, mas foi
peculiar por não depender da consulta real.
Aqui, eles contaram com um filtro de relevância simples
para identificar o conjunto de itens relevantes
e, em seguida, no PageRank para ordenar esses resultados
que continham o termo de consulta.
Hoje em dia, os mecanismos de pesquisa usam *machine learning* e modelos comportamentais
para obter pontuações de relevância dependentes de consulta.
Existem conferências acadêmicas inteiras dedicadas a este assunto.

#### Sistemas de Recomendação
:label:`subsec_recommender_systems`


Os sistemas de recomendação são outra configuração de problema
que está relacionado à pesquisa e classificação.
Os problemas são semelhantes na medida em que o objetivo
é exibir um conjunto de itens relevantes para o usuário.
A principal diferença é a ênfase em
*personalização*
para usuários específicos no contexto de sistemas de recomendação.
Por exemplo, para recomendações de filmes,
a página de resultados para um fã de ficção científica
e a página de resultados
para um conhecedor das comédias de Peter Sellers podem diferir significativamente.
Problemas semelhantes surgem em outras configurações de recomendação,
por exemplo, para produtos de varejo, música e recomendação de notícias.

Em alguns casos, os clientes fornecem *feedback* explícito comunicando
o quanto eles gostaram de um determinado produto
(por exemplo, as avaliações e resenhas de produtos na Amazon, IMDb e GoodReads).
Em alguns outros casos, eles fornecem *feedback* implícito,
por exemplo, pulando títulos em uma lista de reprodução,
o que pode indicar insatisfação, mas pode apenas indicar
que a música era inadequada no contexto.
Nas formulações mais simples, esses sistemas são treinados
para estimar alguma pontuação,
como uma avaliação estimada
ou a probabilidade de compra,
dado um usuário e um item.

Dado esse modelo,
para qualquer usuário,
poderíamos recuperar o conjunto de objetos com as maiores pontuações,
que pode então ser recomendado ao usuário.
Os sistemas de produção são consideravelmente mais avançados e levam a 
atividade detalhada do usuário e características do item em consideração
ao computar essas pontuações. : numref: `fig_deeplearning_amazon` é um exemplo
de livros de *deep learning* recomendados pela Amazon com base em algoritmos de personalização ajustados para capturar as preferências de alguém.

![Livros de *deep learning* recomendados pela Amazon.](../img/deeplearning-amazon.jpg)
:label:`fig_deeplearning_amazon`

Apesar de seu enorme valor econômico,
sistemas de recomendação
ingenuamente construídos em cima de modelos preditivos
sofrem algumas falhas conceituais graves.
Para começar, observamos apenas *feedback censurado*:
os usuários avaliam preferencialmente os filmes que os consideram fortes.
Por exemplo,
em uma escala de cinco pontos,
você pode notar que os itens recebem muitas classificações de cinco e uma estrela
mas que existem visivelmente poucas avaliações de três estrelas.
Além disso, os hábitos de compra atuais são muitas vezes um resultado
do algoritmo de recomendação atualmente em vigor,
mas os algoritmos de aprendizagem nem sempre levam esse detalhe em consideração.
Assim, é possível que se formem ciclos de feedback
onde um sistema de recomendação preferencialmente empurra um item
que então é considerado melhor (devido a maiores compras)
e, por sua vez, é recomendado com ainda mais frequência.
Muitos desses problemas sobre como lidar com a censura,
incentivos e ciclos de *feedback* são importantes questões abertas de pesquisa.

#### Aprendizagem sequencial


Até agora, vimos problemas em que temos
algum número fixo de entradas a partir dos quais produzimos um número fixo de saídas.
Por exemplo,
consideramos prever os preços das casas a partir de um conjunto fixo de recursos: metragem quadrada, número de quartos,
número de banheiros, tempo de caminhada até o centro.
Também discutimos o mapeamento de uma imagem (de dimensão fixa)
às probabilidades previstas de que pertence a cada
de um número fixo de classes, ou pegando um ID de usuário e um ID de produto,
e prever uma classificação por estrelas. Nesses casos,
uma vez que alimentamos nossa entrada de comprimento fixo
no modelo para gerar uma saída,
o modelo esquece imediatamente o que acabou de ver.

Isso pode ser bom se todas as nossas entradas realmente tiverem as mesmas dimensões
e se as entradas sucessivas realmente não têm nada a ver umas com as outras.
Mas como lidaríamos com trechos de vídeo?
Nesse caso, cada fragmento pode consistir em um número diferente de quadros.
E nosso palpite sobre o que está acontecendo em cada quadro pode ser muito mais forte
se levarmos em consideração os quadros anteriores ou posteriores.
O mesmo vale para a linguagem. Um problema popular de *deep learning*
é tradução automática: a tarefa de ingerir frases
em algum idioma de origem e prevendo sua tradução em outro idioma.

Esses problemas também ocorrem na medicina.
Podemos querer um modelo para monitorar pacientes na unidade de terapia intensiva
e disparar alertas se seus riscos de morte
nas próximas 24 horas excederem algum limite.
Definitivamente, não queremos que este modelo jogue fora
tudo o que sabe sobre o histórico do paciente a cada hora
e apenas fazer suas previsões com base nas medições mais recentes.

Esses problemas estão entre as aplicações mais interessantes de *machine learning*
e são instâncias de *aprendizagem sequencial*.
Eles exigem um modelo para ingerir sequências de entradas
ou para emitir sequências de saídas (ou ambos).
Especificamente,
*sequência para aprendizagem de sequencial* considera os problemas
onde entrada e saída são sequências de comprimento variável,
como tradução automática e transcrição de texto da fala falada.
Embora seja impossível considerar todos os tipos de transformações de sequência,
vale a pena mencionar os seguintes casos especiais.

**Marcação e análise**. Isso envolve anotar uma sequência de texto com atributos.
Em outras palavras, o número de entradas e saídas é essencialmente o mesmo.
Por exemplo, podemos querer saber onde estão os verbos e os sujeitos.
Como alternativa, podemos querer saber quais palavras são as entidades nomeadas.
Em geral, o objetivo é decompor e anotar o texto com base na estrutura
e suposições gramaticais para obter algumas anotações.
Isso parece mais complexo do que realmente é.
Abaixo está um exemplo muito simples de uma frase anotada
com marcas que indicam quais palavras se referem a entidades nomeadas (marcadas como "Ent").

```text
Tom has dinner in Washington with Sally
Ent  -    -    -     Ent      -    Ent
```


**Reconhecimento automático de fala **. Com o reconhecimento de fala, a sequência de entrada
é uma gravação de áudio de um alto-falante (mostrado em: numref: `fig_speech`), e a saída
é a transcrição textual do que o locutor disse.
O desafio é que existem muito mais quadros de áudio
(o som é normalmente amostrado em 8kHz ou 16kHz)
do que texto, ou seja, não há correspondência 1: 1 entre áudio e texto,
já que milhares de amostras podem
correspondem a uma única palavra falada.
Estes são problemas de aprendizagem de sequência a sequência em que a saída é muito mais curta do que a entrada.

![`-D-e-e-p- L-ea-r-ni-ng-` in an audio recording.](../img/speech.png)
:width:`700px`
:label:`fig_speech`

***Text to Speech* (Texto para fala)**. Este é o inverso do reconhecimento automático de fala.
Em outras palavras, a entrada é um texto
e a saída é um arquivo de áudio.
Nesse caso, a saída é muito mais longa do que a entrada.
Embora seja fácil para os humanos reconhecerem um arquivo de áudio ruim,
isso não é tão trivial para computadores.

**Tradução por máquina**. Ao contrário do caso do reconhecimento de voz, onde correspondente
entradas e saídas ocorrem na mesma ordem (após o alinhamento),
na tradução automática, a inversão da ordem pode ser vital.
Em outras palavras, enquanto ainda estamos convertendo uma sequência em outra,
nem o número de entradas e saídas, nem o pedido
de exemplos de dados correspondentes são considerados iguais.
Considere o seguinte exemplo ilustrativo
da tendência peculiar dos alemães
para colocar os verbos no final das frases.

```text
German:           Haben Sie sich schon dieses grossartige Lehrwerk angeschaut?
English:          Did you already check out this excellent tutorial?
Wrong alignment:  Did you yourself already this excellent tutorial looked-at?
```


Muitos problemas relacionados surgem em outras tarefas de aprendizagem.
Por exemplo, determinar a ordem em que um usuário
lê uma página da web é um problema de análise de *layout* bidimensional.
Problemas de diálogo apresentam todos os tipos de complicações adicionais,
onde determinar o que dizer a seguir requer levar em consideração
conhecimento do mundo real e o estado anterior da conversa
através de longas distâncias temporais.
Estas são áreas ativas de pesquisa.


### Aprendizagem não-supervisionada


Todos os exemplos até agora foram relacionados à aprendizagem supervisionada,
ou seja, situações em que alimentamos o modelo com um conjunto de dados gigante
contendo os recursos e os valores de rótulo correspondentes.
Você pode pensar no aluno supervisionado como tendo
um trabalho extremamente especializado e um chefe extremamente banal.
O chefe fica por cima do seu ombro e lhe diz exatamente o que fazer
em todas as situações até que você aprenda a mapear as de situações para ações.
Trabalhar para um chefe assim parece muito chato.
Por outro lado, é fácil agradar a esse chefe.
Você apenas reconhece o padrão o mais rápido possível
e imita suas ações.

De uma forma completamente oposta, pode ser frustrante
trabalhar para um chefe que não tem ideia do que eles querem que você faça.
No entanto, se você planeja ser um cientista de dados, é melhor se acostumar com isso.
O chefe pode simplesmente entregar a você uma pilha gigante de dados e dizer para *fazer ciência de dados com eles!*
Isso parece vago porque é.
Chamamos essa classe de problemas de *aprendizagem não supervisionada*,
e o tipo e número de perguntas que podemos fazer
é limitado apenas pela nossa criatividade.
Abordaremos técnicas de aprendizado não supervisionado
nos capítulos posteriores.
Para abrir seu apetite por enquanto,
descrevemos algumas das seguintes perguntas que você pode fazer.

* Podemos encontrar um pequeno número de protótipos
que resumem os dados com precisão?
Dado um conjunto de fotos, podemos agrupá-las em fotos de paisagens,
fotos de cachorros, bebês, gatos e picos de montanhas?
Da mesma forma, dada uma coleção de atividades de navegação dos usuários,
podemos agrupá-los em usuários com comportamento semelhante?
Esse problema é normalmente conhecido como *clusterização*.
* Podemos encontrar um pequeno número de parâmetros
que capturam com precisão as propriedades relevantes dos dados?
As trajetórias de uma bola são muito bem descritas
pela velocidade, diâmetro e massa da bola.
Os alfaiates desenvolveram um pequeno número de parâmetros
que descrevem a forma do corpo humano com bastante precisão
com o propósito de ajustar roupas.
Esses problemas são chamados de *estimativa de subespaço*.
Se a dependência for linear, é chamada de *análise de componentes principais (principal component analysis --- PCA)*.
* Existe uma representação de objetos (estruturados arbitrariamente)
no espaço euclidiano
de modo que as propriedades simbólicas podem ser bem combinadas?
Isso pode ser usado para descrever entidades e suas relações,
como "Roma" $-$ "Itália" $+$ "França" $=$ "Paris".
* Existe uma descrição das causas comuns
de muitos dos dados que observamos?
Por exemplo, se tivermos dados demográficos
sobre preços de casas, poluição, crime, localização,
educação e salários, podemos descobrir
como eles estão relacionados simplesmente com base em dados empíricos?
Os campos relacionados com *causalidade* e
*modelos gráficos probabilísticos* resolvem este problema.
* Outro importante e empolgante desenvolvimento recente na aprendizagem não supervisionada
é o advento de *redes adversárias geradoras*.
Isso nos dá uma maneira processual de sintetizar dados,
até mesmo dados estruturados complicados, como imagens e áudio.
Os mecanismos estatísticos subjacentes são testes
para verificar se os dados reais e falsos são iguais.

### Interagindo com um Ambiente

So far, we have not discussed where data actually
come from,
or what actually happens when a machine learning model generates an output.
That is because supervised learning and unsupervised learning
do not address these issues in a very sophisticated way.
In either case, we grab a big pile of data upfront,
then set our pattern recognition machines in motion
without ever interacting with the environment again.
Because all of the learning takes place
after the algorithm is disconnected from the environment,
this is sometimes called *offline learning*.
For supervised learning,
the process by considering data collection from an environment looks like :numref:`fig_data_collection`.

![Collecting data for supervised learning from an environment.](../img/data-collection.svg)
:label:`fig_data_collection`

This simplicity of offline learning has its charms.
The upside is that
we can worry about pattern recognition
in isolation, without any distraction from these other problems.
But the downside is that the problem formulation is quite limiting.
If you are more ambitious, or if you grew up reading Asimov's Robot series,
then you might imagine artificially intelligent bots capable
not only of making predictions, but also 
of taking actions in the world.
We want to think about intelligent *agents*, not just predictive models.
This means that
we need to think about choosing *actions*,
not just making predictions.
Moreover, unlike predictions,
actions actually impact the environment.
If we want to train an intelligent agent,
we must account for the way its actions might
impact the future observations of the agent.

Considering the interaction with an environment
opens a whole set of new modeling questions.
The following are just a few examples.

* Does the environment remember what we did previously?
* Does the environment want to help us, e.g., a user reading text into a speech recognizer?
* Does the environment want to beat us, i.e., an adversarial setting like spam filtering (against spammers) or playing a game (vs. an opponent)?
* Does the environment not care?
* Does the environment have shifting dynamics? For example, does future data always resemble the past or do the patterns change over time, either naturally or in response to our automated tools?

This last question raises the problem of *distribution shift*,
when training and test data are different.
It is a problem that most of us have experienced
when taking exams written by a lecturer,
while the homework was composed by his teaching assistants.
Next, we will briefly describe reinforcement learning,
a setting that explicitly considers interactions with an environment.

### Reinforcement Learning

If you are interested in using machine learning
to develop an agent that interacts with an environment
and takes actions, then you are probably going to wind up
focusing on *reinforcement learning*.
This might include applications to robotics,
to dialogue systems, 
and even to developing artificial intelligence (AI)
for video games.
*Deep reinforcement learning*, which applies
deep learning to reinforcement learning problems,
has surged in popularity.
The breakthrough deep Q-network that beat humans at Atari games using only the visual input,
and the AlphaGo program that dethroned the world champion at the board game Go are two prominent examples.

Reinforcement learning gives a very general statement of a problem,
in which an agent interacts with an environment over a series of time steps.
At each time step, 
the agent receives some *observation* 
from the environment and must choose an *action*
that is subsequently transmitted back to the environment
via some mechanism (sometimes called an actuator).
Finally, the agent receives a reward from the environment.
This process is illustrated in :numref:`fig_rl-environment`.
The agent then receives a subsequent observation,
and chooses a subsequent action, and so on.
The behavior of an reinforcement learning agent is governed by a policy.
In short, a *policy* is just a function that maps
from observations of the environment to actions.
The goal of reinforcement learning is to produce a good policy.

![The interaction between reinforcement learning and an environment.](../img/rl-environment.svg)
:label:`fig_rl-environment`

It is hard to overstate the generality of the reinforcement learning framework.
For example, we can cast any supervised learning problem as a reinforcement learning problem.
Say we had a classification problem.
We could create a reinforcement learning agent with one action corresponding to each class.
We could then create an environment which gave a reward
that was exactly equal to the loss function
from the original supervised learning problem.

That being said, reinforcement learning can also address many problems
that supervised learning cannot.
For example, in supervised learning we always expect
that the training input comes associated with the correct label.
But in reinforcement learning, we do not assume that for each observation 
the environment tells us the optimal action.
In general, we just get some reward.
Moreover, the environment may not even tell us which actions led to the reward.

Consider for example the game of chess.
The only real reward signal comes at the end of the game
when we either win, which we might assign a reward of 1,
or when we lose, which we could assign a reward of -1.
So reinforcement learners must deal with the *credit assignment* problem:
determining which actions to credit or blame for an outcome.
The same goes for an employee who gets a promotion on October 11.
That promotion likely reflects a large number
of well-chosen actions over the previous year.
Getting more promotions in the future requires figuring out
what actions along the way led to the promotion.

Reinforcement learners may also have to deal
with the problem of partial observability.
That is, the current observation might not
tell you everything about your current state.
Say a cleaning robot found itself trapped
in one of many identical closets in a house.
Inferring the precise location (and thus state) of the robot
might require considering its previous observations before entering the closet.

Finally, at any given point, reinforcement learners
might know of one good policy,
but there might be many other better policies
that the agent has never tried.
The reinforcement learner must constantly choose
whether to *exploit* the best currently-known strategy as a policy,
or to *explore* the space of strategies,
potentially giving up some short-run reward in exchange for knowledge.

The general reinforcement learning problem
is a very general setting.
Actions affect subsequent observations.
Rewards are only observed corresponding to the chosen actions.
The environment may be either fully or partially observed.
Accounting for all this complexity at once may ask too much of researchers.
Moreover, not every practical problem exhibits all this complexity.
As a result, researchers have studied a number of
special cases of reinforcement learning problems.

When the environment is fully observed,
we call the reinforcement learning problem a *Markov decision process*.
When the state does not depend on the previous actions,
we call the problem a *contextual bandit problem*.
When there is no state, just a set of available actions
with initially unknown rewards, this problem
is the classic *multi-armed bandit problem*.

## Roots

We have just reviewed
a small subset of problems that machine learning 
can address.
For a diverse set of machine learning problems,
deep learning provides powerful tools for solving them.
Although many deep learning methods
are recent inventions,
the core idea of programming with data and neural networks (names of many deep learning models)
has been studied for centuries.
In fact,
humans have held the desire to analyze data
and to predict future outcomes for long
and much of natural science has its roots in this.
For instance, the Bernoulli distribution is named after
[Jacob Bernoulli (1655--1705)](https://en.wikipedia.org/wiki/Jacob_Bernoulli), and the Gaussian distribution was discovered
by [Carl Friedrich Gauss (1777--1855)](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss).
He invented, for instance, the least mean squares algorithm,
which is still used today for countless problems
from insurance calculations to medical diagnostics.
These tools gave rise to an experimental approach
in the natural sciences---for instance, Ohm's law
relating current and voltage in a resistor
is perfectly described by a linear model.

Even in the middle ages, mathematicians had a keen intuition of estimates.
For instance, the geometry book of [Jacob Köbel (1460--1533)](https://www.maa.org/press/periodicals/convergence/mathematical-treasures-jacob-kobels-geometry) illustrates
averaging the length of 16 adult men's feet to obtain the average foot length.

![Estimating the length of a foot.](../img/koebel.jpg)
:width:`500px`
:label:`fig_koebel`

:numref:`fig_koebel` illustrates how this estimator works.
The 16 adult men were asked to line up in a row, when leaving the church.
Their aggregate length was then divided by 16
to obtain an estimate for what now amounts to 1 foot.
This "algorithm" was later improved to deal with misshapen feet---the
2 men with the shortest and longest feet respectively were sent away,
averaging only over the remainder.
This is one of the earliest examples of the trimmed mean estimate.

Statistics really took off with the collection and availability of data.
One of its titans, [Ronald Fisher (1890--1962)](https://en.wikipedia.org/wiki/Ronald_Fisher),
contributed significantly to its theory
and also its applications in genetics.
Many of his algorithms (such as linear discriminant analysis)
and formula (such as the Fisher information matrix)
are still in frequent use today. 
In fact,
even the Iris dataset
that Fisher released in 1936 is still used sometimes
to illustrate machine learning algorithms.
He was also a proponent of eugenics,
which should remind us that the morally dubious use of data science
has as long and enduring a history as its productive use
in industry and the natural sciences.

A second influence for machine learning came from information theory by
[Claude Shannon (1916--2001)](https://en.wikipedia.org/wiki/Claude_Shannon) and the theory of computation via [Alan Turing (1912--1954)](https://en.wikipedia.org/wiki/Alan_Turing).
Turing posed the question "can machines think?”
in his famous paper *Computing Machinery and Intelligence* :cite:`Turing.1950`.
In what he described as the Turing test, a machine
can be considered *intelligent* if it is difficult
for a human evaluator to distinguish between the replies
from a machine and a human based on textual interactions.

Another influence can be found in neuroscience and psychology.
After all, humans clearly exhibit intelligent behavior.
It is thus only reasonable to ask whether one could explain
and possibly reverse engineer this capacity.
One of the oldest algorithms inspired in this fashion
was formulated by [Donald Hebb (1904--1985)](https://en.wikipedia.org/wiki/Donald_O._Hebb).
In his groundbreaking book *The Organization of Behavior* :cite:`Hebb.Hebb.1949`,
he posited that neurons learn by positive reinforcement.
This became known as the Hebbian learning rule.
It is the prototype of Rosenblatt's perceptron learning algorithm
and it laid the foundations of many stochastic gradient descent algorithms
that underpin deep learning today: reinforce desirable behavior
and diminish undesirable behavior to obtain good settings
of the parameters in a neural network.

Biological inspiration is what gave *neural networks* their name.
For over a century (dating back to the models of Alexander Bain, 1873
and James Sherrington, 1890), researchers have tried to assemble
computational circuits that resemble networks of interacting neurons.
Over time, the interpretation of biology has become less literal
but the name stuck. At its heart, lie a few key principles
that can be found in most networks today:

* The alternation of linear and nonlinear processing units, often referred to as *layers*.
* The use of the chain rule (also known as *backpropagation*) for adjusting parameters in the entire network at once.

After initial rapid progress, research in neural networks
languished from around 1995 until 2005.
This was mainly due to two reasons.
First, training a network is computationally very expensive.
While random-access memory was plentiful at the end of the past century,
computational power was scarce.
Second, datasets were relatively small.
In fact, Fisher's Iris dataset from 1932
was a popular tool for testing the efficacy of algorithms.
The MNIST dataset with its 60000 handwritten digits was considered huge.

Given the scarcity of data and computation,
strong statistical tools such as kernel methods,
decision trees and graphical models proved empirically superior.
Unlike neural networks, they did not require weeks to train
and provided predictable results with strong theoretical guarantees.


## The Road to Deep Learning

Much of this changed with 
the ready availability of large amounts of data,
due to the World Wide Web, 
the advent of companies serving
hundreds of millions of users online, 
a dissemination of cheap, high-quality sensors, 
cheap data storage (Kryder's law),
and cheap computation (Moore's law), in particular in the form of GPUs, originally engineered for computer gaming.
Suddenly algorithms and models that seemed computationally infeasible
became relevant (and vice versa).
This is best illustrated in :numref:`tab_intro_decade`.

:Dataset vs. computer memory and computational power

|Decade|Dataset|Memory|Floating point calculations per second|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (House prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (Nvidia C2050)|
|2020|1 T (social network)|100 GB|1 PF (Nvidia DGX-2)|
:label:`tab_intro_decade`

It is evident that random-access memory has not kept pace with the growth in data.
At the same time, the increase in computational power
has outpaced that of the data available.
This means that statistical models need to become more memory efficient
(this is typically achieved by adding nonlinearities)
while simultaneously being able to spend more time
on optimizing these parameters, due to an increased computational budget.
Consequently, the sweet spot in machine learning and statistics
moved from (generalized) linear models and kernel methods to deep neural networks.
This is also one of the reasons why many of the mainstays
of deep learning, such as multilayer perceptrons
:cite:`McCulloch.Pitts.1943`, convolutional neural networks
:cite:`LeCun.Bottou.Bengio.ea.1998`, long short-term memory
:cite:`Hochreiter.Schmidhuber.1997`,
and Q-Learning :cite:`Watkins.Dayan.1992`,
were essentially "rediscovered" in the past decade,
after laying comparatively dormant for considerable time.

The recent progress in statistical models, applications, and algorithms
has sometimes been likened to the Cambrian explosion:
a moment of rapid progress in the evolution of species.
Indeed, the state of the art is not just a mere consequence
of available resources, applied to decades old algorithms.
Note that the list below barely scratches the surface
of the ideas that have helped researchers achieve tremendous progress
over the past decade.


* Novel methods for capacity control, such as *dropout*
  :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`,
  have helped to mitigate the danger of overfitting.
  This was achieved by applying noise injection :cite:`Bishop.1995`
  throughout the neural network, replacing weights by random variables
  for training purposes.
* Attention mechanisms solved a second problem
  that had plagued statistics for over a century:
  how to increase the memory and complexity of a system without
  increasing the number of learnable parameters.
  Researchers found an elegant solution
  by using what can only be viewed as a learnable pointer structure :cite:`Bahdanau.Cho.Bengio.2014`.
  Rather than having to remember an entire text sequence, e.g.,
  for machine translation in a fixed-dimensional representation,
  all that needed to be stored was a pointer to the intermediate state
  of the translation process. This allowed for significantly
  increased accuracy for long sequences, since the model
  no longer needed to remember the entire sequence before
  commencing the generation of a new sequence.
* Multi-stage designs, e.g., via the memory networks 
  :cite:`Sukhbaatar.Weston.Fergus.ea.2015` and the neural programmer-interpreter :cite:`Reed.De-Freitas.2015`
  allowed statistical modelers to describe iterative approaches to reasoning. These tools allow for an internal state of the deep neural network
  to be modified repeatedly, thus carrying out subsequent steps
  in a chain of reasoning, similar to how a processor
  can modify memory for a computation.
* Another key development was the invention of generative adversarial networks
  :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`.
  Traditionally, statistical methods for density estimation
  and generative models focused on finding proper probability distributions
  and (often approximate) algorithms for sampling from them.
  As a result, these algorithms were largely limited by the lack of
  flexibility inherent in the statistical models.
  The crucial innovation in generative adversarial networks was to replace the sampler
  by an arbitrary algorithm with differentiable parameters.
  These are then adjusted in such a way that the discriminator
  (effectively a two-sample test) cannot distinguish fake from real data.
  Through the ability to use arbitrary algorithms to generate data,
  it opened up density estimation to a wide variety of techniques.
  Examples of galloping Zebras :cite:`Zhu.Park.Isola.ea.2017`
  and of fake celebrity faces :cite:`Karras.Aila.Laine.ea.2017`
  are both testimony to this progress.
  Even amateur doodlers can produce
  photorealistic images based on just sketches that describe
  how the layout of a scene looks like :cite:`Park.Liu.Wang.ea.2019`.
* In many cases, a single GPU is insufficient to process
  the large amounts of data available for training.
  Over the past decade the ability to build parallel and
  distributed training algorithms has improved significantly.
  One of the key challenges in designing scalable algorithms
  is that the workhorse of deep learning optimization,
  stochastic gradient descent, relies on relatively
  small minibatches of data to be processed.
  At the same time, small batches limit the efficiency of GPUs.
  Hence, training on 1024 GPUs with a minibatch size of,
  say 32 images per batch amounts to an aggregate minibatch
  of about 32000 images. Recent work, first by Li :cite:`Li.2017`,
  and subsequently by :cite:`You.Gitman.Ginsburg.2017`
  and :cite:`Jia.Song.He.ea.2018` pushed the size up to 64000 observations,
  reducing training time for the ResNet-50 model on the ImageNet dataset to less than 7 minutes.
  For comparison---initially training times were measured in the order of days.
* The ability to parallelize computation has also contributed quite crucially
  to progress in reinforcement learning, at least whenever simulation is an
  option. This has led to significant progress in computers achieving
  superhuman performance in Go, Atari games, Starcraft, and in physics
  simulations (e.g., using MuJoCo). See e.g.,
  :cite:`Silver.Huang.Maddison.ea.2016` for a description
  of how to achieve this in AlphaGo. In a nutshell,
  reinforcement learning works best if plenty of (state, action, reward) triples are available, i.e., whenever it is possible to try out lots of things to learn how they relate to each
  other. Simulation provides such an avenue.
* Deep learning frameworks have played a crucial role
  in disseminating ideas. The first generation of frameworks
  allowing for easy modeling encompassed
  [Caffe](https://github.com/BVLC/caffe),
  [Torch](https://github.com/torch), and
  [Theano](https://github.com/Theano/Theano).
  Many seminal papers were written using these tools.
  By now, they have been superseded by
  [TensorFlow](https://github.com/tensorflow/tensorflow) (often used via its high level API [Keras](https://github.com/keras-team/keras)), [CNTK](https://github.com/Microsoft/CNTK), [Caffe 2](https://github.com/caffe2/caffe2), and [Apache MXNet](https://github.com/apache/incubator-mxnet). The third generation of tools, namely imperative tools for deep learning,
  was arguably spearheaded by [Chainer](https://github.com/chainer/chainer),
  which used a syntax similar to Python NumPy to describe models.
  This idea was adopted by both [PyTorch](https://github.com/pytorch/pytorch),
  the [Gluon API](https://github.com/apache/incubator-mxnet) of MXNet, and [Jax](https://github.com/google/jax).


The division of labor between system researchers building better tools
and statistical modelers building better neural networks
has greatly simplified things. For instance,
training a linear logistic regression model
used to be a nontrivial homework problem,
worthy to give to new machine learning
Ph.D. students at Carnegie Mellon University in 2014.
By now, this task can be accomplished with less than 10 lines of code,
putting it firmly into the grasp of programmers.

## Success Stories

AI has a long history of delivering results
that would be difficult to accomplish otherwise.
For instance, 
the mail sorting systems
using optical character recognition
have been deployed since the 1990s.
This is, after all, the source of the famous MNIST dataset  of handwritten digits.
The same applies to reading checks for bank deposits and scoring
creditworthiness of applicants.
Financial transactions are checked for fraud automatically.
This forms the backbone of many e-commerce payment systems,
such as PayPal, Stripe, AliPay, WeChat, Apple, Visa, and MasterCard.
Computer programs for chess have been competitive for decades.
Machine learning feeds search, recommendation, personalization,
and ranking on the Internet.
In other words, machine learning is pervasive, albeit often hidden from sight.

It is only recently that AI
has been in the limelight, mostly due to
solutions to problems
that were considered intractable previously
and that are directly related to consumers.
Many of such advances are attributed to deep learning.

* Intelligent assistants, such as Apple's Siri, Amazon's Alexa, and Google's
  assistant, are able to answer spoken questions with a reasonable degree of
  accuracy. This includes menial tasks such as turning on light switches (a boon to the disabled) up to making barber's appointments and offering phone support dialog. This is likely the most noticeable sign that AI is affecting our lives.
* A key ingredient in digital assistants is the ability to recognize speech
  accurately. Gradually the accuracy of such systems has increased to the point
  where they reach human parity for certain
  applications :cite:`Xiong.Wu.Alleva.ea.2018`.
* Object recognition likewise has come a long way. Estimating the object in a
  picture was a fairly challenging task in 2010. On the ImageNet benchmark researchers from NEC Labs and University of Illinois at Urbana-Champaign achieved a top-5 error rate of 28% :cite:`Lin.Lv.Zhu.ea.2010`. By 2017,
  this error rate was reduced to 2.25% :cite:`Hu.Shen.Sun.2018`. Similarly, stunning
  results have been achieved for identifying birds or diagnosing skin cancer.
* Games used to be a bastion of human intelligence.
  Starting from TD-Gammon, a program for playing backgammon using temporal difference reinforcement learning, algorithmic and computational progress has led to algorithms
  for a wide range of applications. Unlike backgammon,
  chess has a much more complex state space and set of actions.
  DeepBlue beat Garry Kasparov using massive parallelism,
  special-purpose hardware and efficient search through the game tree :cite:`Campbell.Hoane-Jr.Hsu.2002`.
  Go is more difficult still, due to its huge state space.
  AlphaGo reached human parity in 2015, using deep learning combined with Monte Carlo tree sampling :cite:`Silver.Huang.Maddison.ea.2016`.
  The challenge in Poker was that the state space is
  large and it is not fully observed (we do not know the opponents'
  cards). Libratus exceeded human performance in Poker using efficiently
  structured strategies :cite:`Brown.Sandholm.2017`.
  This illustrates the impressive progress in games
  and the fact that advanced algorithms played a crucial part in them.
* Another indication of progress in AI is the advent of self-driving cars
  and trucks. While full autonomy is not quite within reach yet,
  excellent progress has been made in this direction,
  with companies such as Tesla, NVIDIA,
  and Waymo shipping products that enable at least partial autonomy.
  What makes full autonomy so challenging is that proper driving
  requires the ability to perceive, to reason and to incorporate rules
  into a system. At present, deep learning is used primarily
  in the computer vision aspect of these problems.
  The rest is heavily tuned by engineers.



Again, the above list barely scratches the surface of where machine learning has impacted practical applications. For instance, robotics, logistics, computational biology, particle physics, and astronomy owe some of their most impressive recent advances at least in parts to machine learning. Machine learning is thus becoming a ubiquitous tool for engineers and scientists.

Frequently, the question of the AI apocalypse, or the AI singularity
has been raised in non-technical articles on AI.
The fear is that somehow machine learning systems
will become sentient and decide independently from their programmers
(and masters) about things that directly affect the livelihood of humans.
To some extent, AI already affects the livelihood of humans
in an immediate way:
creditworthiness is assessed automatically,
autopilots mostly navigate vehicles, decisions about
whether to grant bail use statistical data as input.
More frivolously, we can ask Alexa to switch on the coffee machine.

Fortunately, we are far from a sentient AI system
that is ready to manipulate its human creators (or burn their coffee).
First, AI systems are engineered, trained and deployed in a specific,
goal-oriented manner. While their behavior might give the illusion
of general intelligence, it is a combination of rules, heuristics
and statistical models that underlie the design.
Second, at present tools for *artificial general intelligence*
simply do not exist that are able to improve themselves,
reason about themselves, and that are able to modify,
extend, and improve their own architecture
while trying to solve general tasks.

A much more pressing concern is how AI is being used in our daily lives.
It is likely that many menial tasks fulfilled by truck drivers
and shop assistants can and will be automated.
Farm robots will likely reduce the cost for organic farming
but they will also automate harvesting operations.
This phase of the industrial revolution
may have profound consequences on large swaths of society,
since truck drivers and shop assistants are some
of the most common jobs in many countries.
Furthermore, statistical models, when applied without care
can lead to racial, gender, or age bias and raise
reasonable concerns about procedural fairness
if automated to drive consequential decisions.
It is important to ensure that these algorithms are used with care.
With what we know today, this strikes us a much more pressing concern
than the potential of malevolent superintelligence to destroy humanity.


## Characteristics

Thus far, we have talked about machine learning broadly, which is both a branch of AI and an approach to AI.
Though deep learning is a subset of machine learning,
the dizzying set of algorithms and applications makes it difficult to assess what specifically the ingredients for deep learning might be. 
This is as difficult as trying to pin down required ingredients for pizza since almost every component is substitutable.

As we have described, machine learning can
use data to learn transformations between inputs and outputs,
such as transforming audio into text in speech recognition.
In doing so, it is often necessary to represent data in a way suitable for algorithms to transform such representations into the output.
*Deep learning* is *deep* in precisely the sense
that its models
learn many *layers* of transformations,
where each layer offers the representation
at one level.
For example,
layers near the input may represent 
low-level details of the data,
while layers closer to the classification output
may represent more abstract concepts used for discrimination.
Since *representation learning* aims at
finding the representation itself,
deep learning can be referred to as multi-level
representation learning.

The problems that we have discussed so far, such as learning
from the raw audio signal, 
the raw pixel values of images,
or mapping between sentences of arbitrary lengths and
their counterparts in foreign languages,
are those
where deep learning excels and where traditional 
machine learning
methods falter.
It turns out that these many-layered models
are capable of addressing low-level perceptual data
in a way that previous tools could not.
Arguably the most significant commonality in deep learning methods is the use of *end-to-end training*. 
That is, rather than assembling a system based on components that are individually tuned, one builds the system and then tunes their performance jointly.
For instance, in computer vision scientists used to separate the process of *feature engineering* from the process of building machine learning models. The Canny edge detector :cite:`Canny.1987` and Lowe's SIFT feature extractor :cite:`Lowe.2004` reigned supreme for over a decade as algorithms for mapping images into feature vectors.
In bygone days, the crucial part of applying machine learning to these problems
consisted of coming up with manually-engineered ways
of transforming the data into some form amenable to shallow models.
Unfortunately, there is only so little that humans can accomplish by ingenuity in comparison with a consistent evaluation over millions of choices carried out automatically by an algorithm.
When deep learning took over,
these feature extractors were replaced by automatically tuned filters, yielding superior accuracy.

Thus,
one key advantage of deep learning is that it replaces not
only the shallow models at the end of traditional learning pipelines,
but also the labor-intensive process of 
feature engineering.
Moreover, by replacing much of the domain-specific preprocessing,
deep learning has eliminated many of the boundaries
that previously separated computer vision, speech recognition,
natural language processing, medical informatics, and other application areas,
offering a unified set of tools for tackling diverse problems.

Beyond end-to-end training, 
we are experiencing a transition from parametric statistical descriptions to fully nonparametric models. When data are scarce, one needs to rely on simplifying assumptions about reality in order to obtain useful models. When data are abundant, this can be replaced by nonparametric models that fit reality more accurately. To some extent, this mirrors the progress that physics experienced in the middle of the previous century with the availability of computers. Rather than solving parametric approximations of how electrons behave by hand, one can now resort to numerical simulations of the associated partial differential equations. This has led to much more accurate models, albeit often at the expense of explainability.

Another difference to previous work is the acceptance of suboptimal solutions, dealing with nonconvex nonlinear optimization problems, and the willingness to try things before proving them. This newfound empiricism in dealing with statistical problems, combined with a rapid influx of talent has led to rapid progress of practical algorithms, albeit in many cases at the expense of modifying and re-inventing tools that existed for decades.

In the end, the deep learning community prides itself of sharing tools across academic and corporate boundaries, releasing many excellent libraries, statistical models, and trained networks as open source.
It is in this spirit that the notebooks forming this book are freely available for distribution and use. We have worked hard to lower the barriers of access for everyone to learn about deep learning and we hope that our readers will benefit from this.



## Summary

* Machine learning studies how computer systems can leverage experience (often data) to improve performance at specific tasks. It combines ideas from statistics, data mining, and optimization. Often, it is used as a means of implementing AI solutions.
* As a class of machine learning, representational learning focuses on how to automatically find the appropriate way to represent data. Deep learning is multi-level representation learning through learning many layers of transformations.
* Deep learning replaces not only the shallow models at the end of traditional machine learning pipelines, but also the labor-intensive process of feature engineering. 
* Much of the recent progress in deep learning has been triggered by an abundance of data arising from cheap sensors and Internet-scale applications, and by significant progress in computation, mostly through GPUs.
* Whole system optimization is a key component in obtaining high performance. The availability of efficient deep learning frameworks has made design and implementation of this significantly easier.

## Exercises

1. Which parts of code that you are currently writing could be "learned", i.e., improved by learning and automatically determining design choices that are made in your code? Does your code include heuristic design choices?
1. Which problems that you encounter have many examples for how to solve them, yet no specific way to automate them? These may be prime candidates for using deep learning.
1. Viewing the development of AI as a new industrial revolution, what is the relationship between algorithms and data? Is it similar to steam engines and coal? What is the fundamental difference?
1. Where else can you apply the end-to-end training approach, such as in :numref:`fig_ml_loop`, physics, engineering, and econometrics?

[Discussions](https://discuss.d2l.ai/t/22)
<!--stackedit_data:
eyJoaXN0b3J5IjpbODE3ODk3NTA2LC0yNDYzNDM1OTEsLTQ3NT
AyODU4MSwxNzI1MjgwOTk0LDE4ODkyNTI3NjcsLTExNjQzMjQ4
ODEsOTg3NDIxMDcxLDE1NjQwNzg3MDksMjEwNzA2Mjk5NiwtMT
A2NDI5MDE5OSw3MjE1MjgwOTAsLTk0MTY2MTMyMCwtMjI4ODMy
MzMxLC0xMDc4NTg0NDcxLC01MzQxODE2MTUsMTk1MTM0NTYyMS
wtMTY4NzI3Mzg2OCwtMTg2NzM2NTQ5NCw0OTYxMjc0MzgsMTc2
MjYzMDIyXX0=
-->
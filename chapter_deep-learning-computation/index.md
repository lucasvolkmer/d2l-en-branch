# *Deep Learning* Computacional
:label:`chap_computation`

Junto com conjuntos de dados gigantes e hardware poderoso,
ótimas ferramentas de software desempenharam um papel indispensável
no rápido progresso do *Deep Learning*.
Começando com a revolucionária biblioteca Theano lançada em 2007,
ferramentas de código aberto flexíveis têm permitido aos pesquisadores
para prototipar modelos rapidamente, evitando trabalho repetitivo
ao reciclar componentes padrão
ao mesmo tempo em que mantém a capacidade de fazer modificações de baixo nível.
Com o tempo, as bibliotecas de *Deep Learning* evoluíram
para oferecer abstrações cada vez mais grosseiras.
Assim como os designers de semicondutores passaram a especificar transistores
para circuitos lógicos para escrever código,
pesquisadores de redes neurais deixaram de pensar sobre
o comportamento de neurônios artificiais individuais
para conceber redes em termos de camadas inteiras,
e agora frequentemente projeta arquiteturas com *blocos* muito mais grosseiros em mente.

So far, we have introduced some basic machine learning concepts,
ramping up to fully-functional deep learning models.
In the last chapter,
we implemented each component of an MLP from scratch
and even showed how to leverage high-level APIs
to roll out the same models effortlessly.
To get you that far that fast, we *called upon* the libraries,
but skipped over more advanced details about *how they work*.
In this chapter, we will peel back the curtain,
digging deeper into the key components of deep learning computation,
namely model construction, parameter access and initialization,
designing custom layers and blocks, reading and writing models to disk,
and leveraging GPUs to achieve dramatic speedups.
These insights will move you from *end user* to *power user*,
giving you the tools needed to reap the benefits
of a mature deep learning library while retaining the flexibility
to implement more complex models, including those you invent yourself!
While this chapter does not introduce any new models or datasets,
the advanced modeling chapters that follow rely heavily on these techniques.

Até agora, apresentamos alguns conceitos básicos de aprendizado de máquina,
evoluindo para modelos de *Deep Learning* totalmente funcionais.
No último capítulo,
implementamos cada componente de um MLP do zero
e até mostrou como aproveitar APIs de alto nível
para lançar os mesmos modelos sem esforço.
Para chegar tão longe tão rápido, nós * chamamos * as bibliotecas,
mas pulei detalhes mais avançados sobre *como eles funcionam*.
Neste capítulo, vamos abrir a cortina,
aprofundando os principais componentes da computação de *Deep Learning*,
ou seja, construção de modelo, acesso de parâmetro e inicialização,
projetando camadas e blocos personalizados, lendo e gravando modelos em disco,
e aproveitando GPUs para obter acelerações dramáticas.
Esses insights o moverão de *usuário final* para *usuário avançado*,
dando a você as ferramentas necessárias para colher os benefícios
de uma biblioteca de aprendizagem profunda madura, mantendo a flexibilidade
para implementar modelos mais complexos, incluindo aqueles que você mesmo inventa!
Embora este capítulo não introduza nenhum novo modelo ou conjunto de dados,
os capítulos de modelagem avançada que se seguem dependem muito dessas técnicas.

```toc
:maxdepth: 2

model-construction
parameters
deferred-init
custom-layer
read-write
use-gpu
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIxMTkyMTg3MTIsLTE1NTEzMzc3NjddfQ
==
-->
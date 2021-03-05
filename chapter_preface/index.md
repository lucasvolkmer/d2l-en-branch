# Prefácio

Apenas alguns anos atrás, não havia legiões de cientistas de *deep learning*  desenvolvendo produtos e serviços inteligentes em grandes empresas e *startups*.
Quando o mais jovem entre nós (os autores) entrou no campo,  o *machine learning* não comandava as manchetes dos jornais diários.
Nossos pais não faziam ideia do que era *machine learning*,  muito menos por que podemos preferir isso a uma carreira em medicina ou direito.
Machine learning era uma disciplina acadêmica voltada para o futuro  com um conjunto restrito de aplicações do mundo real.
E essas aplicações, por exemplo, reconhecimento de voz e visão computacional,  exigiam tanto conhecimento de domínio que muitas vezes eram considerados  como áreas inteiramente separadas para as quais o aprendizado de máquina era um pequeno componente.
Redes neurais, então, os antecedentes dos modelos de aprendizagem profunda  nos quais nos concentramos neste livro, eram considerados ferramentas obsoletas.


Apenas nos últimos cinco anos, o *deep learning* pegou o mundo de surpresa,
impulsionando o rápido progresso em campos tão diversos como a visão computacional,
processamento de linguagem natural, reconhecimento automático de fala,
aprendizagem por reforço e modelagem estatística.
Com esses avanços em mãos, agora podemos construir carros que se dirigem sozinhos
com mais autonomia do que nunca (e menos autonomia
do que algumas empresas podem fazer você acreditar),
sistemas de resposta inteligente que redigem automaticamente os e-mails mais comuns,
ajudando as pessoas a se livrarem de caixas de entrada opressivamente grandes,
e agentes de *software* que dominam os melhores humanos do mundo
em jogos de tabuleiro como Go, um feito que se pensava estar a décadas de distância.
Essas ferramentas já exercem impactos cada vez maiores na indústria e na sociedade,
mudando a forma como os filmes são feitos, as doenças são diagnosticadas,
e desempenhando um papel crescente nas ciências básicas --- da astrofísica à biologia.



## Sobre Este Livro
Esse livro representa nossa tentativa de tornar o *deep learning* acessível, lhes ensinando os *conceitos*, o *contexto* e o *código*.


### Um Meio (?) Combinando Código, Matemática e HTML

Para que qualquer tecnologia de computação alcance seu impacto total,
deve ser bem compreendido, bem documentado e apoiado por
ferramentas maduras e bem conservadas.
As ideias-chave devem ser claramente destiladas,
minimizando o tempo de integração necessário para atualizar os novos praticantes.
Bibliotecas maduras devem automatizar tarefas comuns,
e o código exemplar deve tornar mais fácil para os profissionais
para modificar, aplicar e estender aplicativos comuns para atender às suas necessidades.
Considere os aplicativos da Web dinâmicos como exemplo.
Apesar de um grande número de empresas, como a Amazon,
desenvolver aplicativos da web baseados em banco de dados de sucesso na década de 1990,
o potencial desta tecnologia para auxiliar empreendedores criativos
foi percebido em um grau muito maior nos últimos dez anos,
devido em parte ao desenvolvimento de *frameworks* poderosos e bem documentados.


Testar o potencial do *deep learning* apresenta desafios únicos
porque qualquer aplicativo reúne várias disciplinas.
Aplicar o *deep learning* requer compreensão simultânea
(i) as motivações para definir um problema de uma maneira particular;
(ii) a matemática de uma dada abordagem de modelagem;
(iii) os algoritmos de otimização para ajustar os modelos aos dados;
e (iv) a engenharia necessária para treinar modelos de forma eficiente,
navegando nas armadilhas da computação numérica
e obter o máximo do *hardware* disponível.
Ensinar as habilidades de pensamento crítico necessárias para formular problemas,
a matemática para resolvê-los e as ferramentas de *software* para implementar tais
soluções em um só lugar apresentam desafios formidáveis.
Nosso objetivo neste livro é apresentar um recurso unificado
para trazer os praticantes em potencial.

Na época em que começamos o projeto deste livro,
não havia recursos que simultaneamente
(i) estavam em dia; (ii) cobriam toda a largura
de *machine learning* moderno com profundidade técnica substancial;
e (iii) intercalassem exposição da qualidade que se espera
de um livro envolvente com o código limpo executável
que se espera encontrar em tutoriais práticos.
Encontramos muitos exemplos de código para
como usar um determinado *framework* de aprendizado profundo
(por exemplo, como fazer computação numérica básica com matrizes no *TensorFlow*)
ou para a implementação de técnicas particulares
(por exemplo, *snippets* de código para LeNet, AlexNet, ResNets, etc)
espalhados por vários posts de blog e repositórios GitHub.
No entanto, esses exemplos normalmente se concentram em
*como* implementar uma determinada abordagem,
mas deixou de fora a discussão de *por que* certas decisões algorítmicas são feitas.
Embora alguns recursos interativos tenham surgido esporadicamente
para abordar um tópico específico, por exemplo, as postagens de blog envolventes
publicado no site [Distill] (http://distill.pub), ou blogs pessoais,
eles cobriram apenas tópicos selecionados no aprendizado profundo,
e muitas vezes não tinham código associado.
Por outro lado, embora vários livros tenham surgido,
mais notavelmente: cite: `Goodfellow.Bengio.Courville.2016`,
que oferece uma pesquisa abrangente dos conceitos por trás do aprendizado profundo,
esses recursos não combinam com as descrições
às realizações dos conceitos no código,
às vezes deixando os leitores sem noção de como implementá-los.
Além disso, muitos recursos estão escondidos atrás dos *paywalls*
de fornecedores de cursos comerciais.

We set out to create a resource that could
(i) be freely available for everyone;
(ii) offer sufficient technical depth to provide a starting point on the path
to actually becoming an applied machine learning scientist;
(iii) include runnable code, showing readers *how* to solve problems in practice;
(iv) allow for rapid updates, both by us
and also by the community at large;
and (v) be complemented by a [forum](http://discuss.d2l.ai)
for interactive discussion of technical details and to answer questions.

Propusemo-nos a criar um recurso que pudesse
(i) estar disponível gratuitamente para todos;
(ii) oferecer profundidade técnica suficiente para fornecer um ponto de partida no caminho
para realmente se tornar um cientista de *machine learning* aplicado;
(iii) incluir código executável, mostrando aos leitores *como* resolver problemas na prática;
(iv) permitir atualizações rápidas, tanto por nós
e também pela comunidade em geral;
e (v) ser complementado por um [fórum] (http://discuss.d2l.ai)
para uma discussão interativa de detalhes técnicos e para responder a perguntas.

Esses objetivos costumavam estar em conflito.
Equações, teoremas e citações são melhor gerenciados e apresentados em LaTeX.
O código é melhor descrito em Python.
E as páginas da web são nativas em HTML e JavaScript.
Além disso, queremos que o conteúdo seja
acessível tanto como código executável, como livro físico,
como um PDF para *download* e na Internet como um site.
No momento não existem ferramentas e nenhum *workflow*
perfeitamente adequado a essas demandas, então tivemos que montar o nosso próprio.
Descrevemos nossa abordagem em detalhes em: numref: `sec_how_to_contribute`.
Decidimos usar o GitHub para compartilhar a fonte e permitir edições,
*Notebooks* Jupyter para misturar código, equações e texto,
Sphinx como um mecanismo de renderização para gerar várias saídas,
e *Discourse* para o fórum.
Embora nosso sistema ainda não seja perfeito,
essas escolhas fornecem um bom compromisso entre as preocupações concorrentes.
Acreditamos que este seja o primeiro livro publicado
usando um *workflow* integrado.


### Aprendendo Fazendo


Muitos livros ensinam uma série de tópicos, cada um com detalhes exaustivos.
Por exemplo, o excelente livro de Chris Bishop: cite: `Bishop.2006`,
ensina cada tópico tão completamente, que chegar ao capítulo
na regressão linear requer uma quantidade não trivial de trabalho.
Embora os especialistas amem este livro precisamente por sua eficácia,
para iniciantes, essa propriedade limita sua utilidade como um texto introdutório.

Neste livro, ensinaremos a maioria dos conceitos *just in time*.
Em outras palavras, você aprenderá conceitos no exato momento
que eles são necessários para realizar algum fim prático.
Enquanto levamos algum tempo no início para ensinar
preliminares fundamentais, como álgebra linear e probabilidade,
queremos que você experimente a satisfação de treinar seu primeiro modelo
antes de se preocupar com distribuições de probabilidade mais esotéricas.

Além de alguns cadernos preliminares que fornecem um curso intensivo
no *background* matemático básico,
cada capítulo subsequente apresenta um número razoável de novos conceitos
e fornece exemplos de trabalho auto-contidos únicos --- usando *datasets* reais.
Isso representa um desafio organizacional.
Alguns modelos podem ser agrupados logicamente em um único *notebook*.
E algumas idéias podem ser melhor ensinadas executando vários modelos em sucessão.
Por outro lado, há uma grande vantagem em aderir
a uma política de *um exemplo funcional, um notebook*:
Isso torna o mais fácil possível para você
comece seus próprios projetos de pesquisa aproveitando nosso código.
Basta copiar um *notebook* e começar a modificá-lo.


Vamos intercalar o código executável com o *background* de  material, conforme necessário.
Em geral, muitas vezes erramos por fazer ferramentas
disponíveis antes de explicá-los totalmente (e vamos acompanhar por
explicando o *background* mais tarde).
Por exemplo, podemos usar *gradiente descendente estocástico*
antes de explicar completamente porque é útil ou porque funciona.
Isso ajuda a dar aos profissionais a
munição necessária para resolver problemas rapidamente,
às custas de exigir do leitor
que nos confie algumas decisões curatoriais.

Este livro vai ensinar conceitos de *deep learning* do zero.
Às vezes, queremos nos aprofundar em detalhes sobre os modelos
que normalmente ficaria oculto do usuário
pelas abstrações avançadas dos *frameworks* de *deep learning*.
Isso surge especialmente nos tutoriais básicos,
onde queremos que você entenda tudo
que acontece em uma determinada camada ou otimizador.
Nesses casos, apresentaremos frequentemente duas versões do exemplo:
onde implementamos tudo do zero,
contando apenas com a interface NumPy e diferenciação automática,
e outro exemplo mais prático,
onde escrevemos código sucinto usando APIs de alto nível de *frameworks* de *deep learning*.
Depois de ensinar a você como alguns componentes funcionam,
podemos apenas usar as APIs de alto nível em tutoriais subsequentes.


### Conteúdo e Estrutura

O livro pode ser dividido em três partes,
que são apresentados por cores diferentes em: numref: `fig_book_org`:

![Book structure](../img/book-org.svg)
:label:`fig_book_org`


* A primeira parte cobre os princípios básicos e preliminares.
: numref: `chap_introduction` oferece uma introdução ao *deep learning*.
Então, em: numref: `chap_preliminaries`,
nós o informamos rapidamente sobre os pré-requisitos exigidos
para *deep learning* prático, como armazenar e manipular dados,
e como aplicar várias operações numéricas com base em conceitos básicos
da álgebra linear, cálculo e probabilidade.
: numref: `chap_linear` e: numref:` chap_perceptrons`
cobrem os conceitos e técnicas mais básicos de aprendizagem profunda,
como regressão linear, *multilayer perceptrons* e regularização.

* Os próximos cinco capítulos enfocam as técnicas modernas de *deep learning*.
: numref: `chap_computation` descreve os vários componentes-chave dos cálculos do *deep learning* e estabelece as bases
para que possamos posteriormente implementar modelos mais complexos.
A seguir, em: numref: `chap_cnn` e: numref:` chap_modern_cnn`,
apresentamos redes neurais convolucionais (CNNs, do inglês *convolutional neural networks*), ferramentas poderosas
que formam a espinha dorsal da maioria dos sistemas modernos de visão computacional.
Posteriormente, em: numref: `chap_rnn` e: numref:` chap_modern_rnn`, apresentamos
redes neurais recorrentes (RNNs, do inglês *recurrent neural networks*), modelos que exploram
estrutura temporal ou sequencial em dados, e são comumente usados
para processamento de linguagem natural e previsão de séries temporais.
Em: numref: `chap_attention`, apresentamos uma nova classe de modelos
que empregam uma técnica chamada mecanismos de atenção,
que recentemente começaram a deslocar RNNs no processamento de linguagem natural.
Estas seções irão ajudá-lo a aprender sobre as ferramentas básicas
por trás da maioria das aplicações modernas de *deep learning*.

* A parte três discute escalabilidade, eficiência e aplicações.
Primeiro, em: numref: `chap_optimization`,
discutimos vários algoritmos de otimização comuns
usado para treinar modelos de *deep learning*.
O próximo capítulo,: numref: `chap_performance` examina vários fatores-chave
que influenciam o desempenho computacional de seu código de *deep learning*.
Em: numref: `chap_cv`,
nós ilustramos as
principais aplicações de *deep learning* em visão computacional.
Em: numref: `chap_nlp_pretrain` e: numref:` chap_nlp_app`,
mostramos como pré-treinar modelos de representação de linguagem e aplicar
para tarefas de processamento de linguagem natural.

### Códigos
:label:`sec_code`


A maioria das seções deste livro apresenta código executável devido a acreditarmos
na importância de uma experiência de aprendizagem interativa em *deep learning*.
No momento, certas intuições só podem ser desenvolvidas por tentativa e erro,
ajustando o código em pequenas formas e observando os resultados.
Idealmente, uma elegante teoria matemática pode nos dizer
precisamente como ajustar nosso código para alcançar o resultado desejado.
Infelizmente, no momento, essas teorias elegantes nos escapam.
Apesar de nossas melhores tentativas, explicações formais para várias técnicas
ainda faltam, tanto porque a matemática para caracterizar esses modelos
pode ser tão difícil e também porque uma investigação séria sobre esses tópicos
só recentemente entrou em foco.
Temos esperança de que, à medida que a teoria do *deep learning* avança,
futuras edições deste livro serão capazes de fornecer *insights*
em lugares em que a presente edição não pode.

Às vezes, para evitar repetição desnecessária, encapsulamos
as funções, classes, etc. importadas e mencionadas com frequência
neste livro no *package* `d2l`.
Para qualquer bloco, como uma função, uma classe ou vários *imports*
ser salvo no pacote, vamos marcá-lo com
`# @ save`. Oferecemos uma visão geral detalhada dessas funções e classes em: numref: `sec_d2l`.
O *package* `d2l` é leve e requer apenas
os seguintes *packages* e módulos como dependências:

```{.python .input}
#@tab all
#@save
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
Most of the code in this book is based on Apache MXNet.
MXNet is an open-source framework for deep learning
and the preferred choice of AWS (Amazon Web Services),
as well as many colleges and companies.
All of the code in this book has passed tests under the newest MXNet version.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of MXNet.
However, we plan to keep the online version up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from MXNet.
:end_tab:

:begin_tab:`pytorch`
Most of the code in this book is based on PyTorch.
PyTorch is an open-source framework for deep learning, which is extremely
popular in the research community.
All of the code in this book has passed tests under the newest PyTorch.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of PyTorch.
However, we plan to keep the online version up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from PyTorch.
:end_tab:

:begin_tab:`tensorflow`
Most of the code in this book is based on TensorFlow.
TensorFlow is an open-source framework for deep learning, which is extremely
popular in both the research community and industry.
All of the code in this book has passed tests under the newest TensorFlow.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of TensorFlow.
However, we plan to keep the online version up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from TensorFlow.
:end_tab:

```{.python .input}
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from PIL import Image
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

### Target Audience

This book is for students (undergraduate or graduate),
engineers, and researchers, who seek a solid grasp
of the practical techniques of deep learning.
Because we explain every concept from scratch,
no previous background in deep learning or machine learning is required.
Fully explaining the methods of deep learning
requires some mathematics and programming,
but we will only assume that you come in with some basics,
including (the very basics of) linear algebra, calculus, probability,
and Python programming.
Moreover, in the Appendix, we provide a refresher
on most of the mathematics covered in this book.
Most of the time, we will prioritize intuition and ideas
over mathematical rigor.
There are many terrific books which can lead the interested reader further.
For instance, Linear Analysis by Bela Bollobas :cite:`Bollobas.1999`
covers linear algebra and functional analysis in great depth.
All of Statistics :cite:`Wasserman.2013` is a terrific guide to statistics.
And if you have not used Python before,
you may want to peruse this [Python tutorial](http://learnpython.org/).


### Forum

Associated with this book, we have launched a discussion forum,
located at [discuss.d2l.ai](https://discuss.d2l.ai/).
When you have questions on any section of the book,
you can find the associated discussion page link at the end of each chapter.


## Acknowledgments

We are indebted to the hundreds of contributors for both
the English and the Chinese drafts.
They helped improve the content and offered valuable feedback.
Specifically, we thank every contributor of this English draft
for making it better for everyone.
Their GitHub IDs or names are (in no particular order):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, 315930399, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino.

We thank Amazon Web Services, especially Swami Sivasubramanian,
Raju Gulabani, Charlie Bell, and Andrew Jassy for their generous support in writing this book. Without the available time, resources, discussions with colleagues, and continuous encouragement this book would not have happened.


## Summary

* Deep learning has revolutionized pattern recognition, introducing technology that now powers a wide range of  technologies, including computer vision, natural language processing, automatic speech recognition.
* To successfully apply deep learning, you must understand how to cast a problem, the mathematics of modeling, the algorithms for fitting your models to data, and the engineering techniques to implement it all.
* This book presents a comprehensive resource, including prose, figures, mathematics, and code, all in one place.
* To answer questions related to this book, visit our forum at https://discuss.d2l.ai/.
* All notebooks are available for download on GitHub.


## Exercises

1. Register an account on the discussion forum of this book [discuss.d2l.ai](https://discuss.d2l.ai/).
1. Install Python on your computer.
1. Follow the links at the bottom of the section to the forum, where you will be able to seek out help and discuss the book and find answers to your questions by engaging the authors and broader community.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/186)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTkxMDIwMjk5MiwtMjk2MjA4NDI1LDEyOT
E5MTkzMzcsNzY2MDkzMjQyLDExODA4Mzg0MzAsNjUwMTUyODk1
LDI0MDUxOTY0NiwtMTQ0Nzk3NzYyOSwtNjY2NjIzODcwLC0xNj
M2OTI4MTA4LDE1MjkzMTIzMDMsMTQzMzgxNTk5MiwxODAyMTM1
MjY3LDE1NDU3NDI3OThdfQ==
-->
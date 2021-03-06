#  Preliminaries
:label:`chap_preliminaries`

Para iniciarmos o nosso aprendizado de *Deep Learning*,
precisaremos desenvolver algumas habilidades básicas.
Todo aprendizado de máquina está relacionado
com a extração de informações dos dados.
Portanto, começaremos aprendendo as habilidades práticas
para armazenar, manipular e pré-processar dados.

Além disso, o aprendizado de máquina normalmente requer
trabalhar com grandes conjuntos de dados, que podemos considerar como tabelas,
onde as linhas correspondem a exemplos
e as colunas correspondem aos atributos.
A álgebra linear nos dá um poderoso conjunto de técnicas
para trabalhar com dados tabulares.
Não iremos muito longe nas teoria, mas sim nos concentraremos no básico
das operações matriciais e sua implementação.

Additionally, deep learning is all about optimization.
We have a model with some parameters and
we want to find those that fit our data *the best*.
Determining which way to move each parameter at each step of an algorithm
requires a little bit of calculus, which will be briefly introduced.
Fortunately, the `autograd` package automatically computes differentiation for us,
and we will cover it next.
Além disso, o *Deep Learning* tem tudo a ver com otimização.
Temos um modelo com alguns parâmetros e
queremos encontrar aqueles que melhor se ajustam aos nossos dados.
Determinar como mover cada parâmetro em cada etapa de um algoritmo
requer um pouco de cálculo, que será brevemente apresentado.
Felizmente, o pacote `autograd` calcula automaticamente a diferenciação para nós,
e vamos cobrir isso a seguir.

Next, machine learning is concerned with making predictions:
what is the likely value of some unknown attribute,
given the information that we observe?
To reason rigorously under uncertainty
we will need to invoke the language of probability.

In the end, the official documentation provides
plenty of descriptions and examples that are beyond this book.
To conclude the chapter, we will show you how to look up documentation for
the needed information.

This book has kept the mathematical content to the minimum necessary
to get a proper understanding of deep learning.
However, it does not mean that
this book is mathematics free.
Thus, this chapter provides a rapid introduction to
basic and frequently-used mathematics to allow anyone to understand
at least *most* of the mathematical content of the book.
If you wish to understand *all* of the mathematical content,
further reviewing the [online appendix on mathematics](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html) should be sufficient.

```toc
:maxdepth: 2

ndarray
pandas
linear-algebra
calculus
autograd
probability
lookup-api
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTg3NzQ3MjczM119
-->
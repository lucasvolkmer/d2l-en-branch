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

Além disso, o *Deep Learning* tem tudo a ver com otimização.
Temos um modelo com alguns parâmetros e
queremos encontrar aqueles que melhor se ajustam aos nossos dados.
Determinar como alterar cada parâmetro em cada etapa de um algoritmo
requer um pouco de cálculo, que será brevemente apresentado.
Felizmente, o pacote `autograd` calcula automaticamente a diferenciação para nós,
e vamos cobrir isso a seguir.

Next, machine learning is concerned with making predictions:
what is the likely value of some unknown attribute,
given the information that we observe?
To reason rigorously under uncertainty
we will need to invoke the language of probability.

Em seguida, o aprendizado de máquina se preocupa em fazer previsões:
qual é o valor provável de algum atributo desconhecido,
dada a informação que observamos?
Raciocinar rigorosamente sob a incerteza
precisaremos invocar a linguagem da probabilidade.

No final, a documentação oficial fornece
muitas descrições e exemplos que vão além deste livro.
Para concluir o capítulo, mostraremos como procurar documentação para
as informações necessárias.

This book has kept the mathematical content to the minimum necessary
to get a proper understanding of deep learning.
However, it does not mean that
this book is mathematics free.
Thus, this chapter provides a rapid introduction to
basic and frequently-used mathematics to allow anyone to understand
at least *most* of the mathematical content of the book.
If you wish to understand *all* of the mathematical content,
further reviewing the [online appendix on mathematics](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html) should be sufficient.

Este livro manteve o conteúdo matemático no mínimo necessário
para obter uma compreensão adequada do aprendizado profundo.
No entanto, isso não significa que
este livro é livre de matemática.
Assim, este capítulo fornece uma introdução rápida a
matemática básica e frequentemente usada para permitir que qualquer pessoa entenda
pelo menos * a maior parte * do conteúdo matemático do livro.
Se você deseja entender * todo * o conteúdo matemático,
uma revisão adicional do [apêndice online sobre matemática] (https://d2l.ai/chapter_apencha-mathematics-for-deep-learning/index.html) deve ser suficiente.

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
eyJoaXN0b3J5IjpbMTcwNjI0MjA2M119
-->
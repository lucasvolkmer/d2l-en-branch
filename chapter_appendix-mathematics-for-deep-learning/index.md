# Apêndice: Matemática para *Deep Learning*
:label:`chap_appendix_math`

**Brent Werness** (*Amazon*), **Rachel Hu** (*Amazon*), e autores deste livro



Uma das partes maravilhosas do *deep learning* moderno é o fato de que muito dele pode ser compreendido e usado sem uma compreensão completa da matemática abaixo dele. Isso é um sinal de que o campo está amadurecendo. Assim como a maioria dos desenvolvedores de *software* não precisa mais se preocupar com a teoria das funções computáveis, os profissionais de aprendizado profundo também não devem se preocupar com os fundamentos teóricos do aprendizado de máxima *likelihood*.

Mas ainda não chegamos lá.

Na prática, às vezes você precisará entender como as escolhas arquitetônicas influenciam o fluxo de gradiente ou as suposições implícitas que você faz ao treinar com uma determinada função de perda. Você pode precisar saber o que mede a entropia mundial e como isso pode ajudá-lo a entender exatamente o que cada bit por caractere significam em seu modelo. Tudo isso requer um conhecimento matemático mais profundo.

This appendix aims to provide you the mathematical background you need to understand the core theory of modern deep learning, but it is not exhaustive.  We will begin with examining linear algebra in greater depth.  We develop a geometric understanding of all the common linear algebraic objects and operations that will enable us to visualize the effects of various transformations on our data.  A key element is the development of the basics of eigen-decompositions.

We next develop the theory of differential calculus to the point that we can fully understand why the gradient is the direction of steepest descent, and why back-propagation takes the form it does.  Integral calculus is then discussed to the degree needed to support our next topic, probability theory.

Problems encountered in practice frequently are not certain, and thus we need a language to speak about uncertain things.  We review the theory of random variables and the most commonly encountered distributions so we may discuss models probabilistically.  This provides the foundation for the naive Bayes classifier, a probabilistic classification technique.

Closely related to probability theory is the study of statistics.  While statistics is far too large a field to do justice in a short section, we will introduce fundamental concepts that all machine learning practitioners should be aware of, in particular: evaluating and comparing estimators, conducting hypothesis tests, and constructing confidence intervals.

Last, we turn to the topic of information theory, which is the mathematical study of information storage and transmission.  This provides the core language by which we may discuss quantitatively how much information a model holds on a domain of discourse.

Taken together, these form the core of the mathematical concepts needed to begin down the path towards a deep understanding of deep learning.

```toc
:maxdepth: 2

geometry-linear-algebraic-ops
eigendecomposition
single-variable-calculus
multivariable-calculus
integral-calculus
random-variables
maximum-likelihood
distributions
naive-bayes
statistics
information-theory
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5MDI1OTg2OTJdfQ==
-->
# Documentation
:begin_tab:`mxnet`
Devido a restrições na extensão deste livro, não podemos apresentar todas as funções e classes do MXNet (e você provavelmente não gostaria que o fizéssemos). A documentação da API e os tutoriais e exemplos adicionais fornecem muita documentação além do livro. Nesta seção, fornecemos algumas orientações para explorar a API MXNet.
:end_tab:

:begin_tab:`pytorch`
Due to constraints on the length of this book, we cannot possibly introduce every single PyTorch function and class (and you probably would not want us to). The API documentation and additional tutorials and examples provide plenty of documentation beyond the book. In this section we provide you with some guidance to exploring the PyTorch API.

Devido a restrições na extensão deste livro, não podemos apresentar todas as funções e classes do PyTorch (e você provavelmente não gostaria que o fizéssemos). A documentação da API e os tutoriais e exemplos adicionais fornecem muita documentação além do livro. Nesta seção, fornecemos algumas orientações para explorar a API PyTorch.
:end_tab:

:begin_tab:`tensorflow`
Due to constraints on the length of this book, we cannot possibly introduce every single TensorFlow function and class (and you probably would not want us to). The API documentation and additional tutorials and examples provide plenty of documentation beyond the book. In this section we provide you with some guidance to exploring the TensorFlow API.

Devido a restrições na extensão deste livro, não podemos apresentar todas as funções e classes do TensorFlow (e você provavelmente não gostaria que o fizéssemos). A documentação da API e os tutoriais e exemplos adicionais fornecem muita documentação além do livro. Nesta seção, fornecemos algumas orientações para explorar a API TensorFlow.
:end_tab:


## Encontrando Todas as Funções e Classes em um Módulo

In order to know which functions and classes can be called in a module, we
invoke the `dir` function. For instance, we can (**query all properties in the
module for generating random numbers**):

Para saber quais funções e classes podem ser chamadas em um módulo, nós
invoque a função `dir`. Por exemplo, podemos (**consultar todas as propriedades no
módulo para gerar números aleatórios**):

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

Generally, we can ignore functions that start and end with `__` (special objects in Python) or functions that start with a single `_`(usually internal functions). Based on the remaining function or attribute names, we might hazard a guess that this module offers various methods for generating random numbers, including sampling from the uniform distribution (`uniform`), normal distribution (`normal`), and multinomial distribution  (`multinomial`).

Geralmente, podemos ignorar funções que começam e terminam com `__` (objetos especiais em Python) ou funções que começam com um único `_` (normalmente funções internas). Com base nos nomes de funções ou atributos restantes, podemos arriscar um palpite de que este módulo oferece vários métodos para gerar números aleatórios, incluindo amostragem da distribuição uniforme (`uniforme`), distribuição normal (`normal`) e distribuição multinomial (`multinomial`).

## Buscando o Uso de Funções e Classes Específicas

For more specific instructions on how to use a given function or class, we can invoke the  `help` function. As an example, let us [**explore the usage instructions for tensors' `ones` function**].

Para obter instruções mais específicas sobre como usar uma determinada função ou classe, podemos invocar a função `help`. Como um exemplo, vamos [**explorar as instruções de uso para a função `uns` dos tensores**].

```{.python .input}
help(np.ones)
```

```{.python .input}
#@tab pytorch
help(torch.ones)
```

```{.python .input}
#@tab tensorflow
help(tf.ones)
```

From the documentation, we can see that the `ones` function creates a new tensor with the specified shape and sets all the elements to the value of 1. Whenever possible, you should (**run a quick test**) to confirm your interpretation:

A partir da documentação, podemos ver que a função `uns` cria um novo tensor com a forma especificada e define todos os elementos com o valor de 1. Sempre que possível, você deve (**executar um teste rápido**) para confirmar seu interpretação:

```{.python .input}
np.ones(4)
```

```{.python .input}
#@tab pytorch
torch.ones(4)
```

```{.python .input}
#@tab tensorflow
tf.ones(4)
```

In the Jupyter notebook, we can use `?` to display the document in another
window. For example, `list?` will create content that is almost
identical to `help(list)`, displaying it in a new browser
window. In addition, if we use two question marks, such as
`list??`, the Python code implementing the function will also be
displayed.

No bloco de notas Jupyter, podemos usar `?` Para exibir o documento em outro
janela. Por exemplo, `list?` Criará conteúdo que é quase
idêntico a `help (list)`, exibindo-o em um novo navegador
janela. Além disso, se usarmos dois pontos de interrogação, como
`list ??`, o código Python implementando a função também será
exibido.


## Sumário

* The official documentation provides plenty of descriptions and examples that are beyond this book.
* We can look up documentation for the usage of an API by calling the `dir` and `help` functions, or `?` and `??` in Jupyter notebooks.

* A documentação oficial fornece muitas descrições e exemplos que vão além deste livro.
* Podemos consultar a documentação para o uso de uma API chamando as funções `dir` e` help`, ou `?` E `??` em blocos de notas Jupyter.

## Exercícios

1. Look up the documentation for any function or class in the deep learning framework. Can you also find the documentation on the official website of the framework?

1. Procure a documentação de qualquer função ou classe na estrutura de aprendizado profundo. Você também pode encontrar a documentação no site oficial do framework?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk2ODU4MDY3Nl19
-->
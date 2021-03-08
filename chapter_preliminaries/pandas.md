# Data Preéprocessingamento de Dados
:label:`sec_pandas`

Até agora, introduzimos uma variedade de técnicas para manipular dados que já estão armazenados em tensores.
Para aplicar o *Deep Learning* na solução de problemas do mundo real,
frequentemente começamos com o pré-processamento de dados brutos, em vez daqueles dados bem preparados no formato tensor.
Entre as ferramentas analíticas de dados populares em Python, o pacote `pandas` é comumente usado.
Como muitos outros pacotes de extensão no vasto ecossistema do Python,
`pandas` podem trabalhar em conjunto com tensores.
Então, vamos percorrer brevemente as etapas de pré-processamento de dados brutos com `pandas`
e convertendo-os no formato tensor.
Abordaremos mais técnicas de pré-processamento de dados em capítulos posteriores.

## Lendo o  *Dataset*

Como um exemplo,
começamos (**criando um conjunto de dados artificial que é armazenado em um
arquivo csv (valores separados por vírgula)**)
`../ data / house_tiny.csv`. Dados armazenados em outro
formatos podem ser processados de maneiras semelhantes.

Abaixo, escrevemos o conjunto de dados linha por linha em um arquivo csv.

```{.python .input}
#@tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

Para [**carregar o conjunto de dados bruto do arquivo csv criado**],
importamos o pacote `pandas` e chamamos a função` read_csv`.
Este conjunto de dados tem quatro linhas e três colunas, onde cada linha descreve o número de quartos ("NumRooms"), o tipo de beco ("Alley") e o preço ("Price") de uma casa.

```{.python .input}
#@tab all
# Se o pandas ainda não estiver instalado descomente a linha abaixo:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Lidando com Dados Faltantes

Observe que as entradas "NaN" têm valores ausentes.
Para lidar com dados perdidos, os métodos típicos incluem *imputação* e *exclusão*,
onde a imputação substitui os valores ausentes por outros substituídos,
enquanto a exclusão ignora os valores ausentes. Aqui, consideraremos a imputação.

Por indexação baseada em localização de inteiros (`iloc`), dividimos os `dados` em `entradas` e `saídas`,
onde o primeiro leva as duas primeiras colunas, enquanto o último mantém apenas a última coluna.
Para valores numéricos em `entradas` que estão faltando,
nós [**substituímos as entradas "NaN" pelo valor médio da mesma coluna.**]

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

[**For categorical or discrete values in `inputs`, we consider "NaN" as a category.**]
Since the "Alley" column only takes two types of categorical values "Pave" and "NaN",
`pandas` can automatically convert this column to two columns "Alley_Pave" and "Alley_nan".
A row whose alley type is "Pave" will set values of "Alley_Pave" and "Alley_nan" to 1 and 0.
A row with a missing alley type will set their values to 0 and 1.

[**Para valores categóricos ou discretos em `entradas`, consideramos "NaN" como uma categoria.**]
Como a coluna "Alley" aceita apenas dois tipos de valores categóricos "Pave" e "NaN",
O `pandas` pode converter automaticamente esta coluna em duas colunas "Alley_Pave" e "Alley_nan".
Uma linha cujo tipo de beco é "Pave" definirá os valores de "Alley_Pave" e "Alley_nan" como 1 e 0.
Uma linha com um tipo de beco ausente definirá seus valores para 0 e 1

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## Conversion to the Tensor Format

Now that [**all the entries in `inputs` and `outputs` are numerical, they can be converted to the tensor format.**]
Once data are in this format, they can be further manipulated with those tensor functionalities that we have introduced in :numref:`sec_ndarray`.

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

```{.python .input}
#@tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
X, y
```

## Summary

* Like many other extension packages in the vast ecosystem of Python, `pandas` can work together with tensors.
* Imputation and deletion can be used to handle missing data.


## Exercises

Create a raw dataset with more rows and columns.

1. Delete the column with the most missing values.
2. Convert the preprocessed dataset to the tensor format.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTkzNjY0NzUwMSwyMTE0NzM2NjkxLC04MT
U5NDc1NiwtMTQ2NTE2MDYxNiwtNTM1NTU4MjUwXX0=
-->
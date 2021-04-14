# Machine Translation and the Dataset
:label:`sec_machine_translation`

Usamos RNNs para projetar modelos de linguagem,
que são essenciais para o processamento de linguagem natural.
Outro benchmark emblemático é a *tradução automática*,
um domínio de problema central para modelos de *transdução de sequência*
que transformam sequências de entrada em sequências de saída.
Desempenhando um papel crucial em várias aplicações modernas de IA,
modelos de transdução de sequência formarão o foco do restante deste capítulo
e :numref:`chap_attention`.
Para este fim,
esta seção apresenta o problema da tradução automática
e seu conjunto de dados que será usado posteriormente.

*Tradução automática* refere-se ao
tradução automática de uma sequência
de um idioma para outro.
Na verdade, este campo
pode remontar a 1940
logo depois que os computadores digitais foram inventados,
especialmente considerando o uso de computadores
para decifrar códigos de linguagem na Segunda Guerra Mundial.
Por décadas,
abordagens estatísticas
tinha sido dominante neste campo :cite:`Brown.Cocke.Della-Pietra.ea.1988, Brown.Cocke.Della-Pietra.ea.1990`
antes da ascensão
de
aprendizagem ponta a ponta usando
redes neurais.
O último
é frequentemente chamado
*tradução automática neural*
para se distinguir de
*tradução automática de estatística*
que envolve análise estatística
em componentes como
o modelo de tradução e o modelo de linguagem.

Enfatizando o aprendizado de ponta a ponta,
este livro se concentrará em métodos de tradução automática neural.
Diferente do nosso problema de modelo de linguagem
in :numref:`sec_language_model`
cujo corpus está em um único idioma,
conjuntos de dados de tradução automática
são compostos por pares de sequências de texto
que estão em
o idioma de origem e o idioma de destino, respectivamente.
Desse modo,
em vez de reutilizar a rotina de pré-processamento
para modelagem de linguagem,
precisamos de uma maneira diferente de pré-processar
conjuntos de dados de tradução automática.
Na sequência,
nós mostramos como
carregar os dados pré-processados
em minibatches para treinamento.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

## Download e Pré-processamento do Conjunto de Dados

Começar com,
baixamos um conjunto de dados inglês-francês
que consiste em [pares de frases bilíngues do Projeto Tatoeba](http://www.manythings.org/anki/).
Cada linha no conjunto de dados
é um par delimitado por tabulação
de uma sequência de texto em inglês
e a sequência de texto traduzida em francês.
Observe que cada sequência de texto
pode ser apenas uma frase ou um parágrafo de várias frases.
Neste problema de tradução automática
onde o inglês é traduzido para o francês,
Inglês é o *idioma de origem*
e o francês é o *idioma de destino*.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """Load the English-French dataset."""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```

Depois de baixar o conjunto de dados,
continuamos com várias etapas de pré-processamento
para os dados de texto brutos.
Por exemplo,
substituímos o espaço ininterrupto por espaço,
converter letras maiúsculas em minúsculas,
e insira espaço entre palavras e sinais de pontuação.

```{.python .input}
#@tab all
#@save
def preprocess_nmt(text):
    """Preprocess the English-French dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

## Tokenização

Diferente da tokenização em nível de personagem
in:numref:`sec_language_model`,
para tradução automática
preferimos tokenização em nível de palavra aqui
(modelos de última geração podem usar técnicas de tokenização mais avançadas).
A seguinte função `tokenize_nmt`
tokeniza os primeiros pares de sequência de texto `num_examples`,
Onde
cada token é uma palavra ou um sinal de pontuação.
Esta função retorna
duas listas de listas de tokens: `source` e` target`.
Especificamente,
`source [i]` é uma lista de tokens do
$i^\mathrm{th}$ sequência de texto no idioma de origem (inglês aqui) e `target [i]` é a do idioma de destino (francês aqui).

```{.python .input}
#@tab all
#@save
def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset."""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
```
Deixe-nos representar graficamente o histograma do número de tokens por sequência de texto.
Neste conjunto de dados inglês-francês simples,
a maioria das sequências de texto tem menos de 20 tokens.

```{.python .input}
#@tab all
d2l.set_figsize()
_, _, patches = d2l.plt.hist(
    [[len(l) for l in source], [len(l) for l in target]],
    label=['source', 'target'])
for patch in patches[1].patches:
    patch.set_hatch('/')
d2l.plt.legend(loc='upper right');
```

## Vocabulário

Uma vez que o conjunto de dados da tradução automática
consiste em pares de línguas,
podemos construir dois vocabulários para
tanto o idioma de origem quanto
o idioma de destino separadamente.
Com tokenização em nível de palavra,
o tamanho do vocabulário será significativamente maior
do que usando tokenização em nível de caractere.
Para aliviar isso,
aqui tratamos tokens infrequentes
que aparecem menos de 2 vezes
como o mesmo token desconhecido ("&lt;bos&gt;").
Além disso,
especificamos tokens especiais adicionais
como para sequências de preenchimento ("&lt;bos&gt;") com o mesmo comprimento em minibatches,
e para marcar o início ("&lt;bos&gt;") ou o fim ("&lt;bos&gt;") das sequências.
Esses tokens especiais são comumente usados em
tarefas de processamento de linguagem natural.

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## Loading the Dataset
:label:`subsec_mt_data_loading`

Recall that in language modeling
each sequence example,
either a segment of one sentence
or a span over multiple sentences,
has a fixed length.
This was specified by the `num_steps`
(number of time steps or tokens) argument in :numref:`sec_language_model`.
In machine translation, each example is
a pair of source and target text sequences,
where each text sequence may have different lengths.

For computational efficiency,
we can still process a minibatch of text sequences
at one time by *truncation* and *padding*.
Suppose that every sequence in the same minibatch
should have the same length `num_steps`.
If a text sequence has fewer than `num_steps` tokens,
we will keep appending the special "&lt;pad&gt;" token
to its end until its length reaches `num_steps`.
Otherwise,
we will truncate the text sequence
by only taking its first `num_steps` tokens
and discarding the remaining.
In this way,
every text sequence
will have the same length
to be loaded in minibatches of the same shape.

The following `truncate_pad` function
truncates or pads text sequences as described before.

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

Now we define a function to transform
text sequences into minibatches for training.
We append the special “&lt;eos&gt;” token
to the end of every sequence to indicate the
end of the sequence.
When a model is predicting
by
generating a sequence token after token,
the generation
of the “&lt;eos&gt;” token
can suggest that
the output sequence is complete.
Besides,
we also record the length
of each text sequence excluding the padding tokens.
This information will be needed by
some models that
we will cover later.

```{.python .input}
#@tab all
#@save
def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len
```

## Putting All Things Together

Finally, we define the `load_data_nmt` function
to return the data iterator, together with
the vocabularies for both the source language and the target language.

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
```

Let us read the first minibatch from the English-French dataset.

```{.python .input}
#@tab all
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', d2l.astype(X, d2l.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', d2l.astype(Y, d2l.int32))
    print('valid lengths for Y:', Y_valid_len)
    break
```

## Summary

* Machine translation refers to the automatic translation of a sequence from one language to another.
* Using word-level tokenization, the vocabulary size will be significantly larger than that using character-level tokenization. To alleviate this, we can treat infrequent tokens as the same unknown token.
* We can truncate and pad text sequences so that all of them will have the same length to be loaded in minibatches.


## Exercises

1. Try different values of the `num_examples` argument in the `load_data_nmt` function. How does this affect the vocabulary sizes of the source language and the target language?
1. Text in some languages such as Chinese and Japanese does not have word boundary indicators (e.g., space). Is word-level tokenization still a good idea for such cases? Why or why not?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwMzc0OTY5NjNdfQ==
-->
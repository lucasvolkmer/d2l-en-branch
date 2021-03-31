# Preprocessamento de Texto
:label:`sec_text_preprocessing`


Nós revisamos e avaliamos
ferramentas estatísticas
e desafios de previsão
para dados de sequência.
Esses dados podem assumir várias formas.
Especificamente,
como vamos nos concentrar em
em muitos capítulos do livro,
text é um dos exemplos mais populares de dados de sequência.
Por exemplo,
um artigo pode ser visto simplesmente como uma sequência de palavras ou mesmo uma sequência de caracteres.
Para facilitar nossos experimentos futuros
com dados de sequência,
vamos dedicar esta seção
para explicar as etapas comuns de pré-processamento para texto.
Normalmente, essas etapas são:

1. Carregar o texto como strings na memória.
1. Dividir as strings em tokens (por exemplo, palavras e caracteres).
1. Construir uma tabela de vocabulário para mapear os tokens divididos em índices numéricos.
1. Converter o texto em sequências de índices numéricos para que possam ser facilmente manipulados por modelos.

```{.python .input}
import collections
from d2l import mxnet as d2l
import re
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import re
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

## Lendo o Dataset

Para começar, carregamos o texto de H. G. Wells '[*The Time Machine*] (http://www.gutenberg.org/ebooks/35).
Este é um corpus bastante pequeno de pouco mais de 30000 palavras, mas para o propósito do que queremos ilustrar, está tudo bem.
Coleções de documentos mais realistas contêm muitos bilhões de palavras.
A função a seguir lê o conjunto de dados em uma lista de linhas de texto, onde cada linha é uma *string*.
Para simplificar, aqui ignoramos a pontuação e a capitalização.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])
```

## Tokenização

The following `tokenize` function
takes a list (`lines`) as the input,
where each list is a text sequence (e.g., a text line).
Each text sequence is split into a list of tokens.
A *token* is the basic unit in text.
In the end,
a list of token lists are returned,
where each token is a string.

```{.python .input}
#@tab all
def tokenize(lines, token='word'):  #@save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

## Vocabulary

The string type of the token is inconvenient to be used by models, which take numerical inputs.
Now let us build a dictionary, often called *vocabulary* as well, to map string tokens into numerical indices starting from 0.
To do so, we first count the unique tokens in all the documents from the training set,
namely a *corpus*,
and then assign a numerical index to each unique token according to its frequency.
Rarely appeared tokens are often removed to reduce the complexity.
Any token that does not exist in the corpus or has been removed is mapped into a special unknown token “&lt;unk&gt;”.
We optionally add a list of reserved tokens, such as
“&lt;pad&gt;” for padding,
“&lt;bos&gt;” to present the beginning for a sequence, and “&lt;eos&gt;” for the end of a sequence.

```{.python .input}
#@tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = [] 
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The index for the unknown token is 0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(tokens):  #@save
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

We construct a vocabulary using the time machine dataset as the corpus. 
Then we print the first few frequent tokens with their indices.

```{.python .input}
#@tab all
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

Now we can convert each text line into a list of numerical indices.

```{.python .input}
#@tab all
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

## Putting All Things Together

Using the above functions, we package everything into the `load_corpus_time_machine` function, which returns `corpus`, a list of token indices, and `vocab`, the vocabulary of the time machine corpus.
The modifications we did here are:
i) we tokenize text into characters, not words, to simplify the training in later sections;
ii) `corpus` is a single list, not a list of token lists, since each text line in the time machine dataset is not necessarily a sentence or a paragraph.

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## Summary

* Text is an important form of sequence data.
* To preprocess text, we usually split text into tokens, build a vocabulary to map token strings into numerical indices, and convert text data into token indices for  models to manipulate.


## Exercises

1. Tokenization is a key preprocessing step. It varies for different languages. Try to find another three commonly used methods to tokenize text.
1. In the experiment of this section, tokenize text into words and vary the `min_freq` arguments of the `Vocab` instance. How does this affect the vocabulary size? 

[Discussions](https://discuss.d2l.ai/t/115)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjk3NDI1Mjg5LC0xMTU3MDE3Mzk0XX0=
-->
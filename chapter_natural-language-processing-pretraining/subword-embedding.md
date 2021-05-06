# Incorporação de subpalavra
:label:`sec_fasttext`

As palavras em inglês geralmente têm estruturas internas e métodos de formação. Por exemplo, podemos deduzir a relação entre "cachorro", "cachorros" e "dogcatcher" por sua grafia. Todas essas palavras têm a mesma raiz, "cachorro", mas usam sufixos diferentes para mudar o significado da palavra. Além disso, essa associação pode ser estendida a outras palavras. Por exemplo, a relação entre "cachorro" e "cachorros" é exatamente como a relação entre "gato" e "gatos". A relação entre "menino" e "namorado" é igual à relação entre "menina" e "namorada". Essa característica não é exclusiva do inglês. Em francês e espanhol, muitos verbos podem ter mais de 40 formas diferentes, dependendo do contexto. Em finlandês, um substantivo pode ter mais de 15 formas. Na verdade, a morfologia, que é um importante ramo da linguística, estuda a estrutura interna e a formação das palavras.


## fastText

No word2vec, não usamos informações morfológicas diretamente. Em ambos os
modelo skip-gram e modelo de saco de palavras contínuo, usamos diferentes vetores para
representam palavras com diferentes formas. Por exemplo, "cachorro" e "cachorros" são
representado por dois vetores diferentes, enquanto a relação entre esses dois
vetores não é representado diretamente no modelo. Em vista disso, fastText :cite:`Bojanowski.Grave.Joulin.ea.2017`
propõe o método de incorporação de subpalavra, tentando assim introduzir
informação morfológica no modelo skip-gram em word2vec.

Em fastText, cada palavra central é representada como uma coleção de subpalavras. A seguir, usamos a palavra "onde" como exemplo para entender como as subpalavras são formadas. Primeiro, adicionamos os caracteres especiais “&lt;” e “&gt;” no início e no final da palavra para distinguir as subpalavras usadas como prefixos e sufixos. Em seguida, tratamos a palavra como uma sequência de caracteres para extrair os $n$-gramas. Por exemplo, quando $n=3$, podemos obter todas as subpalavras com um comprimento de $3$:

$$\textrm{"<wh"}, \ \textrm{"whe"}, \ \textrm{"her"}, \ \textrm{"ere"}, \ \textrm{"re>"},$$

e a subpalavra especial $\textrm{"<where>"}$.

Em fastText, para uma palavra $w$, registramos a união de todas as suas subpalavras com comprimento de $3$ a $6$ e as subpalavras especiais como $\mathcal{G}_w$. Assim, o dicionário é a união da coleção de subpalavras de todas as palavras. Suponha que o vetor da subpalavra $g$ no dicionário seja $\mathbf{z}_g$. Então, o vetor de palavra central $\mathbf{u}_w$ para a palavra $w$ no modelo de grama de salto pode ser expresso como

$$\mathbf{u}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

O resto do processo fastText é consistente com o modelo skip-gram, portanto, não é repetido aqui. Como podemos ver, em comparação com o modelo skip-gram, o dicionário em fastText é maior, resultando em mais parâmetros do modelo. Além disso, o vetor de uma palavra requer a soma de todos os vetores de subpalavra, o que resulta em maior complexidade de computação. No entanto, podemos obter vetores melhores para palavras complexas mais incomuns, mesmo palavras que não existem no dicionário, olhando para outras palavras com estruturas semelhantes.


## Codificação de par de bytes
:label:`subsec_Byte_Pair_Encoding`

In fastText, all the extracted subwords have to be of the specified lengths, such as $3$ to $6$, thus the vocabulary size cannot be predefined.
To allow for variable-length subwords in a fixed-size vocabulary,
we can apply a compression algorithm
called *byte pair encoding* (BPE) to extract subwords :cite:`Sennrich.Haddow.Birch.2015`.

Byte pair encoding performs a statistical analysis of the training dataset to discover common symbols within a word,
such as consecutive characters of arbitrary length.
Starting from symbols of length $1$,
byte pair encoding iteratively merges the most frequent pair of consecutive symbols to produce new longer symbols.
Note that for efficiency, pairs crossing word boundaries are not considered.
In the end, we can use such symbols as subwords to segment words.
Byte pair encoding and its variants has been used for input representations in popular natural language processing pretraining models such as GPT-2 :cite:`Radford.Wu.Child.ea.2019` and RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`.
In the following, we will illustrate how byte pair encoding works.

Em fastText, todas as subpalavras extraídas devem ter os comprimentos especificados, como $ 3 $ a $ 6 $, portanto, o tamanho do vocabulário não pode ser predefinido.
Para permitir subpalavras de comprimento variável em um vocabulário de tamanho fixo,
podemos aplicar um algoritmo de compressão
chamado de *codificação de par de bytes* (BPE) para extrair subpalavras: cite: `Sennrich.Haddow.Birch.2015`.

A codificação de pares de bytes realiza uma análise estatística do conjunto de dados de treinamento para descobrir símbolos comuns em uma palavra,
como caracteres consecutivos de comprimento arbitrário.
Começando com símbolos de comprimento $ 1 $,
a codificação de pares de bytes mescla iterativamente o par mais frequente de símbolos consecutivos para produzir novos símbolos mais longos.
Observe que, para eficiência, os pares que cruzam os limites das palavras não são considerados.
No final, podemos usar esses símbolos como subpalavras para segmentar palavras.
A codificação de pares de bytes e suas variantes foram usadas para representações de entrada em modelos populares de pré-treinamento de processamento de linguagem natural, como GPT-2: cite: `Radford.Wu.Child.ea.2019` e RoBERTa: cite:` Liu.Ott.Goyal. ea.2019`.
A seguir, ilustraremos como funciona a codificação de pares de bytes.

First, we initialize the vocabulary of symbols as all the English lowercase characters, a special end-of-word symbol `'_'`, and a special unknown symbol `'[UNK]'`.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

Since we do not consider symbol pairs that cross boundaries of words,
we only need a dictionary `raw_token_freqs` that maps words to their frequencies (number of occurrences)
in a dataset.
Note that the special symbol `'_'` is appended to each word so that
we can easily recover a word sequence (e.g., "a taller man")
from a sequence of output symbols ( e.g., "a_ tall er_ man").
Since we start the merging process from a vocabulary of only single characters and special symbols, space is inserted between every pair of consecutive characters within each word (keys of the dictionary `token_freqs`).
In other words, space is the delimiter between symbols within a word.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

We define the following `get_max_freq_pair` function that 
returns the most frequent pair of consecutive symbols within a word,
where words come from keys of the input dictionary `token_freqs`.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

As a greedy approach based on frequency of consecutive symbols,
byte pair encoding will use the following `merge_symbols` function to merge the most frequent pair of consecutive symbols to produce new symbols.

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

Now we iteratively perform the byte pair encoding algorithm over the keys of the dictionary `token_freqs`. In the first iteration, the most frequent pair of consecutive symbols are `'t'` and `'a'`, thus byte pair encoding merges them to produce a new symbol `'ta'`. In the second iteration, byte pair encoding continues to merge `'ta'` and `'l'` to result in another new symbol `'tal'`.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

After 10 iterations of byte pair encoding, we can see that list `symbols` now contains 10 more symbols that are iteratively merged from other symbols.

```{.python .input}
#@tab all
print(symbols)
```

For the same dataset specified in the keys of the dictionary `raw_token_freqs`,
each word in the dataset is now segmented by subwords "fast_", "fast", "er_", "tall_", and "tall"
as a result of the byte pair encoding algorithm.
For instance, words "faster_" and "taller_" are segmented as "fast er_" and "tall er_", respectively.

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

Note that the result of byte pair encoding depends on the dataset being used.
We can also use the subwords learned from one dataset
to segment words of another dataset.
As a greedy approach, the following `segment_BPE` function tries to break words into the longest possible subwords from the input argument `symbols`.

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

In the following, we use the subwords in list `symbols`, which is learned from the aforementioned dataset,
to segment `tokens` that represent another dataset.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## Summary

* FastText proposes a subword embedding method. Based on the skip-gram model in word2vec, it represents the central word vector as the sum of the subword vectors of the word.
* Subword embedding utilizes the principles of morphology, which usually improves the quality of representations of uncommon words.
* Byte pair encoding performs a statistical analysis of the training dataset to discover common symbols within a word. As a greedy approach, byte pair encoding iteratively merges the most frequent pair of consecutive symbols.


## Exercises

1. When there are too many subwords (for example, 6 words in English result in about $3\times 10^8$ combinations), what problems arise? Can you think of any methods to solve them? Hint: Refer to the end of section 3.2 of the fastText paper :cite:`Bojanowski.Grave.Joulin.ea.2017`.
1. How can you design a subword embedding model based on the continuous bag-of-words model?
1. To get a vocabulary of size $m$, how many merging operations are needed when the initial symbol vocabulary size is $n$?
1. How can we extend the idea of byte pair encoding to extract phrases?



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/386)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwODE0NTcwNjRdfQ==
-->
# Ajuste Fino de BERT para Aplicações de Nível de Sequência e de Token
:label:`sec_finetuning-bert`

Nas seções anteriores deste capítulo, projetamos diferentes modelos para aplicações de processamento de linguagem natural, como os baseados em RNNs, CNNs, atenção e MLPs.
Esses modelos são úteis quando há restrição de espaço ou tempo,
no entanto, elaborar um modelo específico para cada tarefa de processamento de linguagem natural é praticamente inviável.
Em :numref:`sec_bert`, introduzimos um modelo de pré-treinamento, BERT, que requer mudanças mínimas de arquitetura para uma ampla gama de tarefas de processamento de linguagem natural.
Por um lado, na altura da sua proposta, o BERT melhorou o estado da arte em várias tarefas de processamento de linguagem natural.
Por outro lado, conforme observado em :numref:`sec_bert-pretraining`, as duas versões do modelo BERT original vêm com 110 milhões e 340 milhões de parâmetros.
Assim, quando há recursos computacionais suficientes, podemos considerar o ajuste fino do BERT para aplicativos de processamento de linguagem natural *downstream*.

A seguir, generalizamos um subconjunto de aplicações de processamento de linguagem natural como nível de sequência e nível de *token*.
No nível da sequência, apresentamos como transformar a representação BERT da entrada de texto no rótulo de saída em classificação de texto único e classificação ou regressão de par de texto.
No nível do *token*, apresentaremos brevemente novos aplicativos, como marcação de texto e resposta a perguntas, e esclareceremos como o BERT pode representar suas entradas e ser transformado em rótulos de saída.
Durante o ajuste fino, as "mudanças mínimas de arquitetura" exigidas pelo BERT em diferentes aplicativos são as camadas extras totalmente conectadas.
Durante o aprendizado supervisionado de uma aplicação *downstream*, os parâmetros das camadas extras são aprendidos do zero, enquanto todos os parâmetros no modelo BERT pré-treinado são ajustados.


## Single Text Classification

*Single text classification* takes a single text sequence as the input and outputs its classification result.
Besides sentiment analysis that we have studied in this chapter,
the Corpus of Linguistic Acceptability (CoLA)
is also a dataset for single text classification,
judging whether a given sentence is grammatically acceptable or not :cite:`Warstadt.Singh.Bowman.2019`.
For instance, "I should study." is acceptable but "I should studying." is not.

![Fine-tuning BERT for single text classification applications, such as sentiment analysis and testing linguistic acceptability. Suppose that the input single text has six tokens.](../img/bert-one-seq.svg)
:label:`fig_bert-one-seq`

:numref:`sec_bert` describes the input representation of BERT.
The BERT input sequence unambiguously represents both single text and text pairs,
where the special classification token 
“&lt;cls&gt;” is used for sequence classification and 
the special classification token 
“&lt;sep&gt;” marks the end of single text or separates a pair of text.
As shown in :numref:`fig_bert-one-seq`,
in single text classification applications,
the BERT representation of the special classification token 
“&lt;cls&gt;” encodes the information of the entire input text sequence.
As the representation of the input single text,
it will be fed into a small MLP consisting of fully-connected (dense) layers
to output the distribution of all the discrete label values.


## Text Pair Classification or Regression

We have also examined natural language inference in this chapter.
It belongs to *text pair classification*,
a type of application classifying a pair of text.

Taking a pair of text as the input but outputting a continuous value,
*semantic textual similarity* is a popular *text pair regression* task.
This task measures semantic similarity of sentences.
For instance, in the Semantic Textual Similarity Benchmark dataset,
the similarity score of a pair of sentences
is an ordinal scale ranging from 0 (no meaning overlap) to 5 (meaning equivalence) :cite:`Cer.Diab.Agirre.ea.2017`.
The goal is to predict these scores.
Examples from the Semantic Textual Similarity Benchmark dataset include (sentence 1, sentence 2, similarity score):

* "A plane is taking off.", "An air plane is taking off.", 5.000;
* "A woman is eating something.", "A woman is eating meat.", 3.000;
* "A woman is dancing.", "A man is talking.", 0.000.


![Fine-tuning BERT for text pair classification or regression applications, such as natural language inference and semantic textual similarity. Suppose that the input text pair has two and three tokens.](../img/bert-two-seqs.svg)
:label:`fig_bert-two-seqs`

Comparing with single text classification in :numref:`fig_bert-one-seq`,
fine-tuning BERT for text pair classification in :numref:`fig_bert-two-seqs` 
is different in the input representation.
For text pair regression tasks such as semantic textual similarity,
trivial changes can be applied such as outputting a continuous label value
and using the mean squared loss: they are common for regression.


## Text Tagging

Now let us consider token-level tasks, such as *text tagging*,
where each token is assigned a label.
Among text tagging tasks,
*part-of-speech tagging* assigns each word a part-of-speech tag (e.g., adjective and determiner)
according to the role of the word in the sentence.
For example,
according to the Penn Treebank II tag set,
the sentence "John Smith 's car is new"
should be tagged as
"NNP (noun, proper singular) NNP POS (possessive ending) NN (noun, singular or mass) VB (verb, base form) JJ (adjective)".

![Fine-tuning BERT for text tagging applications, such as part-of-speech tagging. Suppose that the input single text has six tokens.](../img/bert-tagging.svg)
:label:`fig_bert-tagging`

Fine-tuning BERT for text tagging applications
is illustrated in :numref:`fig_bert-tagging`.
Comparing with :numref:`fig_bert-one-seq`,
the only distinction lies in that
in text tagging, the BERT representation of *every token* of the input text
is fed into the same extra fully-connected layers to output the label of the token,
such as a part-of-speech tag.



## Question Answering

As another token-level application,
*question answering* reflects capabilities of reading comprehension.
For example,
the Stanford Question Answering Dataset (SQuAD v1.1)
consists of reading passages and questions,
where the answer to every question
is just a segment of text (text span) from the passage that the question is about :cite:`Rajpurkar.Zhang.Lopyrev.ea.2016`.
To explain,
consider a passage
"Some experts report that a mask's efficacy is inconclusive. However, mask makers insist that their products, such as N95 respirator masks, can guard against the virus."
and a question "Who say that N95 respirator masks can guard against the virus?".
The answer should be the text span "mask makers" in the passage.
Thus, the goal in SQuAD v1.1 is to predict the start and end of the text span in the passage given a pair of question and passage.

![Fine-tuning BERT for question answering. Suppose that the input text pair has two and three tokens.](../img/bert-qa.svg)
:label:`fig_bert-qa`

To fine-tune BERT for question answering,
the question and passage are packed as
the first and second text sequence, respectively,
in the input of BERT.
To predict the position of the start of the text span,
the same additional fully-connected layer will transform
the BERT representation of any token from the passage of position $i$
into a scalar score $s_i$.
Such scores of all the passage tokens
are further transformed by the softmax operation
into a probability distribution,
so that each token position $i$ in the passage is assigned
a probability $p_i$ of being the start of the text span.
Predicting the end of the text span
is the same as above, except that
parameters in its additional fully-connected layer
are independent from those for predicting the start.
When predicting the end,
any passage token of position $i$
is transformed by the same fully-connected layer
into a scalar score $e_i$.
:numref:`fig_bert-qa`
depicts fine-tuning BERT for question answering.

For question answering,
the supervised learning's training objective is as straightforward as
maximizing the log-likelihoods of the ground-truth start and end positions.
When predicting the span,
we can compute the score $s_i + e_j$ for a valid span
from position $i$ to position $j$ ($i \leq j$),
and output the span with the highest score.


## Summary

* BERT requires minimal architecture changes (extra fully-connected layers) for sequence-level and token-level natural language processing applications, such as single text classification (e.g., sentiment analysis and testing linguistic acceptability), text pair classification or regression (e.g., natural language inference and semantic textual similarity), text tagging (e.g., part-of-speech tagging), and question answering.
* During supervised learning of a downstream application, parameters of the extra layers are learned from scratch while all the parameters in the pretrained BERT model are fine-tuned.



## Exercises

1. Let us design a search engine algorithm for news articles. When the system receives an query (e.g., "oil industry during the coronavirus outbreak"), it should return a ranked list of news articles that are most relevant to the query. Suppose that we have a huge pool of news articles and a large number of queries. To simplify the problem, suppose that the most relevant article has been labeled for each query. How can we apply negative sampling (see :numref:`subsec_negative-sampling`) and BERT in the algorithm design?
1. How can we leverage BERT in training language models?
1. Can we leverage BERT in machine translation?

[Discussions](https://discuss.d2l.ai/t/396)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgzMzIwNzM4LC0zNzU5MTM4NjEsLTYzOD
IxOTczMV19
-->
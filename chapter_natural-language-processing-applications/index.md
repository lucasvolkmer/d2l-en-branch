# Processamento de Linguagem Natural: Aplicações
:label:`chap_nlp_app`


Vimos como representar tokens de texto e treinar suas representações em :numref:`chap_nlp_pretrain`.
Essas representações de texto pré-treinadas podem ser fornecidas a vários modelos para diferentes tarefas de processamento de linguagem natural *downstream*.

Este livro não pretende cobrir as aplicações de processamento de linguagem natural de uma maneira abrangente.
Nosso foco é *como aplicar a aprendizagem de representação (profunda) de idiomas para resolver problemas de processamento de linguagem natural*.
No entanto, já discutimos várias aplicações de processamento de linguagem natural sem pré-treinamento nos capítulos anteriores,
apenas para explicar arquiteturas de aprendizado profundo.
Por exemplo, em :numref:`chap_rnn`,
contamos com RNNs para projetar modelos de linguagem para gerar textos semelhantes a novelas.
Em :numref:`chap_modern_rnn` e :numref:`chap_attention`,
também projetamos modelos baseados em RNNs e mecanismos de atenção
para tradução automática.
Dadas as representações de texto pré-treinadas,
neste capítulo, consideraremos mais duas tarefas de processamento de linguagem natural *downstream*:
análise de sentimento e inferência de linguagem natural.
Estes são aplicativos de processamento de linguagem natural populares e representativos:
o primeiro analisa um único texto e o último analisa as relações de pares de texto.

![As representações de texto pré-treinadas podem ser alimentadas para várias arquiteturas de *deep learning*  para diferentes aplicações de processamento de linguagem natural *downstream*. Este capítulo enfoca como projetar modelos para diferentes aplicações de processamento de linguagem natural *downstream*.](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`

As depicted in :numref:`fig_nlp-map-app`,
this chapter focuses on describing the basic ideas of designing natural language processing models using different types of deep learning architectures, such as MLPs, CNNs, RNNs, and attention.
Though it is possible to combine any pretrained text representations with any architecture for either downstream natural language processing task in :numref:`fig_nlp-map-app`,
we select a few representative combinations.
Specifically, we will explore popular architectures based on RNNs and CNNs for sentiment analysis.
For natural language inference, we choose attention and MLPs to demonstrate how to analyze text pairs.
In the end, we introduce how to fine-tune a pretrained BERT model
for a wide range of natural language processing applications,
such as on a sequence level (single text classification and text pair classification)
and a token level (text tagging and question answering).
As a concrete empirical case,
we will fine-tune BERT for natural language processing.

As we have introduced in :numref:`sec_bert`,
BERT requires minimal architecture changes
for a wide range of natural language processing applications.
However, this benefit comes at the cost of fine-tuning
a huge number of BERT parameters for the downstream applications.
When space or time is limited,
those crafted models based on MLPs, CNNs, RNNs, and attention
are more feasible.
In the following, we start by the sentiment analysis application
and illustrate the model design based on RNNs and CNNs, respectively.

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbNDc5OTgzNzkzLDY5MDY3MjA3MF19
-->
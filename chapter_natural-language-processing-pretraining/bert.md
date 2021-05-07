# Representações de codificador bidirecional de transformadores (BERT)
:label:`sec_bert`

Introduzimos vários modelos de incorporação de palavras para a compreensão da linguagem natural.
Após o pré-treinamento, a saída pode ser pensada como uma matriz
onde cada linha é um vetor que representa uma palavra de um vocabulário predefinido.
Na verdade, esses modelos de incorporação de palavras são todos *independentes do contexto*.
Vamos começar ilustrando essa propriedade.

## De Independente do Contexto para Sensível ao Contexto

Lembre-se dos experimentos em :numref:`sec_word2vec_pretraining` e :numref:`sec_synonyms`.
Por exemplo, word2vec e GloVe atribuem o mesmo vetor pré-treinado à mesma palavra, independentemente do contexto da palavra (se houver).
Formalmente, uma representação independente de contexto de qualquer token $x$
é uma função $f(x)$ que leva apenas $x$ como entrada.
Dada a abundância de polissemia e semântica complexa em linguagens naturais,
representações independentes de contexto têm limitações óbvias.
Por exemplo, a palavra "guindaste" em contextos
"um guindaste está voando" e "um motorista de guindaste veio" têm significados completamente diferentes;
assim, a mesma palavra pode receber diferentes representações dependendo dos contextos.

Isso motiva o desenvolvimento de representações de palavras *sensíveis ao contexto*,
onde as representações de palavras dependem de seus contextos.
Portanto, uma representação sensível ao contexto do token $x$ é uma função $f(x, c(x))$
dependendo de $x$ e de seu contexto $c(x)$.
Representações contextuais populares
incluem TagLM (tagger de sequência aumentada de modelo de linguagem) :cite:`Peters.Ammar.Bhagavatula.ea.2017`,
CoVe (vetores de contexto) :cite:`McCann.Bradbury.Xiong.ea.2017`,
e ELMo (Embeddings from Language Models) :cite:`Peters.Neumann.Iyyer.ea.2018`.

Por exemplo, tomando toda a sequência como entrada,
ELMo é uma função que atribui uma representação a cada palavra da sequência de entrada.
Especificamente, o ELMo combina todas as representações da camada intermediária do LSTM bidirecional pré-treinado como a representação de saída.
Em seguida, a representação ELMo será adicionada ao modelo supervisionado existente de uma tarefa downstream
como recursos adicionais, como concatenando a representação ELMo e a representação original (por exemplo, GloVe) de tokens no modelo existente.
Por um lado,
todos os pesos no modelo LSTM bidirecional pré-treinado são congelados após as representações ELMo serem adicionadas.
Por outro lado,
o modelo supervisionado existente é personalizado especificamente para uma determinada tarefa.
Aproveitando os melhores modelos diferentes para diferentes tarefas naquele momento,
adicionar ELMo melhorou o estado da arte em seis tarefas de processamento de linguagem natural:
análise de sentimento, inferência de linguagem natural,
rotulagem de função semântica, resolução de co-referência,
reconhecimento de entidade nomeada e resposta a perguntas.


## De Task-Specific para Task-Agnostic

Embora o ELMo tenha melhorado significativamente as soluções para um conjunto diversificado de tarefas de processamento de linguagem natural,
cada solução ainda depende de uma arquitetura *específica para a tarefa*.
No entanto, é praticamente não trivial criar uma arquitetura específica para cada tarefa de processamento de linguagem natural.
O modelo GPT (Generative Pre-Training) representa um esforço na concepção
um modelo *agnóstico de tarefa* geral para representações sensíveis ao contexto :cite:`Radford.Narasimhan.Salimans.ea.2018`.
Construído em um decodificador de transformador,
O GPT pré-treina um modelo de linguagem que será usado para representar sequências de texto.
Ao aplicar GPT a uma tarefa downstream,
a saída do modelo de linguagem será alimentada em uma camada de saída linear adicionada
para prever o rótulo da tarefa.
Em nítido contraste com o ELMo que congela os parâmetros do modelo pré-treinado,
GPT ajusta *todos* os parâmetros no decodificador de transformador pré-treinado
durante a aprendizagem supervisionada da tarefa a jusante.
GPT foi avaliada em doze tarefas de inferência de linguagem natural,
resposta a perguntas, similaridade de sentenças e classificação,
e melhorou o estado da arte em nove deles com mudanças mínimas
para a arquitetura do modelo.

No entanto, devido à natureza autoregressiva dos modelos de linguagem,
O GPT apenas olha para a frente (da esquerda para a direita).
Em contextos "fui ao banco para depositar dinheiro" e "fui ao banco para me sentar",
como "banco" é sensível ao contexto à sua esquerda,
GPT retornará a mesma representação para "banco",
embora tenha significados diferentes.

## BERT: Combinando o melhor dos dois mundos

Como nós vimos,
O ELMo codifica o contexto bidirecionalmente, mas usa arquiteturas específicas para tarefas;
enquanto o GPT é agnóstico em relação à tarefa, mas codifica o contexto da esquerda para a direita.
Combinando o melhor dos dois mundos,
BERT (Representações de Codificador Bidirecional de Transformadores)
codifica o contexto bidirecionalmente e requer mudanças mínimas de arquitetura
para uma ampla gama de tarefas de processamento de linguagem natural :cite:`Devlin.Chang.Lee.ea.2018`.
Usando um codificador de transformador pré-treinado,
O BERT é capaz de representar qualquer token com base em seu contexto bidirecional.
Durante a aprendizagem supervisionada de tarefas posteriores,
O BERT é semelhante ao GPT em dois aspectos.
Primeiro, as representações de BERT serão alimentadas em uma camada de saída adicionada,
com mudanças mínimas na arquitetura do modelo, dependendo da natureza das tarefas,
como a previsão para cada token versus a previsão para a sequência inteira.
Segundo,
todos os parâmetros do codificador de transformador pré-treinado são ajustados,
enquanto a camada de saída adicional será treinada do zero.
:numref:`fig_elmo-gpt-bert` descreve as diferenças entre ELMo, GPT e BERT.

![A comparison of ELMo, GPT, and BERT.](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`

O BERT melhorou ainda mais o estado da arte em onze tarefas de processamento de linguagem natural
sob categorias amplas de i) classificação de texto único (por exemplo, análise de sentimento), ii) classificação de pares de texto (por exemplo, inferência de linguagem natural),
iii) resposta a perguntas, iv) marcação de texto (por exemplo, reconhecimento de entidade nomeada).
Tudo proposto em 2018,
de ELMo sensível ao contexto a GPT e BERT agnósticos de tarefa,
O pré-treinamento conceitualmente simples, mas empiricamente poderoso, de representações profundas para linguagens naturais revolucionou as soluções para várias tarefas de processamento de linguagem natural.

No resto deste capítulo,
vamos mergulhar no pré-treinamento de BERT.
Quando os aplicativos de processamento de linguagem natural são explicados em :numref:`chap_nlp_app`,
ilustraremos o ajuste fino de BERT para aplicações downstream.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## Representação de entrada
:label:`subsec_bert_input_rep`

No processamento de linguagem natural,
algumas tarefas (por exemplo, análise de sentimento) usam um único texto como entrada,
enquanto em algumas outras tarefas (por exemplo, inferência de linguagem natural),
a entrada é um par de sequências de texto.
A sequência de entrada de BERT representa sem ambiguidade texto único e pares de texto.
Na antiga,
a sequência de entrada de BERT é a concatenação de
o token de classificação especial “&lt;sep&gt;”,
tokens de uma sequência de texto,
e o token de separação especial “&lt;sep&gt;”.
No ultimo,
a sequência de entrada de BERT é a concatenação de
“&lt;sep&gt;”, tokens da primeira sequência de texto,
“&lt;sep&gt;”, tokens da segunda sequência de texto e “&lt;sep&gt;”.
Iremos distinguir de forma consistente a terminologia "sequência de entrada de BERT"
de outros tipos de "sequências".
Por exemplo, uma *sequência de entrada de BERT* pode incluir uma *sequência de texto* ou duas *sequências de texto*.

Para distinguir pares de texto,
o segmento aprendido embeddings $\mathbf{e}_A$ e $\mathbf{e}_B$
são adicionados aos embeddings de token da primeira e da segunda sequência, respectivamente.
Para entradas de texto único, apenas $\mathbf{e}_A$ é usado.

O seguinte `get_tokens_and_segments` leva uma ou duas frases
como entrada, em seguida, retorna tokens da sequência de entrada BERT
e seus IDs de segmento correspondentes.

```{.python .input}
#@tab all
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

O BERT escolhe o codificador do transformador como sua arquitetura bidirecional.
Comum no codificador do transformador,
embeddings posicionais são adicionados em cada posição da sequência de entrada BERT.
No entanto, diferente do codificador do transformador original,
O BERT usa embeddings posicionais *aprendíveis*.
Para resumir, :numref:`fig_bert-input` mostra que
os embeddings da sequência de entrada de BERT são a soma
dos embeddings de token, embeddings de segmento e embeddings posicionais.

![Os embeddings da sequência de entrada de BERT são a soma
dos embeddings de token, embeddings de segmento e embeddings posicionais.](../img/bert-input.svg)
:label:`fig_bert-input`

A seguinte classe `BERTEncoder` é semelhante à classe `TransformerEncoder`
conforme implementado em :numref:`sec_transformer`.
Diferente de `TransformerEncoder`, `BERTEncoder` usa
embeddings de segmento e embeddings posicionais aprendíveis.

```{.python .input}
#@save
class BERTEncoder(nn.Block):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab pytorch
#@save
class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

Suppose that the vocabulary size is 10,000.
To demonstrate forward inference of `BERTEncoder`,
let us create an instance of it and initialize its parameters.

Suponha que o tamanho do vocabulário seja 10.000.
Para demonstrar a inferência direta de `BERTEncoder`,
vamos criar uma instância dele e inicializar seus parâmetros.

```{.python .input}
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_layers, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_layers, dropout)
encoder.initialize()
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
```

Definimos `tokens` como sendo 2 sequências de entrada BERT de comprimento 8,
onde cada token é um índice do vocabulário.
A inferência direta de `BERTEncoder` com os `tokens` de entrada
retorna o resultado codificado onde cada token é representado por um vetor
cujo comprimento é predefinido pelo hiperparâmetro `num_hiddens`.
Esse hiperparâmetro geralmente é conhecido como *tamanho oculto*
(número de unidades ocultas) do codificador do transformador.

```{.python .input}
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab pytorch
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

## Tarefas de pré-treinamento
:label:`subsec_bert_pretraining_tasks`

A inferência direta de `BERTEncoder` dá a representação de BERT
de cada token do texto de entrada e o inserido
tokens especiais “&lt;cls&gt;” e “&lt;seq&gt;”.
A seguir, usaremos essas representações para calcular a função de perda
para pré-treinamento de BERT.
O pré-treinamento é composto pelas duas tarefas a seguir:
modelagem de linguagem mascarada e previsão da próxima frase.

### Modelagem de linguagem mascarada
:label:`subsec_mlm`

Conforme ilustrado em :numref:`sec_language_model`,
um modelo de linguagem prevê um token usando o contexto à sua esquerda.
Para codificar o contexto bidirecionalmente para representar cada token,
BERT mascara tokens aleatoriamente e usa tokens do contexto bidirecional para
prever os tokens mascarados.
Esta tarefa é conhecida como *modelo de linguagem mascarada*.

Nesta tarefa de pré-treinamento,
15% dos tokens serão selecionados aleatoriamente como os tokens mascarados para previsão.
Para prever um token mascarado sem trapacear usando o rótulo,
uma abordagem direta é sempre substituí-lo por um “&lt;mask&gt;” especial token na sequência de entrada BERT.
No entanto, o token especial artificial “&lt;mask&gt;” nunca aparecerá
no ajuste fino.
Para evitar essa incompatibilidade entre o pré-treinamento e o ajuste fino,
se um token for mascarado para previsão (por exemplo, "ótimo" foi selecionado para ser mascarado e previsto em "este filme é ótimo"),
na entrada, ele será substituído por:

* uma “&lt;mask&gt;” especial token 80% do tempo (por exemplo, "este filme é ótimo" torna-se "este filme é &lt;mask&gt;");
* um token aleatório 10% do tempo (por exemplo, "este filme é ótimo" torna-se "este filme é uma bebida");
* o token de rótulo inalterado em 10% do tempo (por exemplo, "este filme é ótimo" torna-se "este filme é ótimo").

Observe que por 10% de 15% do tempo, um token aleatório é inserido.
Este ruído ocasional encoraja o BERT a ser menos inclinado para o token mascarado (especialmente quando o token de rótulo permanece inalterado) em sua codificação de contexto bidirecional.

Implementamos a seguinte classe `MaskLM` para prever tokens mascarados
na tarefa de modelo de linguagem mascarada de pré-treinamento de BERT.
A previsão usa um MLP de uma camada oculta (`self.mlp`).
Na inferência direta, são necessárias duas entradas:
o resultado codificado de `BERTEncoder` e as posições do token para predição.
A saída são os resultados da previsão nessas posições.

```{.python .input}
#@save
class MaskLM(nn.Block):
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `np.array([0, 0, 0, 1, 1, 1])`
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

To demonstrate the forward inference of `MaskLM`,
we create its instance `mlm` and initialize it.
Recall that `encoded_X` from the forward inference of `BERTEncoder`
represents 2 BERT input sequences.
We define `mlm_positions` as the 3 indices to predict in either BERT input sequence of `encoded_X`.
The forward inference of `mlm` returns prediction results `mlm_Y_hat`
at all the masked positions `mlm_positions` of `encoded_X`.
For each prediction, the size of the result is equal to the vocabulary size.

Para demonstrar a inferência direta de `MaskLM`,
nós criamos sua instância `mlm` e a inicializamos.
Lembre-se de que `encoded_X` da inferência direta de`BERTEncoder`
representa 2 sequências de entrada de BERT.
Definimos `mlm_positions` como os 3 índices a serem previstos em qualquer sequência de entrada de BERT de` encoded_X`.
A inferência direta de `mlm` retorna os resultados de predição` mlm_Y_hat`
em todas as posições mascaradas `mlm_positions` de` encoded_X`.
Para cada previsão, o tamanho do resultado é igual ao tamanho do vocabulário.

```{.python .input}
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab pytorch
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

With the ground truth labels `mlm_Y` of the predicted tokens `mlm_Y_hat` under masks,
we can calculate the cross entropy loss of the masked language model task in BERT pretraining.

```{.python .input}
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab pytorch
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

### Next Sentence Prediction
:label:`subsec_nsp`

Although masked language modeling is able to encode bidirectional context
for representing words, it does not explicitly model the logical relationship
between text pairs.
To help understand the relationship between two text sequences,
BERT considers a binary classification task, *next sentence prediction*, in its pretraining.
When generating sentence pairs for pretraining,
for half of the time they are indeed consecutive sentences with the label "True";
while for the other half of the time the second sentence is randomly sampled from the corpus with the label "False".

The following `NextSentencePred` class uses a one-hidden-layer MLP
to predict whether the second sentence is the next sentence of the first
in the BERT input sequence.
Due to self-attention in the transformer encoder,
the BERT representation of the special token “&lt;cls&gt;”
encodes both the two sentences from the input.
Hence, the output layer (`self.output`) of the MLP classifier takes `X` as the input,
where `X` is the output of the MLP hidden layer whose input is the encoded “&lt;cls&gt;” token.

```{.python .input}
#@save
class NextSentencePred(nn.Block):
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

```{.python .input}
#@tab pytorch
#@save
class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

We can see that the forward inference of an `NextSentencePred` instance
returns binary predictions for each BERT input sequence.

```{.python .input}
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab pytorch
# PyTorch by default won't flatten the tensor as seen in mxnet where, if
# flatten=True, all but the first axis of input data are collapsed together
encoded_X = torch.flatten(encoded_X, start_dim=1)
# input_shape for NSP: (batch size, `num_hiddens`)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

The cross-entropy loss of the 2 binary classifications can also be computed.

```{.python .input}
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab pytorch
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

It is noteworthy that all the labels in both the aforementioned pretraining tasks
can be trivially obtained from the pretraining corpus without manual labeling effort.
The original BERT has been pretrained on the concatenation of BookCorpus :cite:`Zhu.Kiros.Zemel.ea.2015`
and English Wikipedia.
These two text corpora are huge:
they have 800 million words and 2.5 billion words, respectively.


## Putting All Things Together

When pretraining BERT, the final loss function is a linear combination of
both the loss functions for masked language modeling and next sentence prediction.
Now we can define the `BERTModel` class by instantiating the three classes
`BERTEncoder`, `MaskLM`, and `NextSentencePred`.
The forward inference returns the encoded BERT representations `encoded_X`,
predictions of masked language modeling `mlm_Y_hat`,
and next sentence predictions `nsp_Y_hat`.

```{.python .input}
#@save
class BERTModel(nn.Block):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_layers, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## Summary

* Word embedding models such as word2vec and GloVe are context-independent. They assign the same pretrained vector to the same word regardless of the context of the word (if any). It is hard for them to handle well polysemy or complex semantics in natural languages.
* For context-sensitive word representations such as ELMo and GPT, representations of words depend on their contexts.
* ELMo encodes context bidirectionally but uses task-specific architectures (however, it is practically non-trivial to craft a specific architecture for every natural language processing task); while GPT is task-agnostic but encodes context left-to-right.
* BERT combines the best of both worlds: it encodes context bidirectionally and requires minimal architecture changes for a wide range of natural language processing tasks.
* The embeddings of the BERT input sequence are the sum of the token embeddings, segment embeddings, and positional embeddings.
* Pretraining BERT is composed of two tasks: masked language modeling and next sentence prediction. The former is able to encode bidirectional context for representing words, while the later explicitly models the logical relationship between text pairs.


## Exercises

1. Why does BERT succeed?
1. All other things being equal, will a masked language model require more or fewer pretraining steps to converge than a left-to-right language model? Why?
1. In the original implementation of BERT, the positionwise feed-forward network in `BERTEncoder` (via `d2l.EncoderBlock`) and the fully-connected layer in `MaskLM` both use the Gaussian error linear unit (GELU) :cite:`Hendrycks.Gimpel.2016` as the activation function. Research into the difference between GELU and ReLU.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/388)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1490)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc0NTAzOTU3MiwxOTI3NzQ2NTE1LDEwMD
AxMzA5MjldfQ==
-->
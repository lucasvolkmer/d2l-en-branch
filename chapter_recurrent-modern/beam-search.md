# Beam SearchPesquisa de feixe
:label:`sec_beam-search`

InEm :numref:`sec_seq2seq`,
we predicted the outputprevimos a sequeênceia de saída token bypor token
until the special end-of-até o final da sequeênceia especial "&lt;eos&gt;" token
is predicted.
In this section,
we will begin with formalizing this *greedy search* strategy
and exploring issues with it,
then compare this strategy with othersímbolo
é previsto.
Nesta secção,
começaremos formalizando essa estratégia de *busca gananciosa*
e explorando problemas com isso,
em seguida, compare essa estratégia com outras alternativeas:
* pesquisa exhaustive search* and *beam searcha* e *pesquisa por feixe*.

Antes de uma introdução formal à busca gananciosa,
vamos formalizar o problema de pesquisa
usando
a mesma notação matemática de :numref:`sec_seq2seq`.
A qualquer momento, passo $t'$,
a probabilidade de saída do decodificador $y_{t '}$
é condicional
na subseqüência de saída
$y_1, \ldots, y_{t'-1}$ antes de $t'$ e
a variável de contexto $\mathbf{c}$ que
codifica as informações da sequência de entrada.
Para quantificar o custo computacional,
denotar por
$\mathcal{Y}$ (contém "&lt;eos&gt;")
o vocabulário de saída.
Portanto, a cardinalidade $\left|\mathcal{Y}\right|$ deste conjunto de vocabulário
é o tamanho do vocabulário.
Vamos também especificar o número máximo de tokens
de uma sequência de saída como $T'$.
Como resultado,
nosso objetivo é procurar um resultado ideal
de todo o
$\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$
possíveis sequências de saída.
Claro,
para todas essas sequências de saída,
porções incluindo e após "& lt; eos & gt;" será descartado
na saída real.

## Busca Gulosa

Primeiro, vamos dar uma olhada em
uma estratégia simples: *busca gananciosa*.
Esta estratégia foi usada para prever sequências em :numref:`sec_seq2seq`.
Em busca gananciosa,
a qualquer momento, etapa $t'$ da sequência de saída,
nós procuramos pelo token
com a maior probabilidade condicional de $\mathcal{Y}$, ou seja,

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$

como a saída.
Uma vez que "&lt;eos&gt;" é emitida ou a sequência de saída atingiu seu comprimento máximo $T'$, a sequência de saída é concluída.

Então, o que pode dar errado com a busca gananciosa?
Na verdade,
a *sequência ideal*
deve ser a sequência de saída
com o máximo
$\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$,
qual é
a probabilidade condicional de gerar uma sequência de saída com base na sequência de entrada.
Infelizmente, não há garantia
que a sequência ótima será obtida
por busca gananciosa.

![At each time step, greedy search selects the token with the highest conditional probability.](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

Vamos ilustrar com um exemplo.
Suponha que existam quatro tokens
"A", "B", "C" e "&lt;eos&gt;" no dicionário de saída.
Em: numref: `fig_s2s-prob1`,
os quatro números em cada etapa de tempo representam as probabilidades condicionais de gerar "A", "B", "C" e "&lt;eos&gt;" nessa etapa de tempo, respectivamente.
Em cada etapa de tempo,
a pesquisa gananciosa seleciona o token com a probabilidade condicional mais alta.
Portanto, a sequência de saída "A", "B", "C" e "&lt;eos&gt;" será previsto
in :numref:`fig_s2s-prob1`.
A probabilidade condicional desta sequência de saída é $0.5\times0.4\times0.4\times0.6 = 0.048$.

![The four numbers under each time step represent the conditional probabilities of generating "A", "B", "C", and "&lt;eos&gt;" at that time step.  At time step 2, the token "C", which has the second highest conditional probability, is selected.](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`


Next, let us look at another example 
in :numref:`fig_s2s-prob2`. 
Unlike in :numref:`fig_s2s-prob1`, 
at time step 2
we select the token "C"
in :numref:`fig_s2s-prob2`, 
which has the *second* highest conditional probability.
Since the output subsequences at time steps 1 and 2, 
on which time step 3 is based, 
have changed from "A" and "B" in :numref:`fig_s2s-prob1` to "A" and "C" in :numref:`fig_s2s-prob2`, 
the conditional probability of each token 
at time step 3 has also changed in :numref:`fig_s2s-prob2`. 
Suppose that we choose the token "B" at time step 3. 
Now time step 4 is conditional on
the output subsequence at the first three time steps
"A", "C", and "B", 
which is different from "A", "B", and "C" in :numref:`fig_s2s-prob1`. 
Therefore, the conditional probability of generating each token at time step 4 in :numref:`fig_s2s-prob2` is also different from that in :numref:`fig_s2s-prob1`. 
As a result, 
the conditional probability of the output sequence "A", "C", "B", and "&lt;eos&gt;" 
in :numref:`fig_s2s-prob2`
is $0.5\times0.3 \times0.6\times0.6=0.054$, 
which is greater than that of greedy search in :numref:`fig_s2s-prob1`. 
In this example, 
the output sequence "A", "B", "C", and "&lt;eos&gt;" obtained by the greedy search is not an optimal sequence.

## Exhaustive Search

If the goal is to obtain the optimal sequence, we may consider using *exhaustive search*: 
exhaustively enumerate all the possible output sequences with their conditional probabilities,
then output the one 
with the highest conditional probability.

Although we can use exhaustive search to obtain the optimal sequence, 
its computational cost $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ is likely to be excessively high. 
For example, when $|\mathcal{Y}|=10000$ and $T'=10$, we will need to evaluate $10000^{10} = 10^{40}$ sequences. This is next to impossible!
On the other hand,
the computational cost of greedy search is 
$\mathcal{O}(\left|\mathcal{Y}\right|T')$: 
it is usually significantly smaller than
that of exhaustive search. For example, when $|\mathcal{Y}|=10000$ and $T'=10$, we only need to evaluate $10000\times10=10^5$ sequences.


## Beam Search

Decisions about sequence searching strategies
lie on a spectrum,
with easy questions at either extreme.
What if only accuracy matters?
Obviously, exhaustive search.
What if only computational cost matters?
Clearly, greedy search.
A real-world application usually asks
a complicated question,
somewhere in between those two extremes.

*Beam search* is an improved version of greedy search. It has a hyperparameter named *beam size*, $k$. 
At time step 1, 
we select $k$ tokens with the highest conditional probabilities.
Each of them will be the first token of 
$k$ candidate output sequences, respectively.
At each subsequent time step, 
based on the $k$ candidate output sequences
at the previous time step,
we continue to select $k$ candidate output sequences 
with the highest conditional probabilities 
from $k\left|\mathcal{Y}\right|$ possible choices.

![The process of beam search (beam size: 2, maximum length of an output sequence: 3). The candidate output sequences are $A$, $C$, $AB$, $CE$, $ABD$, and $CED$.](../img/beam-search.svg)
:label:`fig_beam-search`


:numref:`fig_beam-search` demonstrates the 
process of beam search with an example. 
Suppose that the output vocabulary
contains only five elements: 
$\mathcal{Y} = \{A, B, C, D, E\}$, 
where one of them is “&lt;eos&gt;”. 
Let the beam size be 2 and 
the maximum length of an output sequence be 3. 
At time step 1, 
suppose that the tokens with the highest conditional probabilities $P(y_1 \mid \mathbf{c})$ are $A$ and $C$. At time step 2, for all $y_2 \in \mathcal{Y},$ we compute 

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$  

and pick the largest two among these ten values, say
$P(A, B \mid \mathbf{c})$ and $P(C, E \mid \mathbf{c})$.
Then at time step 3, for all $y_3 \in \mathcal{Y}$, we compute 

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$ 

and pick the largest two among these ten values, say 
$P(A, B, D \mid \mathbf{c})$   and  $P(C, E, D \mid  \mathbf{c}).$
As a result, we get six candidates output sequences: (i) $A$; (ii) $C$; (iii) $A$, $B$; (iv) $C$, $E$; (v) $A$, $B$, $D$; and (vi) $C$, $E$, $D$. 


In the end, we obtain the set of final candidate output sequences based on these six sequences (e.g., discard portions including and after “&lt;eos&gt;”).
Then
we choose the sequence with the highest of the following score as the output sequence:

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$
:eqlabel:`eq_beam-search-score`

where $L$ is the length of the final candidate sequence and $\alpha$ is usually set to 0.75. 
Since a longer sequence has more logarithmic terms in the summation of :eqref:`eq_beam-search-score`,
the term $L^\alpha$ in the denominator penalizes
long sequences.

The computational cost of beam search is $\mathcal{O}(k\left|\mathcal{Y}\right|T')$. 
This result is in between that of greedy search and that of exhaustive search. In fact, greedy search can be treated as a special type of beam search with 
a beam size of 1. 
With a flexible choice of the beam size,
beam search provides a tradeoff between
accuracy versus computational cost.



## Summary

* Sequence searching strategies include greedy search, exhaustive search, and beam search.
* Beam search provides a tradeoff between accuracy versus computational cost via its flexible choice of the beam size.


## Exercises

1. Can we treat exhaustive search as a special type of beam search? Why or why not?
1. Apply beam search in the machine translation problem in :numref:`sec_seq2seq`. How does the beam size affect the translation results and the prediction speed?
1. We used language modeling for generating text following  user-provided prefixes in :numref:`sec_rnn_scratch`. Which kind of search strategy does it use? Can you improve it?

[Discussions](https://discuss.d2l.ai/t/338)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTkzNTc3MDI5M119
-->
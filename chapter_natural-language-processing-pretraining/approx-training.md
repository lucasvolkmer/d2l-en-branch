# Treinamento Aproximado
:label:`sec_approx_train`

Lembre-se do conteúdo da última seção. O principal recurso do modelo skip-gram é o uso de operações softmax para calcular a probabilidade condicional de gerar a palavra de contexto $w_o$ com base na palavra-alvo central fornecida $w_c$.

$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}.$$

A perda logarítmica correspondente à probabilidade condicional é dada como

$$-\log P(w_o \mid w_c) =
-\mathbf{u}_o^\top \mathbf{v}_c + \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$

Como a operação softmax considerou que a palavra de contexto poderia ser qualquer palavra no dicionário $\mathcal{V}$, a perda mencionada acima na verdade inclui a soma do número de itens no tamanho do dicionário. Na última seção, sabemos que para o modelo skip-gram e o modelo CBOW, porque ambos obtêm a probabilidade condicional usando uma operação softmax, o cálculo do gradiente para cada etapa contém a soma do número de itens no tamanho do dicionário. Para dicionários maiores com centenas de milhares ou até milhões de palavras, a sobrecarga para calcular cada gradiente pode ser muito alta. Para reduzir essa complexidade computacional, apresentaremos dois métodos de treinamento aproximados nesta seção: amostragem negativa e softmax hierárquico. Como não há grande diferença entre o modelo skip-gram e o modelo CBOW, usaremos apenas o modelo skip-gram como um exemplo para apresentar esses dois métodos de treinamento nesta seção.


## Amostragem Negativa
:label:`subsec_negative-sampling`

A amostragem negativa modifica a função objetivo original. Dada uma janela de contexto para a palavra alvo central $w_c$, vamos tratá-la como um evento para a palavra de contexto $w_o$ aparecer na janela de contexto e calcular a probabilidade deste evento a partir de

$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

Aqui, a função $\sigma$ tem a mesma definição que a função de ativação sigmóide:

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$

Consideraremos primeiro o treinamento do vetor de palavras maximizando a probabilidade conjunta de todos os eventos na sequência de texto. Dada uma sequência de texto de comprimento $T$, assumimos que a palavra no passo de tempo $t$ é $w^{(t)}$ e o tamanho da janela de contexto é $m$. Agora consideramos maximizar a probabilidade conjunta

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$

No entanto, os eventos incluídos no modelo consideram apenas exemplos positivos. Nesse caso, apenas quando todos os vetores de palavras são iguais e seus valores se aproximam do infinito, a probabilidade conjunta acima pode ser maximizada para 1. Obviamente, esses vetores de palavras não têm sentido. A amostragem negativa torna a função objetivo mais significativa, amostrando com uma adição de exemplos negativos. Assuma que o evento $P$ ocorre quando a palavra de contexto $w_o$ aparece na janela de contexto da palavra alvo central $w_c$, e nós amostramos $K$ palavras que não aparecem na janela de contexto de acordo com a distribuição $P(w)$ para atuar como palavras de ruído. Assumimos que o evento para a palavra de ruído $w_k$ ($k=1, \ldots, K$) não aparecer na janela de contexto da palavra de destino central $w_c$ é $N_k$. Suponha que os eventos $P$ e $N_1, \ldots, N_K$ para os exemplos positivos e negativos sejam independentes um do outro. Ao considerar a amostragem negativa, podemos reescrever a probabilidade conjunta acima, que considera apenas os exemplos positivos, como

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

Aqui, a probabilidade condicional é aproximada a ser
$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$


Seja o índice de sequência de texto da palavra $w^{(t)}$ no passo de tempo $t$ $i_t$ e $h_k$ para a palavra de ruído $w_k$ no dicionário. A perda logarítmica para a probabilidade condicional acima é

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$

Aqui, o cálculo do gradiente em cada etapa do treinamento não está mais relacionado ao tamanho do dicionário, mas linearmente relacionado a $K$. Quando $K$ assume uma constante menor, a amostragem negativa tem uma sobrecarga computacional menor para cada etapa.

## Hierárquico Softmax

O softmax hierárquico é outro tipo de método de treinamento aproximado. Ele usa uma árvore binária para a estrutura de dados conforme ilustrado em :numref:`fig_hi_softmax`, com os nós folha da árvore representando cada palavra no dicionário $\mathcal{V}$.

![Hierarchical Softmax. Each leaf node of the tree represents a word in the dictionary. ](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

Assumimos que $L(w)$ é o número de nós no caminho (incluindo os nós raiz e folha) do nó raiz da árvore binária ao nó folha da palavra $w$. Seja $n(w, j)$ o nó $j^\mathrm{th}$ neste caminho, com o vetor de palavra de contexto $\mathbf{u}_{n(w, j)}$. Usamos :numref:`fig_hi_softmax` como exemplo, então $L(w_3) = 4$. O softmax hierárquico aproximará a probabilidade condicional no modelo skip-gram como

$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

Here the $\sigma$ function has the same definition as the sigmoid activation function, and $\text{leftChild}(n)$ is the left child node of node $n$. If $x$ is true, $[\![x]\!] = 1$; otherwise $[\![x]\!] = -1$.
Now, we will compute the conditional probability of generating word $w_3$ based on the given word $w_c$ in :numref:`fig_hi_softmax`. We need to find the inner product of word vector $\mathbf{v}_c$ (for word $w_c$) and each non-leaf node vector on the path from the root node to $w_3$. Because, in the binary tree, the path from the root node to leaf node $w_3$ needs to be traversed left, right, and left again (the path with the bold line in :numref:`fig_hi_softmax`), we get

Aqui, a função $\sigma$ tem a mesma definição que a função de ativação sigmóide, e $\text{leftChild}(n)$ é o nó filho esquerdo do nó $n$. Se $x$ for verdadeiro, $[\![x]\!] = 1$; caso contrário, $[\![x]\!] = -1$.
Agora, vamos calcular a probabilidade condicional de gerar a palavra $w_3$ com base na palavra $w_c$ dada em :numref:`fig_hi_softmax`. Precisamos encontrar o produto interno do vetor palavra $\mathbf{v}_c$ (para a palavra $w_c$) e cada vetor de nó não folha no caminho do nó raiz para $w_3$. Porque, na árvore binária, o caminho do nó raiz ao nó folha $w_3$ precisa ser percorrido para a esquerda, direita e esquerda novamente (o caminho com a linha em negrito em :numref:`fig_hi_softmax`), obtemos

$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

Because $\sigma(x)+\sigma(-x) = 1$, the condition that the sum of the conditional probability of any word generated based on the given central target word $w_c$ in dictionary $\mathcal{V}$ be 1 will also suffice:

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$

In addition, because the order of magnitude for $L(w_o)-1$ is $\mathcal{O}(\text{log}_2|\mathcal{V}|)$, when the size of dictionary $\mathcal{V}$ is large, the computational overhead for each step in the hierarchical softmax training is greatly reduced compared to situations where we do not use approximate training.

## Summary

* Negative sampling constructs the loss function by considering independent events that contain both positive and negative examples. The gradient computational overhead for each step in the training process is linearly related to the number of noise words we sample.
* Hierarchical softmax uses a binary tree and constructs the loss function based on the path from the root node to the leaf node. The gradient computational overhead for each step in the training process is related to the logarithm of the dictionary size.

## Exercises

1. Before reading the next section, think about how we should sample noise words in negative sampling.
1. What makes the last formula in this section hold?
1. How can we apply negative sampling and hierarchical softmax in the skip-gram model?

[Discussions](https://discuss.d2l.ai/t/382)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTY2OTg3NzY5MF19
-->
# Feature-Rich RSistemas de recommender Systemação com muitos recursos


Os dados de interação são a indicação mais básica das preferências e interesses dos usuários. Ele desempenha um papel crítico em modelos anteriores introduzidos. No entanto, os dados de interação geralmente são extremamente esparsos e podem ser barulhentos às vezes. Para resolver esse problema, podemos integrar informações colaterais, como recursos de itens, perfis de usuários e até mesmo em que contexto a interação ocorreu no modelo de recomendação. A utilização desses recursos é útil para fazer recomendações, pois esses recursos podem ser um indicador eficaz dos interesses dos usuários, especialmente quando faltam dados de interação. Como tal, é essencial que os modelos de recomendação também tenham a capacidade de lidar com esses recursos e fornecer ao modelo algum conhecimento de conteúdo / contexto. Para demonstrar este tipo de modelos de recomendação, apresentamos outra tarefa sobre a taxa de cliques (CTR) para recomendações de publicidade online: cite: `McMahan.Holt.Sculley.ea.2013` e apresente dados anônimos de publicidade. Os serviços de publicidade direcionados têm atraído a atenção generalizada e muitas vezes são enquadrados como mecanismos de recomendação. Recomendar anúncios que correspondam aos gostos e interesses pessoais dos usuários é importante para a melhoria da taxa de cliques.


Os profissionais de marketing digital usam publicidade online para exibir anúncios aos clientes. A taxa de cliques é uma métrica que mede o número de cliques que os anunciantes recebem em seus anúncios por número de impressões e é expressa como uma porcentagem calculada com a fórmula:

$$ \text{CTR} = \frac{\#\text{Clicks}} {\#\text{Impressions}} \times 100 \% .$$

A taxa de cliques é um sinal importante que indica a eficácia dos algoritmos de previsão. A previsão da taxa de cliques é uma tarefa de prever a probabilidade de que algo em um site será clicado. Os modelos de previsão de CTR não podem ser empregados apenas em sistemas de publicidade direcionados, mas também em sistemas de recomendação de itens gerais (por exemplo, filmes, notícias, produtos), campanhas de e-mail e até mesmo mecanismos de pesquisa. Também está intimamente relacionado à satisfação do usuário, taxa de conversão e pode ser útil na definição de metas de campanha, pois pode ajudar os anunciantes a definir expectativas realistas.

```{.python .input}
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## Um conjunto de dados de publicidade online

With the considerable advancements of Internet and mobile technology, online advertising has become an important income resource and generates vast majority of revenue in the Internet industry. It is important to display relevant advertisements or advertisements that pique users' interests so that casual visitors can be converted into paying customers. The dataset we introduced is an online advertising dataset. It consists of 34 fields, with the first column representing the target variable that indicates if an ad was clicked (1) or not (0). All the other columns are categorical features. The columns might represent the advertisement id, site or application id, device id, time, user profiles and so on. The real semantics of the features are undisclosed due to anonymization and privacy concern.

The following code downloads the dataset from our server and saves it into the local data folder.

```{.python .input  n=15}
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

There are a training set and a test set, consisting of 15000 and 3000 samples/lines, respectively.

## Dataset Wrapper

For the convenience of data loading, we implement a `CTRDataset` which loads the advertising dataset from the CSV file and can be used by `DataLoader`.

```{.python .input  n=13}
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

The following example loads the training data and print out the first record.

```{.python .input  n=16}
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

As can be seen, all the 34 fields are categorical features. Each value represents the one-hot index of the corresponding entry. The label $0$ means that it is not clicked. This `CTRDataset` can also be used to load other datasets such as the Criteo display advertising challenge [Dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) and the Avazu click-through rate prediction [Dataset](https://www.kaggle.com/c/avazu-ctr-prediction).  

## Summary 
* Click-through rate is an important metric that is used to measure the effectiveness of advertising systems and recommender systems.
* Click-through rate prediction is usually converted to a binary classification problem. The target is to predict whether an ad/item will be clicked or not based on given features.

## Exercises

* Can you load the Criteo and Avazu dataset with the provided `CTRDataset`. It is worth noting that the Criteo dataset consisting of real-valued features so you may have to revise the code a bit.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/405)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTY0NzE2Nzc5MiwtMjAzMzQ4MTc0NV19
-->
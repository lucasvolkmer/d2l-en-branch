# Visão geral dos sistemas de recomendação


Na última década, a Internet evoluiu para uma plataforma para serviços online de grande escala, o que mudou profundamente a maneira como nos comunicamos, lemos notícias, compramos produtos e assistimos filmes. Nesse ínterim, o número sem precedentes de itens (usamos o termo *item* para nos referir a filmes, notícias, livros e produtos) oferecidos online requer um sistema que pode nos ajudar a descobrir os itens de nossa preferência. Os sistemas de recomendação são, portanto, ferramentas poderosas de filtragem de informações que podem facilitar serviços personalizados e fornecer experiência sob medida para usuários individuais. Em suma, os sistemas de recomendação desempenham um papel fundamental na utilização da riqueza de dados disponíveis para fazer escolhas gerenciáveis. Hoje em dia, os sistemas de recomendação estão no centro de vários provedores de serviços online, como Amazon, Netflix e YouTube. Lembre-se do exemplo de livros de aprendizagem profunda recomendados pela Amazon em :numref:`subsec_recommender_systems`. Os benefícios de empregar sistemas de recomendação são duplos: por um lado, pode reduzir muito o esforço dos usuários em encontrar itens e aliviar a questão da sobrecarga de informações. Por outro lado, pode agregar valor ao negócio online
prestadores de serviços e é uma importante fonte de receita. Este capítulo irá apresentar os conceitos fundamentais, modelos clássicos e avanços recentes com aprendizado profundo no campo de sistemas de recomendação, juntamente com exemplos implementados.

![Illustration of the Recommendation Process](../img/rec-intro.svg)


## Filtragem colaborativa

We start the journey with the important concept in recommender systems---collaborative filtering
(CF), which was first coined by the Tapestry system :cite:`Goldberg.Nichols.Oki.ea.1992`, referring to "people collaborate to help one another perform the filtering process  in order to handle the large amounts of email and messages posted to newsgroups". This term has been enriched with more senses. In a broad sense, it is the process of
filtering for information or patterns using techniques involving collaboration among multiple users, agents, and data sources. CF has many forms and numerous CF methods proposed since its advent.  

Overall, CF techniques can be categorized into: memory-based CF, model-based CF, and their hybrid :cite:`Su.Khoshgoftaar.2009`. Representative memory-based CF techniques are nearest neighbor-based CF such as user-based CF and item-based CF :cite:`Sarwar.Karypis.Konstan.ea.2001`.  Latent factor models such as matrix factorization are examples of model-based CF.  Memory-based CF has limitations in dealing with sparse and large-scale data since it computes the similarity values based on common items.  Model-based methods become more popular with its
better capability in dealing with sparsity and scalability.  Many model-based CF approaches can be extended with neural networks, leading to more flexible and scalable models with the computation acceleration in deep learning :cite:`Zhang.Yao.Sun.ea.2019`.  In general, CF only uses the user-item interaction data to make predictions and recommendations. Besides CF, content-based and context-based recommender systems are also useful in incorporating the content descriptions of items/users and contextual signals such as timestamps and locations.  Obviously, we may need to adjust the model types/structures when different input data is available.



## Explicit Feedback and Implicit Feedback

To learn the preference of users, the system shall collect feedback from them.  The feedback can be either explicit or implicit :cite:`Hu.Koren.Volinsky.2008`. For example, [IMDB](https://www.imdb.com/) collects star ratings ranging from one to ten stars for movies. YouTube provides the thumbs-up and thumbs-down buttons for users to show their preferences.  It is apparent that gathering explicit feedback requires users to indicate their interests proactively.  Nonetheless, explicit feedback is not always readily available as many users may be reluctant to rate products. Relatively speaking, implicit feedback is often readily available since it is mainly concerned with modeling implicit behavior such as user clicks. As such, many recommender systems are centered on implicit feedback which indirectly reflects user's opinion through observing user behavior.  There are diverse forms of implicit feedback including purchase history, browsing history, watches and even mouse movements. For example, a user that purchased many books by the same author probably likes that author.   Note that implicit feedback is inherently noisy.  We can only *guess* their preferences and true motives. A user watched a movie does not necessarily indicate a positive view of that movie.



## Recommendation Tasks

A number of recommendation tasks have been investigated in the past decades.  Based on the domain of applications, there are movies recommendation, news recommendations, point-of-interest recommendation :cite:`Ye.Yin.Lee.ea.2011` and so forth.  It is also possible to differentiate the tasks based on the types of feedback and input data, for example, the rating prediction task aims to predict the explicit ratings. Top-$n$ recommendation (item ranking) ranks all items for each user personally based on the implicit feedback. If time-stamp information is also included, we can build sequence-aware recommendation :cite:`Quadrana.Cremonesi.Jannach.2018`.  Another popular task is called click-through rate prediction, which is also based on implicit feedback, but various categorical features can be utilized. Recommending for new users and recommending new items to existing users are called cold-start recommendation :cite:`Schein.Popescul.Ungar.ea.2002`.



## Summary

* Recommender systems are important for individual users and industries. Collaborative filtering is a key concept in recommendation.
* There are two types of feedbacks: implicit feedback and explicit feedback.  A number of recommendation tasks have been explored during the last decade.

## Exercises

1. Can you explain how recommender systems influence your daily life?
2. What interesting recommendation tasks do you think can be investigated?

[Discussions](https://discuss.d2l.ai/t/398)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0NTc5OTEzNjBdfQ==
-->
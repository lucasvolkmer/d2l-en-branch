# Usando Amazon SageMaker
:label:`sec_sagemaker`

Muitos aplicativos de aprendizado profundo requerem uma quantidade significativa de computação. Sua máquina local pode ser muito lenta para resolver esses problemas em um período de tempo razoável. Os serviços de computação em nuvem fornecem acesso a computadores mais poderosos para executar as partes deste livro com uso intensivo de GPU. Este tutorial o guiará pelo Amazon SageMaker: um serviço que permite que você execute este livro facilmente.


## Registro e login

Primeiro, precisamos registrar uma conta em https://aws.amazon.com/. Nós encorajamos você a usar a autenticação de dois fatores para segurança adicional. Também é uma boa ideia configurar o faturamento detalhado e alertas de gastos para evitar surpresas inesperadas no caso de você se esquecer de interromper qualquer instância em execução.
Observe que você precisará de um cartão de crédito.
Depois de fazer login em sua conta AWS, vá para seu [console](http://console.aws.amazon.com/) e pesquise "SageMaker" (consulte :numref:`fig_sagemaker`) e clique para abrir o painel SageMaker.

![Abra o painel SageMaker.](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`



## Criação de uma instância do SageMaker

A seguir, vamos criar uma instância de notebook conforme descrito em :numref:`fig_sagemaker-create`.

![Crie uma instância SageMaker.](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

O SageMaker fornece vários [tipos de instância](https://aws.amazon.com/sagemaker/pricing/instance-types/) de diferentes poder computacional e preços.
Ao criar uma instância, podemos especificar o nome da instância e escolher seu tipo.
Em :numref:`fig_sagemaker-create-2`, escolhemos `ml.p3.2xlarge`. Com uma GPU Tesla V100 e uma CPU de 8 núcleos, esta instância é poderosa o suficiente para a maioria dos capítulos.

![Escolha o tipo de instância.](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
Uma versão do notebook Jupyter deste livro para ajustar o SageMaker está disponível em https://github.com/d2l-ai/d2l-en-sagemaker. Podemos especificar a URL do repositório GitHub para permitir que o SageMaker clone este repositório durante a criação da instância, conforme mostrado em :numref:`fig_sagemaker-create-3`.
:end_tab:

:begin_tab:`pytorch`
Uma versão do notebook Jupyter deste livro para ajustar o SageMaker está disponível em https://github.com/d2l-ai/d2l-pytorch-sagemaker. Podemos especificar a URL do repositório GitHub para permitir que o SageMaker clone este repositório durante a criação da instância, conforme mostrado em :numref:`fig_sagemaker-create-3`.
:end_tab:

:begin_tab:`tensorflow`
A Jupyter notebook version of this book for fitting SageMaker is available at https://github.com/d2l-ai/d2l-tensorflow-sagemaker. We can specify this GitHub repository URL to let SageMaker clone this repository during instance creation, as shown in :numref:`fig_sagemaker-create-3`.
:end_tab:

![Specify the GitHub repository.](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`



## Running and Stopping an Instance

It may take a few minutes before the instance is ready.
When it is ready, you can click on the "Open Jupyter" link as shown in :numref:`fig_sagemaker-open`.

![Open Jupyter on the created SageMaker instance.](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`

Then, as shown in :numref:`fig_sagemaker-jupyter`, you may navigate through the Jupyter server running on this instance.

![The Jupyter server running on the SageMaker instance.](../img/sagemaker-jupyter.png)
:width:`400px`
:label:`fig_sagemaker-jupyter`

Running and editing Jupyter notebooks on the SageMaker instance is similar to what we have discussed in :numref:`sec_jupyter`.
After finishing your work, do not forget to stop the instance to avoid further charging, as shown in :numref:`fig_sagemaker-stop`.

![Stop a SageMaker instance.](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`


## Updating Notebooks

:begin_tab:`mxnet`
We will regularly update the notebooks in the [d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker) GitHub repository. You can simply use the `git pull` command to update to the latest version.
:end_tab:

:begin_tab:`pytorch`
We will regularly update the notebooks in the [d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker) GitHub repository. You can simply use the `git pull` command to update to the latest version.
:end_tab:

:begin_tab:`tensorflow`
We will regularly update the notebooks in the [d2l-ai/d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker) GitHub repository. You can simply use the `git pull` command to update to the latest version.
:end_tab:

First, you need to open a terminal as shown in :numref:`fig_sagemaker-terminal`.

![Open a terminal on the SageMaker instance.](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

You may want to commit your local changes before pulling the updates. Alternatively, you can simply ignore all your local changes with the following commands in the terminal.

:begin_tab:`mxnet`
```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`pytorch`
```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`tensorflow`
```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```
:end_tab:

## Summary

* We can launch and stop a Jupyter server through Amazon SageMaker to run this book.
* We can update notebooks via the terminal on the Amazon SageMaker instance.


## Exercises

1. Try to edit and run the code in this book using Amazon SageMaker.
1. Access the source code directory via the terminal.


[Discussions](https://discuss.d2l.ai/t/422)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIxMzIyNjE2ODZdfQ==
-->
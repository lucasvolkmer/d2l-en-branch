# Instalação
:label:`chap_installation`

Para prepara-lo a ter uma experiência prática de aprendizado,
precisamos configurar o ambiente para executar Python,
Jupyter notebooks, as bibliotecas relevantes,
e o código necessário para executar o livro em si.

## Instalando Miniconda

A maneira mais simples de começar será instalar
[Miniconda](https://conda.io/en/latest/miniconda.html). A versão Python 3.x
é necessária. Você pode pular as etapas a seguir se o conda já tiver sido instalado.
Baixe o arquivo Miniconda sh correspondente do site
e então execute a instalação a partir da linha de comando
usando  `sh <FILENAME> -b`. Para usuários do macOS:
```bash
# O nome do arquivo pode estar diferente
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```


Para os usuários de Linux:

```bash
# O nome do arquivo pode estar diferente
sh Miniconda3-latest-Linux-x86_64.sh -b
```

A seguir, inicialize o *shell* para que possamos executar `conda` diretamente.

```bash
~/miniconda3/bin/conda init
```


Agora feche e reabra seu *shell* atual. Você deve ser capaz de criar um novo ambiente da seguinte forma:

```bash
conda create --name d2l python=3.8 -y
```


## Baixando os *Notebooks D2L *

Next, we need to download the code of this book. You can click the "All
Notebooks" tab on the top of any HTML page to download and unzip the code.
Alternatively, if you have `unzip` (otherwise run `sudo apt install unzip`) available:

Em seguida, precisamos baixar o código deste livro. Você pode clicar no botão "All Notebooks" na parte superior de qualquer página HTML para baixar e descompactar o código.
Alternativamente, se você tiver `unzip` (caso contrário, execute` sudo apt install unzip`) disponível:

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```


Agora precisamos ativar o ambiente `d2l`.

```bash
conda activate d2l
```


## Instalando o *Framework* e o pacote `d2l` 

Antes de instalar o *Framework* de *Deep Learning*, primeiro verifique
se você tem ou não GPUs adequadas em sua máquina
(as GPUs que alimentam a tela em um laptop padrão
não contam para nossos propósitos).
Se você estiver instalando em um servidor GPU,
proceda para: ref: `subsec_gpu` para instruções
para instalar uma versão compatível com GPU.

Caso contrário, você pode instalar a versão da CPU da seguinte maneira. Isso será mais do que potência suficiente para você
pelos primeiros capítulos, mas você precisará acessar GPUs para executar modelos maiores.

:begin_tab:`mxnet`

```bash
pip install mxnet==1.7.0.post1
```


:end_tab:


:begin_tab:`pytorch`

```bash
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
```


:end_tab:

:begin_tab:`tensorflow`

Você pode instalar o TensorFlow com suporte para CPU e GPU da seguinte maneira:

```bash
pip install tensorflow tensorflow-probability
```


:end_tab:


We also install the `d2l` package that encapsulates frequently used
functions and classes in this book.
Nós também instalamos o pacote `d2l` que encapsula  funções e classes frequentemente usadas neste livro.
```bash
# -U: Atualiza todos os pacotes para as versões mais atuais disponíveis
pip install -U d2l
```

Após realizadas as instalações podemos abrir os *notebooks Jupyter* através do seguinte comando:

```bash
jupyter notebook
```

Nesse ponto, você pode abrir http://localhost:8888 (geralmente abre automaticamente) no navegador da web. Em seguida, podemos executar o código para cada seção do livro.
Sempre execute `conda activate d2l` para ativar o ambiente de execução
antes de executar o código do livro ou atualizar o *framework* de  *Deep Learning* ou o pacote `d2l`.
Para sair do ambiente, execute `conda deactivate`.


## Compatibilidade com GPU
:label:`subsec_gpu`

:begin_tab:`mxnet`
By default, MXNet is installed without GPU support
to ensure that it will run on any computer (including most laptops).
Part of this book requires or recommends running with GPU.
If your computer has NVIDIA graphics cards and has installed [CUDA](https://developer.nvidia.com/cuda-downloads),
then you should install a GPU-enabled version.
If you have installed the CPU-only version,
you may need to remove it first by running:

```bash
pip uninstall mxnet
```


Then we need to find the CUDA version you installed.
You may check it through `nvcc --version` or `cat /usr/local/cuda/version.txt`.
Assume that you have installed CUDA 10.1,
then you can install with the following command:

```bash
# For Windows users
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python

# For Linux and macOS users
pip install mxnet-cu101==1.7.0
```


You may change the last digits according to your CUDA version, e.g., `cu100` for
CUDA 10.0 and `cu90` for CUDA 9.0.
:end_tab:


:begin_tab:`pytorch,tensorflow`
By default, the deep learning framework is installed with GPU support.
If your computer has NVIDIA GPUs and has installed [CUDA](https://developer.nvidia.com/cuda-downloads),
then you are all set.
:end_tab:

## Exercises

1. Download the code for the book and install the runtime environment.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDQ3MTc5NzUxLC05NTYzNTExMzEsLTE3NT
UyMDU5MzldfQ==
-->
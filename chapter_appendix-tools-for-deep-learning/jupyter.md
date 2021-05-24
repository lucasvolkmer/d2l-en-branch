# Usando Jupyter
:label:`sec_jupyter`

Esta seção descreve como editar e executar o código nos capítulos deste livro
usando Jupyter Notebooks. Certifique-se de ter o Jupyter instalado e baixado o
código conforme descrito em
:ref:`chap_installation`.
Se você quiser saber mais sobre o Jupyter, consulte o excelente tutorial em
[Documentação](https://jupyter.readthedocs.io/en/latest/).


## Editando e executando o código localmente

Suponha que o caminho local do código do livro seja "xx/yy/d2l-en/". Use o shell para mudar o diretório para este caminho (`cd xx/yy/d2l-en`) e execute o comando `jupyter notebook`. Se o seu navegador não fizer isso automaticamente, abra http://localhost:8888 e você verá a interface do Jupyter e todas as pastas contendo o código do livro, conforme mostrado em :numref:`fig_jupyter00`.

![As pastas que contêm o código neste livro.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`


Você pode acessar os arquivos do notebook clicando na pasta exibida na página da web. Eles geralmente têm o sufixo ".ipynb".
Para fins de brevidade, criamos um arquivo temporário "test.ipynb". O conteúdo exibido após você clicar é mostrado em :numref:`fig_jupyter01`. Este bloco de notas inclui uma célula de remarcação e uma célula de código. O conteúdo da célula de redução inclui "Este é um título" e "Este é um texto". A célula de código contém duas linhas de código Python.

![Markdown e células de código no arquivo "text.ipynb".](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`


Clique duas vezes na célula de redução para entrar no modo de edição. Adicione uma nova string de texto "Olá, mundo". no final da célula, conforme mostrado em :numref:`fig_jupyter02`.

![Edite a célula de redução.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`

Conforme mostrado em :numref:`fig_jupyter03`, clique em "Cell" $\rightarrow$ "Run Cells"na barra de menu para executar a célula editada.

![Execute a celula.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`


Após a execução, a célula de redução é mostrada em :numref:`fig_jupyter04`.

![A célula de redução após a edição.](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`


Next, click on the code cell. Multiply the elements by 2 after the last line of code, as shown in :numref:`fig_jupyter05`.

![Edit the code cell.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`


You can also run the cell with a shortcut ("Ctrl + Enter" by default) and obtain the output result from :numref:`fig_jupyter06`.

![Run the code cell to obtain the output.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`


When a notebook contains more cells, we can click "Kernel" $\rightarrow$ "Restart & Run All" in the menu bar to run all the cells in the entire notebook. By clicking "Help" $\rightarrow$ "Edit Keyboard Shortcuts" in the menu bar, you can edit the shortcuts according to your preferences.


## Advanced Options

Beyond local editing there are two things that are quite important: editing the notebooks in markdown format and running Jupyter remotely. The latter matters when we want to run the code on a faster server. The former matters since Jupyter's native .ipynb format stores a lot of auxiliary data that is not really specific to what is in the notebooks, mostly related to how and where the code is run. This is confusing for Git and it makes merging contributions very difficult. Fortunately there is an alternative---native editing in Markdown.

### Markdown Files in Jupyter

If you wish to contribute to the content of this book, you need to modify the
source file (md file, not ipynb file) on GitHub. Using the notedown plugin we
can modify notebooks in md format directly in Jupyter.


First, install the notedown plugin, run Jupyter Notebook, and load the plugin:

```
pip install mu-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```


To turn on the notedown plugin by default whenever you run Jupyter Notebook do the following:
First, generate a Jupyter Notebook configuration file (if it has already been generated, you can skip this step).

```
jupyter notebook --generate-config
```


Then, add the following line to the end of the Jupyter Notebook configuration file (for Linux/macOS, usually in the path `~/.jupyter/jupyter_notebook_config.py`):

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```


After that, you only need to run the `jupyter notebook` command to turn on the notedown plugin by default.

### Running Jupyter Notebook on a Remote Server

Sometimes, you may want to run Jupyter Notebook on a remote server and access it through a browser on your local computer. If Linux or MacOS is installed on your local machine (Windows can also support this function through third-party software such as PuTTY), you can use port forwarding:

```
ssh myserver -L 8888:localhost:8888
```


The above is the address of the remote server `myserver`. Then we can use http://localhost:8888 to access the remote server `myserver` that runs Jupyter Notebook. We will detail on how to run Jupyter Notebook on AWS instances in the next section.

### Timing

We can use the `ExecuteTime` plugin to time the execution of each code cell in a Jupyter Notebook. Use the following commands to install the plugin:

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```


## Summary

* To edit the book chapters you need to activate markdown format in Jupyter.
* You can run servers remotely using port forwarding.


## Exercises

1. Try to edit and run the code in this book locally.
1. Try to edit and run the code in this book *remotely* via port forwarding.
1. Measure $\mathbf{A}^\top \mathbf{B}$ vs. $\mathbf{A} \mathbf{B}$ for two square matrices in $\mathbb{R}^{1024 \times 1024}$. Which one is faster?


[Discussions](https://discuss.d2l.ai/t/421)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTg2OTU1OTMzOV19
-->
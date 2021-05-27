# Contribuindo para este livro
:label:`sec_how_to_contribute`

Contribuições de [leitores](https://github.com/d2l-ai/d2l-en/graphs/contributors) nos ajudam a melhorar este livro. Se você encontrar um erro de digitação, um link desatualizado, algo onde você acha que perdemos uma citação, onde o código não parece elegante ou onde uma explicação não é clara, contribua de volta e ajude-nos a ajudar nossos leitores. Embora em livros normais o atraso entre as tiragens (e, portanto, entre as correções de digitação) possa ser medido em anos, normalmente leva horas ou dias para incorporar uma melhoria neste livro. Tudo isso é possível devido ao controle de versão e teste de integração contínua. Para fazer isso, você precisa enviar uma [solicitação de pull](https://github.com/d2l-ai/d2l-en/pulls) para o repositório GitHub. Quando sua solicitação pull for mesclada ao repositório de código pelo autor, você se tornará um contribuidor.

## Pequenas alterações de texto

As contribuições mais comuns são editar uma frase ou corrigir erros de digitação. Recomendamos que você encontre o arquivo de origem no [github repo](https://github.com/d2l-ai/d2l-en) e edite o arquivo diretamente. Por exemplo, você pode pesquisar o arquivo através do botão [Find file](https://github.com/d2l-ai/d2l-en/find/master) (:numref:`fig_edit_file`) para localizar o arquivo de origem, que é um arquivo de redução. Em seguida, você clica no botão "Editar este arquivo" no canto superior direito para fazer as alterações no arquivo de redução.

![Edite o arquivo no Github.](../img/edit-file.png)
:width:`300px`
:label:`fig_edit_file`

Depois de terminar, preencha as descrições das alterações no painel "Propor alteração do arquivo" na parte inferior da página e clique no botão "Propor alteração do arquivo". Ele irá redirecioná-lo para uma nova página para revisar suas alterações (:numref:`fig_git_createpr`). Se tudo estiver certo, você pode enviar uma solicitação de pull clicando no botão "Criar solicitação de pull".

## Propor uma mudança importante

Se você planeja atualizar uma grande parte do texto ou código, precisa saber um pouco mais sobre o formato que este livro está usando. O arquivo de origem é baseado no [formato markdown](https://daringfireball.net/projects/markdown/syntax) com um conjunto de extensões por meio do [d2lbook](http://book.d2l.ai/user/markdown .html) pacote, como referência a equações, imagens, capítulos e citações. Você pode usar qualquer editor do Markdown para abrir esses arquivos e fazer suas alterações.

Se você deseja alterar o código, recomendamos que você use o Jupyter para abrir esses arquivos Markdown conforme descrito em :numref:`sec_jupyter`. Para que você possa executar e testar suas alterações. Lembre-se de limpar todas as saídas antes de enviar suas alterações, nosso sistema de CI executará as seções que você atualizou para gerar saídas.

Algumas seções podem suportar múltiplas implementações de framework, você pode usar `d2lbook` para ativar um framework particular, então outras implementações de framework tornam-se blocos de código Markdown e não serão executados quando você "Executar Tudo "no Jupyter. Em outras palavras, primeiro instale `d2lbook` executando

```bash
pip install git+https://github.com/d2l-ai/d2l-book
```


Then in the root directory of `d2l-en`, you can activate a particular implementation by running one of the following commands:

```bash
d2lbook activate mxnet chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate pytorch chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate tensorflow chapter_multilayer-perceptrons/mlp-scratch.md
```


Before submitting your changes, please clear all code block outputs and activate all by

```bash
d2lbook activate all chapter_multilayer-perceptrons/mlp-scratch.md
```

If you add a new code block not for the default implementation, which is MXNet, please use `#@tab` to mark this block on the beginning line. For example, `#@tab pytorch` for a PyTorch code block, `#@tab tensorflow` for a TensorFlow code block, or `#@tab all` a shared code block for all implementations. You may refer to [d2lbook](http://book.d2l.ai/user/code_tabs.html) for more information.


## Adding a New Section or a New Framework Implementation

If you want to create a new chapter, e.g. reinforcement learning, or add implementations of new frameworks, such as TensorFlow, please contact the authors first, either by emailing or using [github issues](https://github.com/d2l-ai/d2l-en/issues).

## Submitting a Major Change

We suggest you to use the standard `git` process to submit a major change. In a nutshell the process works as described in :numref:`fig_contribute`.

![Contributing to the book.](../img/contribute.svg)
:label:`fig_contribute`

We will walk you through the steps in detail. If you are already familiar with Git you can skip this section. For concreteness we assume that the contributor's user name is "astonzhang".

### Installing Git

The Git open source book describes [how to install Git](https://git-scm.com/book/en/v2). This typically works via `apt install git` on Ubuntu Linux, by installing the Xcode developer tools on macOS, or by using GitHub's [desktop client](https://desktop.github.com). If you do not have a GitHub account, you need to sign up for one.

### Logging in to GitHub

Enter the [address](https://github.com/d2l-ai/d2l-en/) of the book's code repository in your browser. Click on the `Fork` button in the red box at the top-right of :numref:`fig_git_fork`, to make a copy of the repository of this book. This is now *your copy* and you can change it any way you want.

![The code repository page.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`


Now, the code repository of this book will be forked (i.e., copied) to your username, such as `astonzhang/d2l-en` shown at the top-left of the screenshot :numref:`fig_git_forked`.

![Fork the code repository.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### Cloning the Repository

To clone the repository (i.e., to make a local copy) we need to get its repository address. The green button in :numref:`fig_git_clone` displays this. Make sure that your local copy is up to date with the main repository if you decide to keep this fork around for longer. For now simply follow the instructions in :ref:`chap_installation` to get started. The main difference is that you are now downloading *your own fork* of the repository.

![Git clone.](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# Replace your_github_username with your GitHub username
git clone https://github.com/your_github_username/d2l-en.git
```


### Editing the Book and Push

Now it is time to edit the book. It is best to edit the notebooks in Jupyter following instructions in :numref:`sec_jupyter`. Make the changes and check that they are OK. Assume we have modified a typo in the file `~/d2l-en/chapter_appendix_tools/how-to-contribute.md`.
You can then check which files you have changed:

At this point Git will prompt that the `chapter_appendix_tools/how-to-contribute.md` file has been modified.

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix_tools/how-to-contribute.md
```


After confirming that this is what you want, execute the following command:

```
git add chapter_appendix_tools/how-to-contribute.md
git commit -m 'fix typo in git documentation'
git push
```


The changed code will then be in your personal fork of the repository. To request the addition of your change, you have to create a pull request for the official repository of the book.

### Pull Request

As shown in :numref:`fig_git_newpr`, go to your fork of the repository on GitHub and select "New pull request". This will open up a screen that shows you the changes between your edits and what is current in the main repository of the book.

![Pull Request.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`


### Submitting Pull Request

Finally, submit a pull request by clicking the button as shown in :numref:`fig_git_createpr`. Make sure to describe the changes you have made in the pull request. This will make it easier for the authors to review it and to merge it with the book. Depending on the changes, this might get accepted right away, rejected, or more likely, you will get some feedback on the changes. Once you have incorporated them, you are good to go.

![Create Pull Request.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`

Your pull request will appear among the list of requests in the main repository. We will make every effort to process it quickly.

## Summary

* You can use GitHub to contribute to this book.
* You can edit the file on GitHub directly for minor changes.
* For a major change, please fork the repository, edit things locally and only contribute back once you are ready.
* Pull requests are how contributions are being bundled up. Try not to submit huge pull requests since this makes them hard to understand and incorporate. Better send several smaller ones.


## Exercises

1. Star and fork the `d2l-en` repository.
1. Find some code that needs improvement and submit a pull request.
1. Find a reference that we missed and submit a pull request.
1. It is usually a better practice to create a pull request using a new branch. Learn how to do it with [Git branching](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell).

[Discussions](https://discuss.d2l.ai/t/426)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE4MDIzODE5MDksMjk1MjE5NzAzLDIxNz
g0OTM2NiwxMzY2MDU3MjQzXX0=
-->
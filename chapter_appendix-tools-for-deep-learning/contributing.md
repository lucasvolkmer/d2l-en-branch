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

Algumas seções podem suportar múltiplas implementações de framework, você pode usar `d2lbook` para ativar um framework particular, então outras implementações de framework tornam-se blocos de código Markdown e não serão executados quando você "Executar Tudo" no Jupyter. Em outras palavras, primeiro instale `d2lbook` executando

```bash
pip install git+https://github.com/d2l-ai/d2l-book
```


Então, no diretório raiz de `d2l-en`, você pode ativar uma implementação particular executando um dos seguintes comandos:

```bash
d2lbook activate mxnet chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate pytorch chapter_multilayer-perceptrons/mlp-scratch.md
d2lbook activate tensorflow chapter_multilayer-perceptrons/mlp-scratch.md
```


Antes de enviar suas alterações, limpe todas as saídas do bloco de código e ative todas por

```bash
d2lbook activate all chapter_multilayer-perceptrons/mlp-scratch.md
```

Se você adicionar um novo bloco de código não para a implementação padrão, que é MXNet, use `#@tab` para marcar este bloco na linha inicial. Por exemplo, `#@tab pytorch` para um bloco de código PyTorch, `#@tab tensorflow` para um bloco de código TensorFlow ou `#@tab all` um bloco de código compartilhado para todas as implementações. Você pode consultar [d2lbook](http://book.d2l.ai/user/code_tabs.html) para obter mais informações.


## Adicionando uma nova seção ou uma nova implementação de estrutura

Se você deseja criar um novo capítulo, por ex. aprendizado de reforço ou adicionar implementações de novas estruturas, como TensorFlow, entre em contato com os autores primeiro, por e-mail ou usando [questões do github](https://github.com/d2l-ai/d2l-en/issues).

## Enviando uma Mudança Principal

Sugerimos que você use o processo `git` padrão para enviar uma grande mudança. Em poucas palavras, o processo funciona conforme descrito em :numref:`fig_contribute`.

![Contribuindo para o livro.](../img/contribute.svg)
:label:`fig_contribute`

Iremos acompanhá-lo detalhadamente nas etapas. Se você já estiver familiarizado com o Git, pode pular esta seção. Para concretizar, assumimos que o nome de usuário do contribuidor é "astonzhang".

### Instalando Git

O livro de código aberto Git descreve [como instalar o Git](https://git-scm.com/book/en/v2). Isso normalmente funciona via `apt install git` no Ubuntu Linux, instalando as ferramentas de desenvolvedor Xcode no macOS ou usando o [cliente de desktop do GitHub ](https://desktop.github.com). Se você não tem uma conta GitHub, você precisa se inscrever para uma.

### Login no GitHub

Digite o [endereço](https://github.com/d2l-ai/d2l-en/) do repositório de código do livro em seu navegador. Clique no botão `Fork` na caixa vermelha no canto superior direito de :numref:`fig_git_fork`, para fazer uma cópia do repositório deste livro. Esta agora é *sua cópia* e você pode alterá-la da maneira que desejar.

![A página do repositório de código.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`


Agora, o repositório de código deste livro será bifurcado (ou seja, copiado) para seu nome de usuário, como `astonzhang/ d2l-en` mostrado no canto superior esquerdo da imagem :numref:`fig_git_forked`.

![Bifurque o repositório de código.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`

### Clonando o repositório

Para clonar o repositório (ou seja, para fazer uma cópia local), precisamos obter o endereço do repositório. O botão verde em :numref:`fig_git_clone` exibe isso. Certifique-se de que sua cópia local esteja atualizada com o repositório principal se você decidir manter esta bifurcação por mais tempo. Por enquanto, basta seguir as instruções em :ref:`chap_installation` para começar. A principal diferença é que agora você está baixando *seu próprio fork* do repositório.

![Clonando Git.](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# Replace your_github_username with your GitHub username
git clone https://github.com/your_github_username/d2l-en.git
```


### Editando o livro e empurrando

Agora é hora de editar o livro. É melhor editar os blocos de notas no Jupyter seguindo as instruções em :numref:`sec_jupyter`. Faça as alterações e verifique se estão corretas. Suponha que modificamos um erro de digitação no arquivo `~/d2l-en/chapter_appendix_tools/how-to-contribute.md`.
Você pode então verificar quais arquivos você alterou:

Neste ponto, o Git irá avisar que o arquivo `chapter_appendix_tools/how-to-contribute.md` foi modificado.

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix_tools/how-to-contribute.md
```


Depois de confirmar que é isso que você deseja, execute o seguinte comando:

```
git add chapter_appendix_tools/how-to-contribute.md
git commit -m 'fix typo in git documentation'
git push
```


O código alterado estará então em sua bifurcação pessoal do repositório. Para solicitar a adição de sua alteração, você deve criar uma solicitação pull para o repositório oficial do livro.

### Solicitação de pull

Conforme mostrado em :numref:`fig_git_newpr`, vá para o fork do repositório no GitHub e selecione "New pull request". Isso abrirá uma tela que mostra as mudanças entre suas edições e o que está em vigor no repositório principal do livro.

![Solicitação de Pull.](../img/git-newpr.png)
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
eyJoaXN0b3J5IjpbLTE0NjkzNTkzMjYsMjk1MjE5NzAzLDIxNz
g0OTM2NiwxMzY2MDU3MjQzXX0=
-->
# Review de code du projet: "umap-clustering-mise-en-prod" 


## Avant-propos :

Dans cette revue de code, je vais passer en revue l'application des bonnes pratiques du développement et proposer des pistes d'améliorations sur les différents points. Cependant, je tiens tout d'abord à dire que votre travail est de grande qualité et que les pistes d'améliorations ne sont que mineurs. L'ensemble du travail est très cohérent, complet et documenté ce qui permet de prendre en main le dépôt très rapidement. 


## Utilisation de Git :

Le projet possède bien un `.gitignore` adapté au langage et avec des règles additionnelles afin de respecter les bonnes pratiques de versioning. Il y a notamment bien une section "Environments" qui permet d'éviter d'avoir les fichiers `.venv` sur Github. De plus, le document avec les token secrets est bien ajouté au `.gitignore` ce qui permet de bien conserver le secret sur les token. Cependant, dans le dossier `tests`, il y a un document `iris.csv` qui pourrait être ajouté dans le `.gitignore` afin d'améliorer le projet et de ne pas conserver de csv sur GitHub. 

Le projet a bien été construit en utilisant des branches et en faisant des pull requests. Vers la fin du projet plusieurs commit d'affilés semblent avoir été fait directement sur la branche main. Ce travail aurait pu être effectué sur une autre branche afin d'éviter des conflits éventuels et de séparer le travail par tâche.


## Fichier README 

Le fichier `README` fait une très bonne explication du sujet du projet, du fonctionnement de l'algorithme implémenté ainsi que du type de base de données utilisées. Il fait également un bilan de ce qui est réalisé dans le projet et donne les liens de l'API et du front-end website implémentés. Il ajoute également des précisions sur ce qui peut être testé sur le front-end website ou non. 
Le fichier contient également des instructions séparées pour un utilisateur de l'API/du site et pour un développeur qui voudrait tester l'API de manière locale. Cette séparation est très claire et permet à tous les lecteurs de sélectionner rapidement les informations qui les concernent pour utiliser rapidement le travail réalisé. Le `README` évoque également la présence de fichiers `.md` dans un dossier `docs` permettant à tous les lecteurs d'avoir accès à des explications complètes des différentes implémentations du projet. Il présente également la structure du code. Finalement, des références sont évoqués permettant aux lecteurs de connaître les sources du projet. 

Ce fichier `README` est très complet et permet de comprendre très rapidement ce que contient le repot Github et comment utiliser les différentes implémentations. Les fichiers `.md` dans un dossier `docs` permettent réellement de bien comprendre les différentes implémentations. Pour améliorer le projet, je pense qu'il serait possible de citer dans le `README` un exemple de base de données réelles auquel UMAP peut être appliqué et ce que l'algorithme apporte dans cet exemple afin que le lecteur comprenne rapidement à quoi peut servir l'algorithme dans un besoin métier. Cette amélioration est en partie présente dans le streamlit mais je pense que l'ajouter au `README` permettrait à l'utilisateur de comprendre directement le but de l'algorithme et donc du dépot GitHub. De plus, je pense que la structure du code peut être présenté de manière un peu plus visuelle comme l'exemple si dessous d'une partie du dépot: 
```bash 
umap-clustering-mise-en-prod/
├── src/                #Code source
│   ├── adapter/
│   │   ├── mlflow_tracker.py
│   │   └── monitoring.py
│   └── umap_algo/
│       ├── knn.py
│       ├── nn_descent.py
│       └── umap_class.py
├── .gitignore          # Fichiers à exclure
└── README.md           # Présentation du projet
```
Finalement, je pense que le projet pourrait être améliorer en ajoutant dans le `README` une explication rapide de k-means and HDBSCAN puisque ces deux algorithmes sont utilisés dans le projet et qu'étant donnés que le `README` explique en quoi consiste UMAP, on peut considérer que l'utilisateur de ce dépot n'a pas forcément connaissance du fonctionnement de ces deux algorithmes. 


## Fichier License 

Le projet possède bien un fichier License qui déclare la license d'exploitation du projet. 


## Versioning des packages

Le projet possède un fichier `pyproject.toml` qui permet bien d'installer les packages nécessaires. 


## Qualité du code 

Le code possède beaucoup de commentaires. Il y a également des docstrings. Le code est écrit avec une programmation fonctionnelle. De plus, un dossier `config` comprend les hyperparamètres. 

Si on effectue des tests avec Ruff on obtient : "All checks passed !" pour le linter et "21 files already formatted" pour le formater. 

Si on effectue des tests avec Pylint, on a des scores proches de 10 pour le dossier `src`. Cependant, pour le dossier `tests`, on a des résultats plus bas :2.29/10 pour `test_mlflow_tracker.py`, 7.17/10 pour `test_knn.py`, 7.89 pour `test_api.py` et 7.89 pour `test_umap_class.py`. Ces scores sont notamment dû à l'absence de docstrings sur certaines fonctions et à des problèmes dans les noms des variables. Pour améliorer le projet, il serait possible de rajouter des docstrings là où il en manque et d'adapter les noms de variables. Voici des permalinks vers les fonctions qui n'ont pas de docstrings : 

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/conftest.py#L18

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/conftest.py#L23

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/conftest.py#L38

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/conftest.py#L51

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/iris_to_csv.py#L14

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/test_mlflow_tracker.py#L23

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/test_mlflow_tracker.py#L38

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/test_mlflow_tracker.py#L69

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/test_mlflow_tracker.py#L97


De plus, certains commentaires et certaines docstrings sont en français tandis que le reste du projet est en anglais. Voici des permalinks vers ces commentaires et docstrings en français : 

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L114

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L119

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L73

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L42

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L15

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L1

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/knn.py#L37

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/knn.py#L39

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/test_api.py#L12

https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/test_api.py#L65

Pour améliorer le projet, changer ces commentaires et les mettre tous en anglais permettrait de garder une seule langue sur tout le projet. 


## Structure des projets
Le projet possède bien une structure de type cookiecutter. Cependant, en regardant le projet sans le connaître il est difficile de savoir rapidement quel fichier est le "main". Pour améliorer le projet il serait possible de mettre un fichier `main` ou `train` dans le dossier principal afin d'avoir rapidement accès au code principal (je pense que celui-ci correspond à umap_class.py qui est dans le dossier src/umap_algo).


## Autres points que les bonnes pratiques du développement 

### Streamlit

Le streamlit est très complet tout en étant pas trop lourd car il compile rapidement. Il permet à l'utilisateur de se représenter rapidement comment fonctionne UMAP et se hyperparamètres. Je pense que pour le améliorer le projet il serait possible de changer les noms des paramètres dans la barre de gauche ou bien d'ajouter une explication rapide sous chaque hyperparamètre car si l'utilisateur ne connait pas les hyperparamètres de UMAP, il ne sait pas forcément ce qu'il fait varier en changeant les hyperparamètres. Par exemple, il serait possible de remplacer "n_components" par "dimension de l'espace d'arrivé", ou bien de l'écrire en dessous. 
 


## Conclusion 

Pour résumé, le projet respecte très bien les bonnes pratiques du développement. Le projet est très complet et d'une très grande qualité. Le `README` est très complet et il y a même d'autres documents explicatifs dans le code. Pour améliorer le projet, il est possible d'uniformiser la langue des commentaires, de rajouter des docstrings dans les documents du dossier `tests`, de rajouter quelques explications de knn et HDBSCAN et de modifier la présentation de la structure du code dans le `README`, de rajouter des explications des hyperparamètres de UMAP dans le streamlit et finalement d'ajouter le document `iris.csv` du dossier `tests` dans le `.gitignore`. Tout ces points d'améliorations ne sont cependant que mineurs face à la qualité du projet. 

# Review de code du projet: "umap-clustering-mise-en-prod" 

## Utilisation de Git :

Le projet possède bien un `.gitignore` adapté au langage et avec des règles additionnelles pour respecter les bonnes pratiques de versioning. Il y a notamment bien une section "Environments" qui permet d'éviter d'avoir les fichiers .venv sur Github. De plus, le document avec les secrets est bien ajouté au `.gitignore` ce qui permet de bien conserver le secret sur les token. Cependant, dans le dossier `tests`, il y a un document `iris.csv` qui pourrait être ajouté dans le `.gitignore` pour améliorer le projet. 

Le projet a bien été construit en utilisant des branches et en faisant des pull requests. Vers la fin du projet une partie du travail semble avoir été fait directement sur la branche main. Ce travail aurait pu être effectué sur une autre branche pour éviter des conflits éventuels et pour séparer le travail par tâche. 

## Fichier README 

Le fichier `README` fait une très bonne explication du sujet du projet, de fonctionnement de l'algorithme implémenté ainsi que du type de base de données utilisées. Il fait également un bilan de ce qui est réalisé dans le projet et donne les liens de l'API et du front-end website implémentés. Il ajoute également des précisions sur ce qui peut être tester sur le front-end website ou non. 
Le fichier contient également des instructions séparées pour un utilisateur de l'API/du site et pour un développeur qui voudrait tester l'API de manière locale. Cette séparation est très claire et permet à tous les lecteurs de sélectionner rapidement les informations qui les concernent pour utiliser rapidement le travail réalisé. Le `README` évoque également la présence de fichiers `.md` dans un dossier `docs` permettant à tous les lecteurs d'avoir accès à des explications complètes des différentes implémentations du projet. Il présente également la structure du code. Finalement, des références sont évoqués permettant aux lecteurs de connaître les sources du projet. 

Ce fichier `README` est très complet et permet de comprendre très rapidement ce que contient le repot Github et comment utiliser les différentes implémentations. Pour améliorer le projet, je pense qu'il serait possible de citer dans le `README` un exemple de base de données auquel UMAP peut être appliqué et ce que ça change afin que le lecteur comprenne rapidement à quoi peut servir l'algorithme dans un besoin métier. De plus, je pense que la structure du code peut être présenté de manière un peu plus visuelle : 
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

## Fichier License 

Le projet possède bien un fichier License qui déclare la license d'exploitation du projet. 

## Versioning des packages

Le projet possède un fichier `pyproject.toml` qui permet d'installer les packages nécessaires. 

## Qualité du code 

Le code possède beaucoup de commentaires. Il y a également des docstrings. Le code est écrit avec une programmation fonctionnelle. De plus, un dossier `config` comprend les hyperparamètres. 

Si on effectue des tests avec Ruff on obtient : "All checks passed !" pour le linter et "21 files already formatted" pour le formateur. 

Si on effectue des tests avec Pylint, on a des scores proches de 10 pour le dossier `src`. Par contre, pour le dossier `tests`, on a des résultats plus bas :2.29/10 pour `test_mlflow_tracker.py`, 7.17/10 pour `test_knn.py`, 7.89 pour `test_api.py` et 7.89 pour `test_umap_class.py`. Ces scores sont notamment dû à l'absence de docstrings sur certaines fonctions. Pour améliorer le projet, il serait possible de rajouter des docstrings là où il en manque. 

Certains commentaires et certaines docstrings sont en français tandis que le reste du projet est en anglais. Voici des permalinks vers ces commentaires et docstrings en français : 

https://github.com/RebeccaBle/fork-umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L114

https://github.com/RebeccaBle/fork-umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L119

https://github.com/RebeccaBle/fork-umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L73

https://github.com/RebeccaBle/fork-umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L42

https://github.com/RebeccaBle/fork-umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L15

https://github.com/RebeccaBle/fork-umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L1

https://github.com/RebeccaBle/fork-umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/knn.py#L37

https://github.com/RebeccaBle/fork-umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/knn.py#L39

https://github.com/RebeccaBle/fork-umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/test_api.py#L65

https://github.com/RebeccaBle/fork-umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/tests/test_knn.py#L8

Pour améliorer le projet, changer ces commentaires et les mettre tous en anglais permettrait de garder une seule langue sur tout le projet. 

## Structure des projets
Le projet possède bien une structure de type cookiecutter. Cependant, en regardant le projet sans le connaître il est difficile de savoir rapidement quel fichier est le "main". Pour améliorer le projet il serait possible de mettre un fichier `main` ou `train` dans le dossier principal afin d'avoir rapidement accès au code principal. 

## Conclusion 
Le projet respecte très bien les bonnes pratiques du développement. Pour améliorer le projet, il est possible d'uniformiser la langue des commentaires et de rajouter des docstrings dans le document `test_umap_class.py`. 

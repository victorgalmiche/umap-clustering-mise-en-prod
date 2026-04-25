# Review de code du projet: "umap-clustering-mise-en-prod" 

## Utilisation de Git :

Le projet possède bien un `.gitignore` adapté au langage et avec des règles additionnelles pour respecter les bonnes pratiques de versioning. Il y a notamment bien une section "Environments" qui permet d'éviter d'avoir les fichiers .venv sur Github.

Le projet a bien été construit en utilisant des branches et en faisant des pull requests. Vers la fin du projet une partie du travail semble avoir été fait directement sur la branche main. Ce travail aurait pu être effectué sur une autre branche pour éviter des conflits éventuels et pour séparer le travail par tâche. 

## Fichier README 

Le fichier `README` fait une très bonne explication du sujet du projet, de fonctionnement de l'algorithme implémenté ainsi que du type de base de données utilisées. Il fait également un bilan de ce qui est réalisé dans le projet et donne les liens de l'API et du front-end website implémentés. Il ajoute également des précisions sur ce qui peut être tester sur le front-end website ou non. 
Le fichier contient également des instructions séparées pour un utilisateur de l'API/du site et pour un développeur qui voudrait tester l'API de manière locale. Cette séparation est très claire et permet à tous les lecteurs de sélectionner rapidement les informations qui les concernent pour utiliser rapidement le travail réalisé. Le `README` évoque également la présence de fichiers `.md` dans un dossier `docs` permettant à tous les lecteurs d'avoir accès à des explications complètes des différentes implémentations du projet. Il présente également la structure du code. Finalement, des références sont évoqués permettant aux lecteurs de connaître les sources du projet. 

Ce fichier `README` est très complet et permet de comprendre très rapidement ce que contient le repot Github et comment utiliser les différentes implémentations. Pour améliorer le projet, je pense qu'il serait possible de citer dans le `README` un exemple de base de données auquel UMAP peut être appliqué et ce que ça change afin que le lecteur comprenne rapidement à quoi peut servir l'algorithme dans un besoin métier. De plus, je pense que la structure du code peut être présenté de manière un peu plus visuelle : 

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

## Fichier License 

Le projet possède bien un fichier License qui déclare la license d'exploitation du projet. 

## Versioning des packages

Le projet possède un fichier `pyproject.toml` qui permet d'installer les packages nécessaires. 

## Qualité du code 

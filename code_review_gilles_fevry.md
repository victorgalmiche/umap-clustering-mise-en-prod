# Revue de code du projet "umap-clustering-mise-en-prod" par Gilles Févry

Bonjour à tous !

Tout d'abord, c'est une super idée de projet. Beaucoup de personnes doivent utiliser UMAP à un moment donné, mais n'ont pas nécessairement le temps de se plonger dans la documentation et les détails de l'implémentation. Votre API et l'application Streamlit sont très pratiques pour cela !
Au global, le code est bien organisé et la mise en production est maîtrisée. Toutefois (et c'est ce qui nous est demandé), je dois souligner quelques petits défauts.

## Checklists

### Bonnes pratiques du développement :

* [ ] .gitignore adapté : parfait et standard.
* [ ] Travail collaboratif (branches + PRs) : très bon. L'ensemble des étudiants a participé, les branches ont été utilisées à bon escient et les noms des commits sont clairs.
* [ ] README clair : oui, si ce n'est que la section "Using the API directly" contient deux parties "Projection" qui peuvent porter à confusion. De plus, la structure du repo pourrait être plus détaillée.
* [ ] LICENSE : oui.
* [ ] Versioning des packages : oui, avec pyproject.toml (dépendances bien marquées) et uv.lock.
* [ ] Linter/formater : oui, ruff et pylint configurés dans le .toml avec auto-formatage des commits
* [ ] Modularité du code : oui, le code est bien découpé et bien rangé dans les dossiers.
* [ ] Structure type cookiecutter : oui.
* [ ] Modularité du projet : tout est bien séparé.

### Bonnes pratiques MLOps :

* [ ] Développer un modèle de ML qui répond à un besoin métier : oui, comme dit précédemment !
* [ ] Entraîner le modèle via validation croisée, avec une procédure de fine-tuning des hyperparamètres : moyennement. On peut bien choisir les hyperparamètres, mais il n'y a pas de validation croisée ni de grid search. C'est relativement logique étant donné le projet, mais cela pourrait apporter des informations sur la sensibilité du modèle et donc sur la confiance dans les résultats !
* [ ] Formaliser le processus de fine-tuning de manière reproductible via MLflow : idem.
* [ ] Construire une API avec FastAPI pour exposer le meilleur modèle : très bon développement de l'API, assez poussé, notamment avec les clés pour accéder aux modèles entraînés.
* [ ] Créer une image Docker pour mettre à disposition l’API : oui, bien structurée et plutôt légère.
* [ ] Déployer l’API sur le SSP Cloud : oui, cela fonctionne très bien.
* [ ] Industrialiser le déploiement en mode GitOps avec ArgoCD : oui, et la documentation est là aussi plutôt bonne.
* [ ] Gérer le monitoring de l’application (logs, dashboard de suivi des performances, etc.) : oui, tout est bien sauvegardé avec MLflow.

## Bonus Streamlit :

L'application fonctionne bien, est utile et tourne plutôt vite ! En revanche, si son but est de s'adresser à des non-initiés, elle n'est pas nécessairement très intuitive : les noms des paramètres peuvent être obscurs au premier abord et il n'y a pas d'indication sur la manière de choisir les hyperparamètres. Je pense que pour une application plus user-friendly, il faudrait éventuellement ajouter une page présentant les différents paramètres.

## Qualité du code

Le code est globalement de bonne qualité, et c'est très bien d'avoir inclus des tests. En revanche, certains aspects sont à améliorer :

* Le code des tests obtient de mauvais scores avec Pylint, notamment en raison du manque de docstrings.
* Certaines fonctions dans src n'ont pas non plus de docstrings, comme :
  https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/umap_class.py#L190
* Ou bien des docstrings trop brèves qui ne mentionnent ni les inputs ni les outputs :
  https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/umap_class.py#L137
  https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/umap_class.py#L150
* Enfin, il y a certains caractères non standards dans le code, comme un emoji feu :
  https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/umap_class.py#L383
* Il y a beaucoup de mélanges anglais-français : souvent les docstrings sont en anglais et les commentaires en français, comme :
  https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/umap_class.py#L394
* Ou encore au sein d'une même docstring :
  https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/src/umap_algo/nn_descent.py#L1

## Autres points forts

* Vous avez développé une API et une application Streamlit !
* La documentation supplémentaire sur l'utilisation de l'API est très pratique.
* Vous faites des renvois aux papiers originaux.
* L'aspect mise en production et déploiement est, au global, très poussé !
* C'est très bien d'avoir mis des limites dans l'utilisation de l'API.
* Implémentation de UMAP depuis zéro.

## Autres points faibles

* On manque parfois d'informations sur les différentes méthodes utilisées (par exemple : quelles différences entre les méthodes de clustering et quelles implications ?).
* Il n'y a pas de description du .toml, c'est encore celle par défaut :
  https://github.com/victorgalmiche/umap-clustering-mise-en-prod/blob/7ee6c5fbfaa9212b54f2600ef995950602eabe4a/pyproject.toml#L4

## Conclusion

Je trouve votre projet très intéressant, et on voit que vous avez exploré les différents aspects de la mise en production présentés dans le cours. La partie déploiement est très solide, mais la partie expérimentation ML pourrait être davantage formalisée. Le principal axe d'amélioration reste, selon moi, la qualité du code Python : consistance des docstrings, suppression du français, et style de commentaires plus homogène. Enfin rendre l'application encore plus user-friendly en décrivant les spécificités des modèles de clustering, renommant les hyper-paramètres et en donnant un peu plus de background sur UMAP serait super. 

En tout cas, profitez bien de l'été et bons stages !

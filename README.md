# Prédiction du cours de l'action Apple (AAPL) avec un LSTM

Projet de prévision de séries temporelles financières : un réseau de neurones récurrent de type **LSTM**, implémenté avec **PyTorch**, apprend à prédire le prix de clôture d'une action à partir des 29 jours précédents.

Les données sont téléchargées automatiquement depuis Yahoo Finance via `yfinance`.

## Aperçu

| Élément | Détail |
|---|---|
| Action étudiée | AAPL (modifiable via la variable `ticker`) |
| Période | depuis le 2020-01-01 |
| Variable prédite | Prix de clôture (`Close`) |
| Modèle | LSTM à 2 couches, 32 unités cachées, couche linéaire finale |
| Fenêtre temporelle | 30 jours (29 en entrée, le 30e est la cible) |
| Découpage | 80 % entraînement / 20 % test (ordre chronologique) |
| Métrique | RMSE |

## Structure du projet

```
.
├── main.ipynb     # Notebook complet : données, modèle, entraînement, évaluation
└── README.md
```

## Installation

Python 3.9+ est recommandé.

```bash
pip install yfinance numpy pandas matplotlib scikit-learn torch jupyter
```

> Le notebook utilise `root_mean_squared_error` de scikit-learn, disponible à partir de la version 1.4.

Un GPU n'est pas obligatoire : le notebook utilise CUDA s'il est disponible, sinon le CPU.

## Utilisation

```bash
jupyter notebook main.ipynb
```

Puis exécuter les cellules dans l'ordre. Pour changer d'action, modifier :

```python
ticker = 'AAPL'   # ex. 'MSFT', 'TSLA', 'GOOGL'
```

## Méthodologie

1. **Collecte** : téléchargement de l'historique des prix avec `yfinance`.
2. **Normalisation** : `StandardScaler` appliqué au prix de clôture.
3. **Création des séquences** : fenêtres glissantes de 30 jours ; les 29 premiers jours servent d'entrée (`X`), le dernier de cible (`y`).
4. **Découpage** : 80 % des séquences pour l'entraînement, 20 % pour le test, sans mélange afin de respecter l'ordre du temps.
5. **Modèle** (`PredictionModel`) : LSTM (`input_dim=1`, `hidden_size=32`, `num_layers=2`) suivi d'une couche `Linear` qui produit la prédiction à partir du dernier pas de temps.
6. **Entraînement** : perte `MSELoss`, optimiseur `Adam` (`lr=0.01`), 200 époques.
7. **Évaluation** : retour aux prix réels avec `inverse_transform`, calcul de la RMSE, puis visualisation.

## Hyperparamètres

| Paramètre | Valeur |
|---|---|
| `seq_length` | 30 |
| `hidden_size` | 32 |
| `num_layers` | 2 |
| `learning_rate` | 0.01 |
| `num_epochs` | 200 |
| Perte | MSE |
| Optimiseur | Adam |

## Résultats

| Jeu | RMSE (en $) |
|---|---|
| Entraînement | ≈ 2,89 |
| Test | ≈ 7,88 |

Le notebook produit deux graphiques : le prix réel comparé au prix prédit sur le jeu de test, et les résidus (erreur de prédiction) avec la RMSE en repère.

## Limites et pistes d'amélioration

- Le `StandardScaler` est ajusté sur toutes les données, y compris la période de test, ce qui introduit une légère fuite d'information. Il vaudrait mieux l'ajuster uniquement sur le jeu d'entraînement.
- Le modèle n'utilise que le prix de clôture. On peut ajouter d'autres variables : volume, ouverture, plus haut/plus bas, indicateurs techniques (RSI, MACD, moyennes mobiles).
- L'entraînement se fait en un seul lot (sans mini-batchs) ; l'usage d'un `DataLoader` serait plus adapté à de plus gros volumes.
- Les résultats peuvent varier d'une exécution à l'autre ; fixer les graines aléatoires améliore la reproductibilité.
- Prédire le cours à 1 jour sur des données historiques ne garantit pas de bonnes performances en conditions réelles.

## Avertissement

Ce projet est à but **éducatif**. Il ne constitue pas un conseil en investissement.

## Technologies

Python · PyTorch · NumPy · pandas · scikit-learn · matplotlib · yfinance

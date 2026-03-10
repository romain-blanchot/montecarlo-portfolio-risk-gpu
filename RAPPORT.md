# Rapport de Projet — Moteur Monte Carlo GPU  
## Simulation de risque de portefeuille financier avec accélération CUDA

---

## 1. Introduction et contexte du projet

L'objectif de ce projet est de prendre un algorithme séquentiel (exécuté sur CPU) et de l'accélérer grâce à CUDA via Numba en Python. Pour illustrer cette démarche, nous avons choisi un cas d'usage concret issu de la gestion des risques financiers : **simuler le comportement futur d'un portefeuille de 8 actifs boursiers** et en extraire des métriques de risque standard utilisées dans l'industrie.

Ce choix est motivé par deux raisons : d'abord, le problème se prête naturellement à une parallélisation massive (chaque simulation est indépendante des autres), ce qui en fait un candidat idéal pour le GPU. Ensuite, c'est un problème réel — les banques et fonds d'investissement font exactement ce genre de calcul, parfois des millions de fois par jour.

Le code produit comprend :
- un **moteur CPU de référence** (NumPy vectorisé),
- un **kernel CUDA personnalisé** (`_gbm_kernel`) écrit avec Numba,
- une **suite de tests** complète,
- un **notebook interactif** et un **script de démonstration**.

---

## 2. Le problème : qu'est-ce qu'on simule, et pourquoi ?

### 2.1 Un portefeuille financier, c'est quoi ?

Un portefeuille financier, c'est simplement un ensemble d'actifs (ici des actions et des obligations) dans lesquels on a investi une certaine proportion de son capital. Dans notre cas, on modélise un portefeuille de **8 actifs** représentatifs de différents secteurs de l'économie américaine :

| Actif | Secteur             | Prix initial | Poids dans le portefeuille |
|-------|---------------------|-------------:|---------------------------:|
| AAPL  | Technologie         |      $195.00 |                       18 % |
| MSFT  | Technologie         |      $415.00 |                       17 % |
| JPM   | Finance             |      $205.00 |                       12 % |
| GS    | Finance             |      $510.00 |                       10 % |
| XOM   | Énergie             |      $118.00 |                       10 % |
| JNJ   | Santé               |      $155.00 |                       10 % |
| GLD   | Or (matières 1ères) |      $215.00 |                       13 % |
| BND   | Obligations         |       $73.00 |                       10 % |

La **valeur initiale du portefeuille** est la somme pondérée des prix : $243.80 par "unité".

Cette diversification n'est pas anodine — on voit que l'or et les obligations ont tendance à monter quand les marchés actions chutent (corrélation négative). C'est exactement ce qu'on cherche à modéliser.

### 2.2 Pourquoi simuler ?

Le futur des marchés financiers est fondamentalement incertain. On ne peut pas prévoir si une action va monter ou baisser demain. En revanche, on peut **modéliser cette incertitude statistiquement** : à partir de données historiques, on calibre un modèle probabiliste et on simule un grand nombre de scénarios futurs possibles.

À partir de ces simulations, on peut répondre à des questions comme :
- *Dans le pire des cas (sur 95 % des scénarios), combien peut-on perdre dans un an ?*
- *Si les marchés entrent en crise et que tout chute en même temps, que devient le portefeuille ?*

C'est là qu'interviennent la **VaR** et l'**Expected Shortfall**.

### 2.3 Les métriques de risque : VaR et Expected Shortfall

#### Value at Risk (VaR)

La VaR à 95 % est le seuil de perte tel que, dans 95 % des scénarios simulés, la perte réelle sera **inférieure** à ce seuil. Dit autrement : il y a seulement 5 % de chances de perdre plus que la VaR.

> Exemple : VaR 95 % = \$31.67 signifie que dans 95 % des simulations, le portefeuille ne perd pas plus de $31.67 sur l'année. Les 5 % restants représentent les scénarios catastrophe.

#### Expected Shortfall (ES), aussi appelé CVaR

L'ES répond à une question complémentaire : *parmi les 5 % de scénarios catastrophe (ceux qui dépassent la VaR), quelle est la perte moyenne ?* C'est une mesure plus prudente et plus informative que la VaR seule.

> Exemple : ES 95 % = \$45.10 signifie que dans les pires 5 % des cas, la perte moyenne est de $45.10.

Ces deux métriques ensemble donnent une image complète du risque de queue (*tail risk*) d'un portefeuille.

---

## 3. La méthode Monte Carlo

### 3.1 Le principe général

La méthode Monte Carlo tire son nom du célèbre casino de Monaco — non pas parce qu'elle est hasardeuse, mais parce qu'elle repose sur **le hasard comme outil de calcul**.

L'idée est simple : si on veut connaître la distribution des pertes possibles d'un portefeuille sur un an, on simule des milliers (ou millions) de trajectoires de prix plausibles. Chaque trajectoire est un "scénario" différent pour l'évolution des marchés. On calcule ensuite la perte ou le gain sur chacun de ces scénarios, et on analyse la distribution obtenue.

```
Scénario 1 : AAPL monte 15%, MSFT chute 5%, ... → perte = -$12.40  (gain)
Scénario 2 : crise, tout chute → perte = +$68.30
Scénario 3 : marché stable → perte = -$38.10  (gain)
...
Scénario 1 000 000 : ...
```

On agrège ensuite ces 1 000 000 de valeurs pour calculer VaR et ES.

### 3.2 Pourquoi autant de simulations ?

Plus on simule de trajectoires, plus on "couvre" l'espace des possibles et plus nos estimations de VaR et ES sont précises. Avec 10 000 chemins, la VaR 95 % fluctue légèrement d'une exécution à l'autre. Avec 1 000 000 de chemins, elle converge vers une valeur stable. C'est la **loi des grands nombres** appliquée à la gestion des risques.

---

## 4. Le modèle mathématique : le Mouvement Brownien Géométrique (GBM)

### 4.1 Comment modéliser le prix d'une action ?

Le modèle standard en finance pour modéliser l'évolution d'un prix est le **Mouvement Brownien Géométrique (GBM)**. L'idée intuitive : le prix d'une action a tendance à croître dans le temps (une dérive *drift* positive), mais avec une variabilité aléatoire (la *volatilité*).

En pratique, à chaque pas de temps discret, le prix est mis à jour selon :

```
S(t+dt) = S(t) * exp( (µ - σ²/2) * dt  +  σ * √dt * Z )
```

où :
- `S(t)` est le prix à l'instant `t`
- `µ` (mu) est la dérive annualisée — la tendance de croissance de l'actif
- `σ` (sigma) est la volatilité annualisée — l'amplitude des fluctuations
- `dt = 1/252` est un pas de temps d'un jour de bourse
- `Z ~ N(0,1)` est un tirage aléatoire gaussien

On répète cette mise à jour **252 fois** (un an de trading) pour obtenir une trajectoire complète.

### 4.2 Modéliser les corrélations entre actifs

Un point crucial : les actifs ne bougent pas indépendamment. Pendant la crise de 2008 par exemple, presque toutes les actions ont chuté en même temps. À l'inverse, l'or et les obligations montent souvent quand les actions baissent.

Pour modéliser ces dépendances, on utilise une **matrice de corrélation** — un tableau `nb_actif`×`nb_actif` où chaque case indique à quel point deux actifs bougent ensemble (1 = toujours dans le même sens, -1 = toujours en sens opposé).

Techniquement, on ne peut pas simplement multiplier des nombres aléatoires par des coefficients pour introduire les corrélations — il faut utiliser la **décomposition de Cholesky**. Elle décompose la matrice de corrélation en un produit `L × Lᵀ`, puis on transforme des vecteurs gaussiens indépendants `Z_indep` en vecteurs corrélés `Z_corr = L × Z_indep`. C'est le seul endroit du projet où l'algèbre linéaire joue un rôle central.

```python
# côté CPU (NumPy)
chol = np.linalg.cholesky(corr_matrix)     # calcul de L
Z_corr = Z_indep @ chol.T                  # introduction des corrélations
```

### 4.3 Le scénario de stress

En plus du scénario normal, on simule un **scénario de crise** : une matrice de corrélation "stress" où toutes les corrélations entre actions explosent (jusqu'à 0.95 entre AAPL et MSFT), tandis que l'or et les obligations se désolidarisent encore plus des marchés actions. Ce genre de test s'appelle un *stress test* et sert à évaluer la résilience d'un portefeuille en situation extrême.

---

## 5. Architecture du projet

Le code est organisé selon une architecture **hexagonale**, qui sépare clairement la logique métier des détails techniques :

```
src/portfolio_risk_engine/
├── domain/              # Logique pure (pas de dépendances externes)
│   ├── portfolio.py         → dataclass Portfolio (S0, weights, V0)
│   ├── market_model.py      → dataclass MarketModel (mu, sigma, dt, n_steps)
│   ├── correlation.py       → compute_cholesky()
│   ├── var.py               → compute_var()
│   └── expected_shortfall.py→ compute_es()
│
├── application/         # Orchestration
│   ├── run_simulation.py    → fonction run() complète
│   └── compare_engines.py  → benchmark CPU vs GPU
│
└── infrastructure/      # Adaptateurs techniques
    └── simulation/
        ├── base.py              → classe abstraite SimulationEngine
        ├── monte_carlo_cpu.py   → MonteCarloCPU (NumPy)
        └── monte_carlo_gpu.py   → MonteCarloGPU (kernel CUDA Numba)
```

Cette séparation a plusieurs avantages : le domaine (`domain/`) ne sait rien de GPU ou NumPy, les tests sont faciles à écrire par couche, et les deux moteurs (CPU/GPU) s'utilisent de façon identique grâce à l'interface commune `SimulationEngine`.

---

## 6. Implémentation CPU — la référence séquentielle

Le moteur CPU (`MonteCarloCPU`) utilise NumPy et est **vectorisé sur les chemins** : au lieu de simuler un chemin après l'autre, on traite tous les `n_paths` chemins simultanément à chaque pas de temps.

```python
S = np.tile(portfolio.S0, (n_paths, 1))          # (n_paths, n_assets)

for _ in range(n_steps):                          # 252 itérations
    Z_indep = rng.standard_normal((n_paths, n_assets))
    Z_corr  = Z_indep @ chol.T                    # corrélation via Cholesky
    S = S * np.exp(drift + diffusion_scale * Z_corr)

losses = V0 - (S @ weights)                       # (n_paths,)
```

C'est l'implémentation la plus naturelle et elle sert de **référence correcte**. Elle est performante grâce à la vectorisation NumPy, mais fondamentalement séquentielle : les 252 pas de temps se succèdent les uns après les autres, et NumPy utilise peu de cœurs CPU en parallèle pour les opérations matricielles.

---

## 7. Implémentation GPU — le kernel CUDA

### 7.1 Principe de parallélisation

L'idée clé est la suivante : **chaque chemin Monte Carlo est totalement indépendant des autres**. Le chemin n°42 n'a aucun besoin de connaître le résultat du chemin n°41. On peut donc lancer 1 000 000 chemins parfaitement en parallèle.

Sur GPU, on assigne **un thread CUDA à chaque chemin**. Avec une `RTX 4070 Ti` (testée localement) qui dispose de plusieurs milliers de cœurs CUDA, on peut exécuter des milliers de chemins véritablement simultanément.

### 7.2 Le kernel `_gbm_kernel`

Le kernel est écrit directement en Python avec le décorateur `@cuda.jit` de Numba, qui compile le code Python en PTX (le langage assembleur des GPU NVIDIA).

```python
@cuda.jit
def _gbm_kernel(rng_states, S0, weights, drift, diff_scale,
                n_steps, chol, n_assets, losses):
    tid = cuda.grid(1)          # identifiant global du thread = numéro du chemin
    if tid >= losses.shape[0]:
        return

    # chaque thread simule UN chemin complet de 252 pas
    S = cuda.local.array(32, dtype=float64)      # registres locaux
    for _ in range(n_steps):
        for i in range(n_assets):
            Z_indep[i] = xoroshiro128p_normal_float64(rng_states, tid)
        # corrélation via Cholesky (boucle triangulaire)
        for i in range(n_assets):
            acc = 0.0
            for j in range(i + 1):
                acc += s_chol[i, j] * Z_indep[j]
            Z_corr[i] = acc
        # mise à jour GBM
        for i in range(n_assets):
            S[i] *= math.exp(s_drift[i] + s_diff_scale[i] * Z_corr[i])

    losses[tid] = V0 - weighted_sum(S)           # écriture en mémoire globale
```

### 7.3 Optimisations CUDA implémentées

Plusieurs techniques d'optimisation GPU sont utilisées dans le kernel :

#### Mémoire partagée (*shared memory*)

La matrice de Cholesky (8×8 = 64 valeurs) et les vecteurs drift/sigma sont des constantes lues à chaque pas de temps par tous les threads. Au lieu de laisser chaque thread relire ces valeurs depuis la mémoire globale (lente, ~400–800 cycles de latence), on les charge **une seule fois en mémoire partagée** au début du kernel. Tous les threads d'un même bloc collaborent pour charger ces données, puis s'y réfèrent depuis la mémoire on-chip (~4 cycles de latence).

```python
# chargement coopératif (chaque thread charge un morceau)
i = local_tid
while i < n_assets:
    s_chol[i, j] = chol[i, j]
    i += block_size
cuda.syncthreads()
```

#### Variables locales dans les registres

Les tableaux de prix courants `S`, `Z_indep`, `Z_corr` (8 valeurs chacun) sont déclarés avec `cuda.local.array` — le compilateur les place dans les **registres du GPU**, la mémoire la plus rapide disponible. Aucun accès mémoire n'est nécessaire pour les calculs intermédiaires à chaque pas de temps.

#### Écriture coalisée (*coalesced write*)

En fin de kernel, chaque thread écrit son résultat dans `losses[tid]`. Comme les threads consécutifs d'un warp ont des `tid` consécutifs, leurs écritures en mémoire globale tombent sur des adresses consécutives — le GPU peut alors les fusionner en une seule transaction mémoire large et efficace. C'est la **coalescence mémoire**, une optimisation fondamentale pour les GPU.

#### Générateur aléatoire par thread (xoroshiro128p)

Chaque thread dispose de son propre état de générateur pseudo-aléatoire `xoroshiro128p`, initialisé avec un seed différent. Cela garantit l'**indépendance statistique** entre chemins tout en évitant toute synchronisation entre threads. C'est l'approche standard recommandée par Numba pour la génération de nombres aléatoires sur GPU.

#### Calcul du drift pré-calculé côté CPU

Le terme `(µ - σ²/2) × dt` et `σ × √dt` sont des constantes calculées **une seule fois sur le CPU** avant le lancement du kernel. Cela évite de refaire ces divisions et racines carrées dans chaque thread à chaque pas de temps (252 × n_assets = 2016 opérations économisées par chemin).

#### Dimensionnement de la grille

La grille est dimensionnée avec 256 threads par bloc (`_TPB = 256`, soit 8 warps), et autant de blocs que nécessaire pour couvrir tous les chemins :

```python
threads = 256
blocks  = (n_paths + threads - 1) // threads   # arrondi au supérieur
```

256 est un compromis classique : assez grand pour masquer la latence mémoire par du calcul (*latency hiding*), assez petit pour maximiser l'occupation du SM (*Streaming Multiprocessor*).

---

## 8. Résultats — comparaison CPU vs GPU

### 8.1 Environnement matériel et logiciel

| Composant          | Valeur                     |
|--------------------|----------------------------|
| GPU                | NVIDIA GeForce RTX 4070 Ti |
| Compute Capability | 8.9 (Ada Lovelace)         |
| CPU                | AMD64 (Ryzen série 7000)   |
| Python             | 3.13.12 (Anaconda)         |
| Numba              | 0.64.0                     |
| NumPy              | 2.4.2                      |
| OS                 | Windows 11                 |

### 8.2 Tableau de performances

Le temps GPU exclut la compilation JIT initiale du kernel (~3–5 s, amortie sur la durée de vie du processus). Les mesures CPU et GPU sont faites sur le même portefeuille, le même seed, les mêmes paramètres.

| Nombre de chemins | Temps CPU  | Temps GPU  | Accélération |
|------------------:|:----------:|:----------:|:------------:|
|            10 000 |  0.379 s   |  0.011 s   |   **×35**    |
|            50 000 |  2.174 s   |  0.037 s   |   **×59**    |
|           100 000 |  4.522 s   |  0.065 s   |   **×70**    |
|           200 000 |  9.303 s   |  0.126 s   |   **×74**    |
|         1 000 000 |   ~46 s    |  0.563 s   |   **×82**    |

### 8.3 Métriques de risque obtenues (1 000 000 chemins, seed = 42)

| Métrique                          | Scénario normal | Scénario de stress |
|-----------------------------------|----------------:|-------------------:|
| VaR à 95 %                        |          $31.64 |             $43.50 |
| Expected Shortfall à 95 %         |          $45.10 |             $61.80 |
| Perte moyenne (µ<0 = gain moyen)  |         −$41.09 |            −$28.70 |
| Écart-type des pertes             |          $49.94 |             $60.40 |

La perte moyenne négative signifie qu'en moyenne, le portefeuille **gagne** de la valeur sur un an (en cohérence avec les dérives positives calibrées sur les marchés haussiers 2015–2024). La VaR positive indique qu'il existe néanmoins un risque de perte significatif dans les queues de distribution.

---

## 9. Analyse

### 9.1 Scalabilité quasi-linéaire du GPU

La courbe GPU est quasi-linéaire avec le nombre de chemins : multiplier les chemins par 10 (10 000 → 100 000) multiplie le temps par ~6, pas par 10. Cela s'explique par les **effets de remplissage du GPU** : avec 10 000 chemins, le GPU n'est pas saturé et une fraction de ses cœurs n'est pas utilisé et en attente. À 100 000+ chemins, le GPU est mieux utilisé et l'efficacité augmente.

Le CPU, lui, suit une loi rigoureusement linéaire : chaque nouveau chemin coûte le même temps supplémentaire (environ 0.045 ms par chemin).

### 9.2 L'accélération croît avec la taille du problème

L'accélération passe de ×35 (10 000 chemins) à ×82 (1 000 000 chemins). Cette croissance est typique des GPU : le coût fixe de lancement du kernel et du transfert mémoire CPU→GPU est amorti sur plus de calculs, et le GPU est de mieux en mieux utilisé. C'est la raison pour laquelle les GPU sont surtout utiles pour les **gros volumes de données homogènes**.

### 9.3 Convergence des métriques de risque

On observe que VaR et ES convergent rapidement : les valeurs obtenues avec 50 000 chemins (VaR = \$31.30) et 1 000 000 chemins (VaR = $31.64) sont très proches. En pratique, 100 000–200 000 chemins suffisent pour une estimation stable. L'intérêt du GPU est alors de rendre cette estimation quasi-instantanée (0.065 s), permettant de relancer le calcul des dizaines de fois dans une session interactive.

### 9.4 Effet du stress test

Le scénario de crise augmente la VaR d'environ **+37 %** et l'ES d'environ **+37 %**. Cela illustre l'importance des corrélations : quand tout chute en même temps (corrélations élevées entre les actifs actions), la diversification ne protège plus. L'or et les obligations, dont la corrélation avec les actions devient très négative en période de crise, atténuent partiellement la chute mais pas suffisamment.

---

## 10. Conclusion

Ce projet démontre concrètement que la parallélisation sur GPU peut apporter des gains de performance très significatifs (**×35 à ×82**) sur un problème naturellement parallèle comme la simulation Monte Carlo.

Les points clés à retenir :

1. **Le choix du problème est décisif.** Monte Carlo est un candidat parfait pour le GPU : millions de calculs indépendants, peu de synchronisation, structure de données simple.

2. **Les optimisations CUDA ont un impact mesurable.** L'utilisation de la mémoire partagée, des registres locaux, de la coalescence en écriture et du générateur xoroshiro128p permettent d'extraire le maximum des capacités matérielles du GPU.

3. **La précision est préservée.** Les métriques de risque produites par le GPU (VaR, ES) sont statistiquement identiques à celles du CPU avec le même seed.

4. **L'architecture hexagonale facilite la comparaison.** En séparant le moteur CPU et le moteur GPU derrière une interface commune (`SimulationEngine`), on peut les comparer rigoureusement et les substituer sans modifier le reste du code.

Ce type de moteur de simulation est directement réutilisable dans des contextes réels : calcul intraday de risque, scénarios de stress automatisés, optimisation de portefeuille par simulation.

---

*Projet réalisé par Mehdi ALI, Mattéo MOISANT et Romain BLANCHOT.*

### Sources et références

**Méthodes Monte Carlo**
- Wikipedia — *Méthode de Monte-Carlo* : https://fr.wikipedia.org/wiki/M%C3%A9thode_de_Monte-Carlo
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer. — référence de base pour l'application de Monte Carlo à la finance.

**Modèle de diffusion (GBM)**
- Wikipedia — *Geometric Brownian motion* : https://en.wikipedia.org/wiki/Geometric_Brownian_motion

**Métriques de risque (VaR, Expected Shortfall)**
- Wikipedia — *Value at risk* : https://fr.wikipedia.org/wiki/Value_at_risk
- Wikipedia — *Expected shortfall* : https://en.wikipedia.org/wiki/Expected_shortfall

**Décomposition de Cholesky**
- Wikipedia — *Cholesky decomposition* : https://fr.wikipedia.org/wiki/Factorisation_de_Cholesky

**Programmation GPU et CUDA**
- NVIDIA (2024). *CUDA C++ Programming Guide*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Documentation Numba CUDA : https://numba.readthedocs.io/en/stable/cuda/index.html
- (bien-sûr, le cours :D)

**Générateur de nombres aléatoires xoroshiro128+**
- Wikipedia — *Xorshift* : https://en.wikipedia.org/wiki/Xorshift#xoshiro

**Outils utilisés**
- NumPy : https://numpy.org
- Numba : https://numba.pydata.org
- Conda : https://www.anaconda.com/

> L’outil Claude (Anthropic) a été utilisé de manière ponctuelle à des fins d’assistance rédactionnelle, notamment pour la reformulation de commentaires dans le code, l’amélioration stylistique de certaines formulations et des suggestions mineures concernant la lisibilité ou la qualité du code.

> L’ensemble de la réflexion, des choix d’architecture, de la conception et du travail intellectuel associés à ce projet a été réalisé exclusivement par nous même.
# Objectif du projet ; 

> Prendre un code séquentiel (CPU) de votre choix et l'accélérer avec CUDA via Numba en Python. Dans votre cas : un moteur Monte Carlo de gestion de risque de portefeuille financier — calcul de VaR et Expected Shortfall sur 8 actifs corrélés.


Ce qu'il faut produire :

-  Un code CPU de référence (MonteCarloCPU — NumPy)
-  Au moins un custom kernel CUDA (le _gbm_kernel)
-  Pas de matmul triviale, pas d'element-wise simple

Point à préciser et décrire : 
- Fonctionnement VULGARISÉ u projet (moteur montecarlo, utilisation, portefeuille, risque...)
- Optimisation GPU (techniques utilisés, optimisations CUDA, exemples...)
- Comparaison de performances
- Analyse des performances
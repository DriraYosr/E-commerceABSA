# 📦 Guide de Filtrage par Produit

## Vue d'ensemble

Le dashboard permet maintenant de faire des prévisions soit sur **tous les produits** (agrégés), soit sur **un produit spécifique**.

---

## 🎯 Comment Utiliser

### Étape 1: Ouvrir la Sidebar
Dans la barre latérale gauche, vous trouverez les filtres globaux.

### Étape 2: Sélectionner un Produit

```
🔧 Global Filters
├─ Date Range: [sélecteur de dates]
├─ Product Category: All / Beauty / Electronics / ...
└─ Product (ASIN): 
   ├─ All Products (défaut)
   ├─ B08X123ABC (250 reviews)
   ├─ B09Y456DEF (180 reviews)
   └─ ...
```

**Options disponibles:**
- **"All Products"**: Analyse tous les produits ensemble (comportement par défaut)
- **Produit spécifique**: Ex. "B08X123ABC (250 reviews)" - analyse uniquement ce produit

---

## 📊 Impact sur la Page Forecasting

### Scénario A: "All Products" Sélectionné

```
📦 Products Included in Forecast (15 products, 1,250 reviews)
[Expander] Cliquer pour voir la liste:
  - B08X123ABC: 250 reviews (20.0%)
  - B09Y456DEF: 180 reviews (14.4%)
  - B07Z789GHI: 95 reviews (7.6%)
  ...
```

**Prédiction:**
- Agrège le sentiment de TOUS les produits
- Calcule la moyenne du sentiment "battery" (par exemple) de tous les produits
- Utile pour: Voir la tendance générale d'un aspect sur toute votre gamme

**Exemple d'interprétation:**
> "Le sentiment global pour 'battery' à travers nos 15 produits va augmenter de 0.65 à 0.75 dans les 90 prochains jours"

### Scénario B: Produit Spécifique Sélectionné

```
📦 Single Product Analysis: B08X123ABC (250 reviews)
```

**Prédiction:**
- Analyse UNIQUEMENT ce produit
- Sentiment "battery" pour CE produit spécifiquement
- Utile pour: Surveiller un produit problématique ou best-seller

**Exemple d'interprétation:**
> "Le sentiment 'battery' pour le produit B08X123ABC va décliner de 0.70 à 0.60 dans les 90 prochains jours"

---

## 🔍 Cas d'Usage

### 1. **Analyse de Gamme de Produits**
**Situation**: Vous avez 20 produits similaires (ex: écouteurs Bluetooth)

**Approche**:
```
1. Sélectionner "All Products"
2. Aspect: "sound quality"
3. Forecast → Voir la tendance générale
```

**Résultat**: Tendance globale du sentiment "son" pour toute la gamme

### 2. **Surveillance d'un Best-Seller**
**Situation**: Un produit représente 40% de vos ventes

**Approche**:
```
1. Sélectionner ce produit spécifique (ex: B08X123ABC)
2. Aspect: "battery"
3. Forecast → Surveiller ce produit critique
```

**Résultat**: Prédiction spécifique pour ce produit important

### 3. **Diagnostic d'un Produit Problématique**
**Situation**: Un produit a beaucoup de reviews négatives récentes

**Approche**:
```
1. Sélectionner ce produit (ex: B09Y456DEF)
2. Aspect: "quality"
3. Date Range: derniers 3 mois
4. Forecast → Voir si la situation s'améliore ou empire
```

**Résultat**: Prédiction pour détecter si le problème persiste

### 4. **Comparaison Avant/Après Nouveau Modèle**
**Situation**: Vous avez lancé une nouvelle version d'un produit

**Approche**:
```
Option A: Ancien modèle (B08X123ABC)
├─ Forecast sur 90 jours
└─ Noter la tendance

Option B: Nouveau modèle (B10A789XYZ)
├─ Forecast sur 90 jours
└─ Comparer avec ancien modèle
```

**Résultat**: Comparer les trajectoires des deux versions

---

## 📈 Interprétation des Résultats

### Indicateur de Scope

Après génération du forecast, vous verrez:

**Pour "All Products":**
```
📊 Analysis Scope: Aspect 'battery' across 15 products (aggregated sentiment)
```
→ Rappel que c'est une moyenne de tous les produits

**Pour un produit spécifique:**
```
📊 Analysis Scope: Aspect 'battery' for product B08X123ABC
```
→ Rappel que c'est uniquement ce produit

### Interprétation des Métriques

Les métriques (Current Sentiment, Predicted Change, etc.) s'appliquent selon le scope:

| Métrique | All Products | Produit Spécifique |
|----------|--------------|-------------------|
| **Current Sentiment** | Moyenne tous produits | Ce produit uniquement |
| **Predicted Change** | Changement agrégé | Changement pour ce produit |
| **Volatility** | Variabilité globale | Variabilité de ce produit |
| **Changepoints** | Shifts globaux | Shifts pour ce produit |

---

## ⚡ Tips & Best Practices

### ✅ Bonnes Pratiques

1. **Commencer large, affiner ensuite**
   - D'abord analyser "All Products" pour vue d'ensemble
   - Puis zoomer sur produits spécifiques si nécessaire

2. **Vérifier le nombre de reviews**
   - Minimum 50-100 reviews pour prédictions fiables
   - Si < 50, considérer agréger plusieurs produits

3. **Combiner avec Date Range**
   - Analyser un produit lancé récemment: filtrer par date de lancement
   - Comparer "avant" vs "après" un changement

4. **Utiliser l'expander des produits**
   - Toujours vérifier quels produits sont inclus
   - Identifier si un produit dominant influence la moyenne

### ⚠️ Pièges à Éviter

1. **Ne pas confondre agrégé et spécifique**
   - Une baisse globale peut cacher qu'1 produit sur 10 a un gros problème
   - Toujours vérifier l'indicateur "Analysis Scope"

2. **Attention aux produits avec peu de données**
   - Un produit avec 10 reviews → prédictions peu fiables
   - Préférer "All Products" si données insuffisantes

3. **Ne pas sur-interpréter les prédictions individuelles**
   - Forecast sur 1 produit = plus de bruit
   - Agréger plusieurs produits = signal plus clair

4. **Vérifier la distribution des reviews**
   - Si 1 produit = 80% des reviews, "All Products" ≈ ce produit
   - Dans ce cas, analyser les 2 scénarios donne résultats similaires

---

## 🔄 Workflow Recommandé

### Pour une Analyse Complète

```
1️⃣ Vue d'ensemble
   ├─ Sélectionner: "All Products"
   ├─ Date Range: Derniers 6 mois
   ├─ Aspect: Choisir aspect critique (ex: "quality")
   └─ Forecast → Noter tendance générale

2️⃣ Identifier produits problématiques
   ├─ Regarder expander "Products Included"
   ├─ Noter produits avec beaucoup de reviews
   └─ Si 1-2 produits dominent, les analyser séparément

3️⃣ Analyse individuelle
   ├─ Sélectionner produit dominant
   ├─ Même aspect
   ├─ Forecast → Comparer avec moyenne
   └─ Répéter pour top 3-5 produits

4️⃣ Action
   ├─ Si tous produits baissent → Problème de gamme
   ├─ Si 1 seul baisse → Problème spécifique produit
   └─ Prioriser actions selon impact (% reviews)
```

---

## 📊 Exemples Visuels

### Exemple 1: Tous Produits vs Spécifique

**Scénario**: Analyse de "battery" pour gamme de montres connectées

```
All Products:
├─ 8 produits, 1,200 reviews
├─ Sentiment actuel: 0.68
├─ Prédiction +90j: 0.72 (+0.04) ✅
└─ Interprétation: Amélioration globale du sentiment batterie

Produit B08X123 (best-seller, 450 reviews):
├─ 1 produit, 450 reviews  
├─ Sentiment actuel: 0.75
├─ Prédiction +90j: 0.73 (-0.02) ⚠️
└─ Interprétation: Best-seller en déclin malgré amélioration globale!

→ Action: Enquêter sur le best-seller spécifiquement
```

### Exemple 2: Lancement Nouveau Produit

**Scénario**: Nouveau modèle lancé il y a 2 mois

```
Filtre:
├─ Date Range: 2 derniers mois
├─ Produit: B10A789XYZ (nouveau)
├─ 85 reviews collectées
└─ Forecast → Voir si momentum positif ou négatif

Si prédiction monte:
└─ ✅ Lancement réussi, continuer marketing

Si prédiction baisse:
└─ ⚠️ Problème détecté tôt, corriger avant scaling
```

---

## 🎓 Questions Fréquentes

**Q: Combien de produits minimum pour "All Products"?**  
A: Pas de minimum, mais 3+ produits recommandés pour moyenne stable.

**Q: Puis-je analyser plusieurs produits spécifiques en même temps?**  
A: Pas encore. Actuellement: "All" ou 1 seul. Fonctionnalité multi-select à venir.

**Q: Le filtre s'applique à toutes les pages?**  
A: Oui! Le filtre "Product (ASIN)" dans la sidebar affecte toutes les pages du dashboard.

**Q: Comment savoir si j'ai assez de données?**  
A: Regardez le nombre de reviews affiché. Minimum 50 recommandé, 100+ idéal.

**Q: Les prédictions "All Products" sont-elles une moyenne simple?**  
A: Oui, c'est une moyenne du sentiment par période (jour/semaine/mois) de tous les produits.

---

**Version**: 1.0  
**Dernière mise à jour**: 21 Novembre 2025

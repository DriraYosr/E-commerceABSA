# Guide de Fine-Tuning des Modèles de Prévision

## 🎯 Objectif
Ce guide vous aide à optimiser les prédictions en ajustant les hyperparamètres des modèles Prophet et ARIMA.

---

## 📊 Prophet: Paramètres à Ajuster

### 1. **Trend Flexibility (changepoint_prior_scale)**
**Valeur par défaut**: 0.05  
**Plage**: 0.001 - 0.5

**Problème à Résoudre**:
- ❌ **Prédictions trop rigides** → La courbe ignore les changements de tendance
- ❌ **Prédictions trop chaotiques** → Suit chaque petit bruit

**Comment Ajuster**:
```
Valeur BASSE (0.001-0.01):
└─ Tendance très lisse et stable
└─ Ignore les petits changements
└─ Utiliser si: Sentiment très stable, peu de variations réelles

Valeur MOYENNE (0.05-0.1):
└─ Balance entre stabilité et flexibilité
└─ Détecte les vrais changements de tendance
└─ Utiliser si: Pattern normal (RECOMMANDÉ)

Valeur HAUTE (0.2-0.5):
└─ S'adapte à chaque changement
└─ Risque de surapprentissage (overfit)
└─ Utiliser si: Sentiment très volatile, beaucoup de changements
```

### 2. **Seasonality Strength (seasonality_prior_scale)**
**Valeur par défaut**: 1.0  
**Plage**: 0.01 - 10.0

**Problème à Résoudre**:
- ❌ **Trop de vagues/oscillations** → Prédictions montent et descendent trop
- ❌ **Pas assez de patterns** → Ignore les cycles hebdomadaires/annuels réels

**Comment Ajuster**:
```
Valeur BASSE (0.01-0.5):
└─ Ignore presque toute la saisonnalité
└─ Prédictions très lisses, presque linéaires
└─ Utiliser si: Pas de pattern hebdomadaire/annuel clair

Valeur MOYENNE (1.0-3.0):
└─ Détecte les patterns modérés
└─ Balance entre tendance et cycles
└─ Utiliser si: Pattern classique (RECOMMANDÉ)

Valeur HAUTE (5.0-10.0):
└─ Patterns saisonniers très marqués
└─ Courbes avec beaucoup d'oscillations
└─ Utiliser si: Forte variabilité hebdomadaire (ex: weekend vs semaine)
```

### 3. **Weekly Seasonality**
**Options**: Activé / Désactivé  
**Défaut**: Activé si >14 jours de données

**Utiliser**:
- ✅ **Activé**: Si sentiment varie selon jour de semaine (weekend vs lundi)
- ❌ **Désactivé**: Si aucun pattern hebdomadaire (données agrégées mensuellement)

### 4. **Yearly Seasonality**
**Options**: Activé / Désactivé  
**Défaut**: Activé si >365 jours de données

**Utiliser**:
- ✅ **Activé**: Si sentiment varie selon saison (Noël, été, etc.)
- ❌ **Désactivé**: Si <1 an de données ou pas de cycle annuel

---

## 📈 ARIMA: Paramètres (p, d, q)

### Format: ARIMA(p, d, q)
- **p**: Ordre AutoRégressif (AR) - utilise valeurs passées
- **d**: Ordre de Différenciation (I) - rend les données stationnaires
- **q**: Ordre Moyenne Mobile (MA) - utilise erreurs passées

### 1. **AR Order (p)**
**Valeur par défaut**: 1  
**Plage**: 0 - 5

**Signification**: Combien de valeurs passées utiliser pour prédire l'avenir

```
p = 0: Aucune dépendance avec le passé
p = 1: Utilise sentiment d'hier (RECOMMANDÉ pour début)
p = 2: Utilise sentiment d'hier et avant-hier
p = 3-5: Utilise plusieurs jours passés (risque overfit)
```

**Quand Augmenter p**:
- Sentiment actuel fortement corrélé avec passé récent
- Patterns d'autocorrélation claire
- Données lisses avec momentum

### 2. **Differencing (d)**
**Valeur par défaut**: 1  
**Plage**: 0 - 2

**Signification**: Combien de fois "différencier" pour stabiliser la série

```
d = 0: Données déjà stationnaires (pas de tendance)
d = 1: Tendance linéaire (RECOMMANDÉ)
d = 2: Tendance quadratique (rare, attention overfit)
```

**Quand Augmenter d**:
- Données non-stationnaires (tendance croissante/décroissante)
- Moyenne change dans le temps
- Variance non constante

**Attention**: d=2 souvent cause overfit, rarement nécessaire!

### 3. **MA Order (q)**
**Valeur par défaut**: 1  
**Plage**: 0 - 5

**Signification**: Combien d'erreurs passées utiliser pour corriger la prédiction

```
q = 0: Pas de correction par erreurs passées
q = 1: Corrige en fonction de dernière erreur (RECOMMANDÉ)
q = 2-3: Utilise plusieurs erreurs passées
q > 3: Risque overfit
```

**Quand Augmenter q**:
- Erreurs de prédiction ont un pattern
- Chocs temporaires affectent plusieurs périodes
- Données bruitées

---

## 🔍 Guide de Diagnostic

### Symptôme 1: **Prédictions trop lisses / ignorent changements**
**Solution Prophet**:
- ↑ Augmenter `changepoint_prior_scale` (0.05 → 0.15)
- ✅ Vérifier que weekly/yearly seasonality est activée si pattern existe

**Solution ARIMA**:
- ↑ Augmenter `p` (1 → 2 ou 3)
- ↓ Réduire `d` si >1 (2 → 1)

### Symptôme 2: **Prédictions trop chaotiques / montagnes russes**
**Solution Prophet**:
- ↓ Réduire `changepoint_prior_scale` (0.05 → 0.01)
- ↓ Réduire `seasonality_prior_scale` (1.0 → 0.3)
- ❌ Désactiver weekly seasonality

**Solution ARIMA**:
- ↓ Réduire `p` (3 → 1)
- ↓ Réduire `q` (3 → 1)

### Symptôme 3: **Prédictions ne capturent pas les cycles hebdomadaires**
**Solution Prophet**:
- ✅ Activer `Weekly Seasonality`
- ↑ Augmenter `seasonality_prior_scale` (1.0 → 3.0)

**Solution ARIMA**:
- Passer à SARIMA (pas encore implémenté)
- Ou agréger par semaine au lieu de jour

### Symptôme 4: **Intervalles de confiance trop larges**
**Cause**: Pas assez de données ou trop de bruit

**Solution**:
- ↑ Augmenter `min_reviews` (20 → 50)
- Changer `Aggregation` (Daily → Weekly)
- Collecter plus de données

### Symptôme 5: **Erreur "Insufficient data"**
**Solution**:
- ↓ Réduire `min_reviews` (20 → 10)
- Changer aspect vers un plus populaire
- Changer `Aggregation` (Daily → Weekly)

---

## 🎓 Recettes Recommandées

### **Sentiment Stable, Peu de Variations**
```
Prophet:
├─ Trend Flexibility: 0.01-0.03 (bas)
├─ Seasonality Strength: 0.3-0.5 (bas)
├─ Weekly Seasonality: Désactivé
└─ Yearly Seasonality: Désactivé

ARIMA: (0, 1, 1) ou (1, 1, 0)
```

### **Sentiment Normal, Variations Modérées** (DÉFAUT)
```
Prophet:
├─ Trend Flexibility: 0.05 (moyen)
├─ Seasonality Strength: 1.0 (moyen)
├─ Weekly Seasonality: Activé
└─ Yearly Seasonality: Selon données

ARIMA: (1, 1, 1)
```

### **Sentiment Volatil, Beaucoup de Changements**
```
Prophet:
├─ Trend Flexibility: 0.15-0.3 (élevé)
├─ Seasonality Strength: 3.0-5.0 (élevé)
├─ Weekly Seasonality: Activé
└─ Yearly Seasonality: Activé

ARIMA: (2, 1, 2) ou (3, 1, 1)
```

### **Données avec Tendance Claire (croissance/déclin)**
```
Prophet:
├─ Trend Flexibility: 0.05-0.1
├─ Seasonality Strength: 0.5-1.0 (ne pas masquer tendance)
├─ Weekly Seasonality: Selon pattern
└─ Yearly Seasonality: Désactivé

ARIMA: (1, 1, 1) ou (2, 1, 0)
```

---

## 📊 Méthode d'Optimisation Systématique

### Étape 1: **Baseline**
1. Lancer avec paramètres par défaut
2. Observer la forme générale de la prédiction
3. Identifier le problème principal (trop lisse? trop chaotique?)

### Étape 2: **Ajustement Principal**
1. Si trop lisse:
   - Prophet: ↑ `changepoint_prior_scale` +0.05
   - ARIMA: ↑ `p` +1
2. Si trop chaotique:
   - Prophet: ↓ `seasonality_prior_scale` -0.5
   - ARIMA: ↓ `p` -1

### Étape 3: **Fine-Tuning**
1. Ajuster la saisonnalité (Prophet uniquement)
2. Vérifier les intervalles de confiance
3. Comparer avec données historiques

### Étape 4: **Validation**
1. La tendance générale est-elle réaliste?
2. Les cycles correspondent-ils aux patterns connus?
3. Les intervalles de confiance sont-ils raisonnables?

---

## ⚡ Quick Tips

### Pour Prophet:
- 🎯 **Commencer par `seasonality_strength`** - impact le plus visible
- 🔄 **Désactiver seasonality** si prédictions trop ondulées
- ⏱️ **Weekly = patterns jour de semaine**, Yearly = patterns saisonniers

### Pour ARIMA:
- 🎯 **Commencer avec (1,1,1)** - bon point de départ
- ⚠️ **Jamais d>1** sauf cas très spéciaux
- 📈 **Augmenter p si autocorrélation**, q si erreurs corrélées

### Général:
- 📊 **Agréger Weekly** si Daily trop bruité
- 🔢 **Min Reviews ≥50** pour prédictions stables
- 🔄 **Comparer les 2 modèles** - choisir le meilleur
- 💾 **Noter les bons paramètres** par aspect

---

**Version**: 1.0  
**Dernière mise à jour**: 21 Novembre 2025

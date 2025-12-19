# 📊 Guide de Qualité des Données pour Time Series

## 🎯 Problème Résolu

### Avant: Données de Mauvaise Qualité
```
Problèmes courants:
❌ Jours avec seulement 1-2 reviews → Moyenne non représentative
❌ Grands trous temporels (ex: 30 jours sans données)
❌ Prédictions basées sur des données sporadiques
❌ Modèle apprend du bruit au lieu du signal
```

### Après: Données Filtrées et Continues
```
Améliorations:
✅ Seulement les périodes avec volume suffisant (min 5 reviews par défaut)
✅ Détection automatique des trous temporels
✅ Sélection du segment continu le plus long
✅ Meilleure qualité de prédiction
```

---

## 🔧 Nouvelle Fonctionnalité: Contrôle de Qualité

### Interface Dashboard

```
📊 Data Quality Controls
├─ Min Reviews per Day: [slider 1-50, défaut 5]
└─ Expected Min Data Points: ~40
```

### Paramètre: `min_samples_per_period`

**Définition**: Nombre minimum de reviews requis par période temporelle (jour/semaine/mois)

**Impact**:
- **Valeur basse (1-3)**: Plus de data points, mais qualité moyenne
- **Valeur moyenne (5-10)**: Balance qualité/quantité ⭐ **RECOMMANDÉ**
- **Valeur haute (15-50)**: Excellente qualité, mais peu de data points

---

## 📈 Filtrage en 3 Étapes

### Étape 1: Filtrage par Volume

**Code**:
```python
ts_data = ts_data[ts_data['count'] >= min_samples_per_period]
```

**Exemple (min_samples_per_period=5)**:
```
AVANT:
Date       | y    | count
-----------|------|-------
2024-01-01 | 0.70 | 8   ✅ Gardé
2024-01-02 | 0.85 | 2   ❌ Supprimé (count < 5)
2024-01-03 | 0.68 | 6   ✅ Gardé
2024-01-04 | 0.90 | 1   ❌ Supprimé (count < 5)
2024-01-05 | 0.72 | 12  ✅ Gardé

APRÈS:
Date       | y    | count
-----------|------|-------
2024-01-01 | 0.70 | 8
2024-01-03 | 0.68 | 6
2024-01-05 | 0.72 | 12
```

**Pourquoi c'est important**:
- Moyenne de 2 reviews n'est pas statistiquement fiable
- Une seule review extrême peut fausser toute la journée
- Volume élevé = confiance dans la mesure

---

### Étape 2: Détection des Trous Temporels

**Seuils de Gap Maximum**:
```python
Agrégation Daily:   max_gap = 7 jours
Agrégation Weekly:  max_gap = 4 semaines
Agrégation Monthly: max_gap = 2 mois
```

**Exemple (Daily)**:
```
Time Series:
2024-01-01 ────┐
2024-01-02     │ Segment 1 (continu)
2024-01-03     │
2024-01-04 ────┘
               ↓ GAP de 25 jours! (> 7 jours max)
2024-01-30 ────┐
2024-01-31     │
2024-02-01     │ Segment 2 (continu)
2024-02-02     │
2024-02-03 ────┘

→ Modèle voit 2 segments séparés
```

**Pourquoi détecter les gaps**:
- Un trou de 30 jours rompt la continuité temporelle
- ARIMA/Prophet supposent des données régulières
- Gap = perte de contexte pour les prédictions

---

### Étape 3: Sélection du Segment le Plus Long

**Logique**:
```python
# Diviser à chaque gap
segments = []
for gap in large_gaps:
    segment = ts_data[start:gap]
    if len(segment) >= 10:
        segments.append(segment)

# Garder le plus long
ts_data = max(segments, key=len)
```

**Exemple Visuel**:
```
Segment 1: 15 jours (Jan 1-15)
Segment 2: 45 jours (Feb 1 - Mar 15)  ← SÉLECTIONNÉ!
Segment 3: 8 jours (Apr 1-8) - Ignoré (< 10 points)

Résultat: Modèle entraîné sur Segment 2 uniquement
```

**Message Dashboard**:
```
⚠️ Time series had gaps. Using longest continuous segment: 
   45 periods from 2024-02-01 to 2024-03-15
```

---

## 🎓 Exemples Concrets

### Exemple 1: Produit Nouveau avec Données Sporadiques

**Scénario**: Produit lancé il y a 3 mois, reviews irrégulières

**Données Brutes** (Daily aggregation):
```
Total: 90 jours possibles
Jours avec ≥1 review: 42 jours
Jours avec ≥5 reviews: 18 jours
```

**Avec min_samples_per_period=5**:
```
✅ Garde seulement 18 jours avec volume suffisant
✅ Trouve un segment continu de 12 jours (Feb 10-21)
✅ Prédictions basées sur données de qualité
```

**Impact**:
- Sans filtre: 42 points bruités
- Avec filtre: 12 points de qualité ← **Meilleur signal!**

---

### Exemple 2: Aspect Populaire avec Bonne Distribution

**Scénario**: "battery" - 500 reviews sur 60 jours

**Données Brutes**:
```
60 jours, chaque jour: 5-15 reviews
Moyenne: 8.3 reviews/jour
Pas de gaps
```

**Avec min_samples_per_period=5**:
```
✅ Garde tous les 60 jours (tous ≥5 reviews)
✅ Aucun gap détecté
✅ Segment unique de 60 jours
✅ Excellente base pour prédiction!
```

**Résultat**: `60 historical data points` (optimal)

---

### Exemple 3: Aspect Rare avec Données Dispersées

**Scénario**: "screen_protector" - 80 reviews sur 90 jours

**Données Brutes**:
```
Jours avec reviews: 35 jours
Distribution irrégulière:
- Semaine 1-2: 15 reviews/jour ✅
- Semaine 3-6: 0-2 reviews/jour ❌
- Semaine 7-8: 12 reviews/jour ✅
```

**Avec min_samples_per_period=5**:
```
❌ Segment 1: 14 jours (Semaine 1-2)
❌ Segment 2: 0 jours (Semaine 3-6, tous < 5)
❌ Segment 3: 14 jours (Semaine 7-8)

→ Sélectionne un des segments de 14 jours
⚠️ Recommandation: Agréger en Weekly au lieu de Daily
```

---

## ⚙️ Comment Choisir les Paramètres

### Paramètre 1: `min_samples_per_period`

| Valeur | Usage | Avantages | Inconvénients |
|--------|-------|-----------|---------------|
| **1-2** | Produits nouveaux, aspects rares | Maximum de data points | Données bruitées |
| **3-5** | Usage standard | Balance qualité/quantité | Quelques points perdus |
| **5-10** | ⭐ **RECOMMANDÉ** | Bonne qualité statistique | Moins de data points |
| **10-20** | Produits populaires | Excellente qualité | Beaucoup de points perdus |
| **>20** | Analyses spéciales | Qualité maximale | Très peu de data points |

### Paramètre 2: `Aggregation Frequency`

**Relation avec min_samples**:
```
Daily + min_samples=5
→ Besoin: 5 reviews/jour
→ Si produit a ~30 reviews/semaine → seulement 4.3/jour
→ Résultat: Données insuffisantes ❌

Solution: Passer à Weekly
→ Weekly + min_samples=5
→ Besoin: 5 reviews/semaine
→ 30 reviews/semaine → OK! ✅
```

**Règle générale**:
```
Daily:   Pour produits avec >50 reviews/jour
Weekly:  Pour produits avec >20 reviews/semaine (STANDARD)
Monthly: Pour produits avec <50 reviews/mois
```

---

## 🔍 Diagnostic: Quand Ajuster

### Erreur: "Insufficient data after volume filtering"

**Cause**: Trop peu de périodes avec le volume minimum requis

**Solutions**:
```
Option 1: ↓ Réduire min_samples_per_period (5 → 3)
Option 2: ↑ Changer aggregation (Daily → Weekly)
Option 3: ↑ Augmenter date range (3 mois → 6 mois)
Option 4: Choisir un aspect plus populaire
```

### Erreur: "No continuous time segment found"

**Cause**: Données trop fragmentées, tous les segments <10 points

**Solutions**:
```
Option 1: ↓ Réduire min_samples_per_period drastiquement
Option 2: ↑ Changer aggregation pour réduire gaps
Option 3: Sélectionner un produit spécifique (pas "All")
Option 4: Vérifier qualité des données source
```

### Warning: "Using longest continuous segment: X periods"

**Cause**: Gaps détectés dans les données

**Interprétation**:
```
✅ X ≥ 30: Excellent, segment suffisant
✅ X ≥ 20: Bon, prédictions fiables
⚠️ X ≥ 10: Acceptable, mais court terme seulement
❌ X < 10: Insuffisant, revoir paramètres
```

---

## 📊 Métriques de Qualité

### Metric 1: Data Coverage

```python
Coverage = (periods_with_data / total_periods) × 100%

Exemple:
90 jours possibles
45 jours avec ≥5 reviews
Coverage = 45/90 = 50%

Interprétation:
✅ >70%: Excellente couverture
✅ 40-70%: Bonne couverture
⚠️ 20-40%: Couverture moyenne
❌ <20%: Couverture faible
```

### Metric 2: Segment Continuity

```python
Continuity = (largest_segment / total_periods) × 100%

Exemple:
90 jours total
60 jours dans plus grand segment
Continuity = 60/90 = 67%

Interprétation:
✅ >80%: Très continu
✅ 50-80%: Relativement continu
⚠️ 30-50%: Fragmenté
❌ <30%: Très fragmenté
```

### Metric 3: Average Volume

```python
Avg Volume = total_reviews / periods_with_data

Exemple:
500 reviews
45 périodes avec données
Avg = 500/45 = 11.1 reviews/période

Interprétation:
✅ >15: Excellent volume
✅ 8-15: Bon volume
⚠️ 5-8: Volume acceptable
❌ <5: Volume faible
```

---

## 🎯 Workflows Recommandés

### Workflow 1: Nouveau Produit

```
1. Commencer avec:
   ├─ Aggregation: Weekly
   ├─ Min samples: 3
   └─ Date range: Depuis lancement

2. Vérifier résultat:
   ├─ Si ≥20 data points: ✅ OK
   └─ Si <20 data points: Passer à Monthly

3. Interpréter avec prudence:
   └─ Peu d'historique = grandes incertitudes
```

### Workflow 2: Produit Établi

```
1. Configuration standard:
   ├─ Aggregation: Daily ou Weekly
   ├─ Min samples: 5
   └─ Date range: 6-12 mois

2. Optimisation:
   ├─ Vérifier warning sur gaps
   ├─ Si gaps: Réduire date range pour capturer période continue
   └─ Comparer Daily vs Weekly

3. Validation:
   └─ Viser ≥30 data points pour prédictions stables
```

### Workflow 3: Aspect Rare

```
1. Ajustements nécessaires:
   ├─ Aggregation: Weekly ou Monthly
   ├─ Min samples: 1-3 (plus permissif)
   └─ Considérer grouper avec aspects similaires

2. Alternatives:
   ├─ Sélectionner produit spécifique (pas All)
   ├─ Analyser seulement période récente
   └─ Utiliser Weekly obligatoirement
```

---

## 💡 Best Practices

### ✅ À Faire

1. **Toujours vérifier le nombre de data points**
   - Minimum absolu: 10 points
   - Recommandé: 30+ points
   - Idéal: 60+ points

2. **Adapter l'agrégation au volume**
   - Beaucoup de reviews → Daily
   - Volume moyen → Weekly
   - Peu de reviews → Monthly

3. **Lire les warnings**
   - "Using longest segment" → Normal si gaps temporels
   - Noter la période utilisée pour interprétation

4. **Commencer conservateur**
   - min_samples=5 par défaut
   - Ajuster seulement si problèmes

### ❌ À Éviter

1. **Ne pas forcer Daily sur aspects rares**
   - Résultat: Beaucoup de jours vides
   - Solution: Passer à Weekly

2. **Ne pas mettre min_samples trop haut**
   - min_samples=20 → Perd trop de données
   - Sauf si volume vraiment élevé

3. **Ne pas ignorer les erreurs**
   - "Insufficient data" = problème réel
   - Ajuster paramètres, pas forcer

4. **Ne pas analyser segments trop courts**
   - <10 points = prédictions non fiables
   - Mieux: Augmenter date range ou changer aspect

---

## 🔬 Validation: Comment Savoir Si C'est Bon?

### Checklist de Qualité

```
✅ Data points ≥ 30
✅ Segment continu (pas de warning gap)
✅ Average volume ≥ 5 reviews/période
✅ Intervalles de confiance raisonnables (<0.3 de largeur)
✅ Trend visuellement cohérent
```

### Indicateurs Visuels

**Bon forecast**:
```
- Courbe historique lisse
- Intervalle de confiance stable
- Tendance claire et progressive
```

**Mauvais forecast**:
```
- Courbe historique erratique (zigzag)
- Intervalle de confiance très large
- Prédiction plate ou extrême
```

---

**Version**: 1.0  
**Dernière mise à jour**: 21 Novembre 2025  
**Auteur**: ABSA Dashboard Team

# 🏫 Analyse Prédictive des Performances Scolaires

## 📋 Description

Ce projet implémente une méthodologie complète d'analyse des données éducatives pour prédire et comprendre les facteurs de réussite scolaire. Il transforme les données brutes d'élèves en insights actionnables pour les équipes pédagogiques.

## 🎯 Objectifs

- **Prédire** la performance scolaire des élèves
- **Identifier** les facteurs clés de réussite et d'échec
- **Fournir** des recommandations pédagogiques basées sur les données
- **Générer** des rapports automatisés pour la prise de décision

## 🔧 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances
```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

### Lancement rapide avec données synthétiques
```python
from analyse_scolaire import AnalyseurPerformanceScolaire

# Créer l'analyseur
analyseur = AnalyseurPerformanceScolaire()

# Lancer l'analyse complète
analyseur.analyse_complete(generer_donnees=True)
```

### Utilisation avec vos propres données
```python
# Charger vos données
analyseur = AnalyseurPerformanceScolaire('mes_donnees.xlsx')

# Ou charger après création
analyseur.charger_donnees('mes_donnees.xlsx')

# Lancer l'analyse
analyseur.analyse_complete(generer_donnees=False)
```

### Utilisation étape par étape
```python
analyseur = AnalyseurPerformanceScolaire()

# 1. Générer ou charger les données
analyseur.generer_donnees_synthetiques(300)  # 300 élèves

# 2. Explorer les données
analyseur.exploration_donnees()

# 3. Nettoyer et visualiser
analyseur.nettoyer_donnees()
analyseur.visualiser_donnees()

# 4. Modéliser
analyseur.preparer_modelisation()
analyseur.entrainer_modeles()

# 5. Analyser les résultats
analyseur.analyser_importance_variables()

# 6. Générer le rapport
analyseur.generer_rapport()
```

## 📊 Structure des Données

### Variables d'entrée attendues

#### 🧑‍🎓 Facteurs Individuels
- `age`: Âge de l'élève
- `genre`: Genre (M/F)
- `note_francais`, `note_maths`, `note_lecture`: Notes par matière
- `absences`, `retards`: Indicateurs d'assiduité
- `heures_devoirs`, `heures_sommeil`: Temps de travail et repos
- `motivation`, `confiance_soi`, `stress`, `perseverance`: Facteurs psychologiques

#### 👨‍👩‍👧‍👦 Facteurs Familiaux
- `niveau_etudes_parents`: Niveau d'éducation des parents
- `revenus_famille`: Niveau socio-économique
- `suivi_parental`: Intensité du suivi parental
- `nombre_fratrie`: Nombre de frères et sœurs

#### 🏫 Facteurs Scolaires
- `taille_classe`: Nombre d'élèves par classe
- `type_etablissement`: Public/Privé
- `climat_scolaire`: Qualité de l'environnement scolaire
- `soutien_scolaire`: Aide supplémentaire

#### 🎨 Activités Extrascolaires
- `sport`, `musique`: Pratique d'activités
- `lecture_loisir`: Temps de lecture plaisir
- `temps_ecrans`: Temps passé sur écrans

### Variable de sortie
- `note_moyenne`: Note moyenne calculée (variable cible à prédire)

## 📈 Fonctionnalités

### 1. Exploration des Données (EDA)
- Statistiques descriptives complètes
- Détection de valeurs manquantes et aberrantes
- Analyse de la qualité des données

### 2. Préparation des Données
- Nettoyage automatique (imputation, correction des types)
- Ingénierie des variables (création de scores composites)
- Standardisation et encodage

### 3. Visualisations
- Distribution des performances
- Corrélations entre variables
- Impact des facteurs clés
- Identification des groupes à risque

### 4. Modélisation Prédictive
- **Régression Linéaire**: Modèle de base interprétable
- **Random Forest**: Modèle d'ensemble performant
- **XGBoost**: Modèle de gradient boosting haute performance
- Validation croisée et optimisation des hyperparamètres

### 5. Analyse des Résultats
- Importance des variables (Feature Importance)
- Métriques de performance (RMSE, MAE, R²)
- Interprétation pédagogique des résultats

### 6. Génération de Rapports
- Rapport de synthèse automatique (Markdown)
- Recommandations actionnables
- Export des données de test

## 📁 Fichiers Générés

Après l'exécution, le système génère automatiquement :

- `rapport_analyse_scolaire.md`: Rapport de synthèse complet
- `test_synthetique.xlsx`: Échantillon de données pour tests
- Graphiques et visualisations affichés

## 🎯 Métriques d'Évaluation

### Métriques de Performance
- **RMSE** (Root Mean Squared Error): Erreur quadratique moyenne
- **MAE** (Mean Absolute Error): Erreur absolue moyenne
- **R²** (Coefficient de détermination): Pourcentage de variance expliquée

### Interprétation des Seuils
- **R² > 0.7**: Modèle très performant, prédictions fiables
- **R² 0.5-0.7**: Modèle moyennement performant
- **R² < 0.5**: Modèle peu performant, à améliorer

## 📚 Recommandations Pédagogiques Types

Le système génère automatiquement des recommandations basées sur l'analyse :

1. **Optimisation du temps de travail personnel**
2. **Amélioration de l'hygiène de vie**
3. **Renforcement de l'engagement parental**
4. **Développement de la motivation et confiance**
5. **Prévention de l'absentéisme**

## 🔧 Personnalisation

### Adapter aux Données Spécifiques
```python
# Modifier les colonnes à exclure de la modélisation
colonnes_exclues = ['note_francais', 'note_maths', 'note_lecture', 'note_moyenne']

# Ajuster les variables pour les corrélations
colonnes_correlation = ['note_moyenne', 'facteur1', 'facteur2', 'facteur3']

# Personnaliser les seuils d'alerte
seuil_difficulte = 10  # Note en dessous de laquelle un élève est "en difficulté"
```

### Créer de Nouvelles Variables
```python
# Exemple : Score d'engagement
df['score_engagement'] = (df['motivation'] + df['perseverance']) / 2

# Ratio temps utile / temps total
df['efficacite_temps'] = df['heures_devoirs'] / (df['heures_devoirs'] + df['temps_ecrans'])
```

## 🐛 Dépannage

### Erreurs Courantes

**Erreur de chargement des données**
```python
# Vérifier le format du fichier
print(df.dtypes)
print(df.head())
```

**Erreur de mémoire**
```python
# Réduire la taille de l'échantillon
analyseur.generer_donnees_synthetiques(100)  # Au lieu de 300
```

**Erreur de modélisation**
```python
# Vérifier les valeurs manquantes
print(df.isnull().sum())
```

## 📞 Support

Pour toute question ou problème :
1. Vérifiez que toutes les dépendances sont installées
2. Consultez les messages d'erreur détaillés
3. Testez avec des données synthétiques d'abord
4. Adaptez les variables selon votre contexte spécifique

## 📄 Licence

Ce projet est développé pour des fins éducatives et de recherche. Libre d'utilisation avec attribution.

## 🔄 Versions

- **v1.0**: Version initiale avec fonctionnalités complètes
  - EDA automatisée
  - Modélisation multi-algorithmes
  - Génération de rapports
  - Données synthétiques intégrées

---

*Développé par un Data Scientist spécialisé en éducation*
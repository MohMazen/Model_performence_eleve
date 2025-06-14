#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse Prédictive des Performances Scolaires
===========================================

Ce script implémente une méthodologie complète d'analyse des données éducatives
pour prédire et comprendre les facteurs de réussite scolaire.

Auteur: Data Scientist Education
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AnalyseurPerformanceScolaire:
    """
    Classe principale pour l'analyse prédictive des performances scolaires
    """
    
    def __init__(self, fichier_donnees=None):
        """
        Initialise l'analyseur
        
        Args:
            fichier_donnees (str): Chemin vers le fichier Excel des données
        """
        self.donnees = None
        self.donnees_nettoyees = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.modeles = {}
        self.meilleur_modele = None
        self.preprocesseur = None
        
        if fichier_donnees:
            self.charger_donnees(fichier_donnees)
    
    def charger_donnees(self, fichier_donnees):
        """
        Charge les données depuis un fichier Excel
        
        Args:
            fichier_donnees (str): Chemin vers le fichier
        """
        try:
            self.donnees = pd.read_excel(fichier_donnees)
            print(f"✓ Données chargées avec succès: {self.donnees.shape}")
            return True
        except Exception as e:
            print(f"✗ Erreur lors du chargement: {e}")
            return False
    
    def generer_donnees_synthetiques(self, n_eleves=300):
        """
        Génère un jeu de données synthétiques réaliste pour les tests
        
        Args:
            n_eleves (int): Nombre d'élèves à générer
        """
        np.random.seed(42)
        
        # Variables individuelles
        donnees_synth = {
            # Données démographiques
            'age': np.random.normal(15, 1.5, n_eleves).clip(12, 18),
            'genre': np.random.choice(['M', 'F'], n_eleves),
            
            # Performance académique (variable cible)
            'note_francais': np.random.normal(13, 3, n_eleves).clip(0, 20),
            'note_maths': np.random.normal(12, 3.5, n_eleves).clip(0, 20),
            'note_lecture': np.random.normal(13.5, 2.8, n_eleves).clip(0, 20),
            
            # Facteurs comportementaux
            'absences': np.random.poisson(8, n_eleves).clip(0, 50),
            'retards': np.random.poisson(3, n_eleves).clip(0, 20),
            'heures_devoirs': np.random.gamma(2, 1.5, n_eleves).clip(0.5, 8),
            'heures_sommeil': np.random.normal(7.5, 1, n_eleves).clip(5, 10),
            'temps_ecrans': np.random.gamma(2, 2, n_eleves).clip(0, 12),
            
            # Facteurs psychologiques (échelle 1-10)
            'motivation': np.random.beta(3, 2, n_eleves) * 9 + 1,
            'confiance_soi': np.random.beta(2.5, 2.5, n_eleves) * 9 + 1,
            'stress': np.random.beta(2, 3, n_eleves) * 9 + 1,
            'perseverance': np.random.beta(3, 2, n_eleves) * 9 + 1,
            
            # Facteurs familiaux
            'niveau_etudes_parents': np.random.choice(['Primaire', 'Secondaire', 'Supérieur'], 
                                                    n_eleves, p=[0.2, 0.5, 0.3]),
            'revenus_famille': np.random.choice(['Faible', 'Moyen', 'Élevé'], 
                                              n_eleves, p=[0.3, 0.5, 0.2]),
            'suivi_parental': np.random.choice(['Faible', 'Modéré', 'Fort'], 
                                             n_eleves, p=[0.25, 0.5, 0.25]),
            'nombre_fratrie': np.random.poisson(1.5, n_eleves).clip(0, 6),
            
            # Facteurs scolaires
            'taille_classe': np.random.normal(25, 4, n_eleves).clip(15, 35),
            'type_etablissement': np.random.choice(['Public', 'Privé'], n_eleves, p=[0.8, 0.2]),
            'climat_scolaire': np.random.beta(3, 2, n_eleves) * 9 + 1,
            'soutien_scolaire': np.random.choice(['Oui', 'Non'], n_eleves, p=[0.3, 0.7]),
            
            # Activités extrascolaires
            'sport': np.random.choice(['Oui', 'Non'], n_eleves, p=[0.6, 0.4]),
            'musique': np.random.choice(['Oui', 'Non'], n_eleves, p=[0.4, 0.6]),
            'lecture_loisir': np.random.gamma(1.5, 2, n_eleves).clip(0, 10)
        }
        
        # Créer des corrélations réalistes
        df = pd.DataFrame(donnees_synth)
        
        # Ajuster les notes en fonction des facteurs influents
        for i in range(len(df)):
            bonus = 0
            
            # Impact du temps de devoirs
            if df.loc[i, 'heures_devoirs'] > 3:
                bonus += 1.5
            
            # Impact du sommeil
            if 7 <= df.loc[i, 'heures_sommeil'] <= 9:
                bonus += 1
            
            # Impact de la motivation
            bonus += (df.loc[i, 'motivation'] - 5) * 0.3
            
            # Impact du suivi parental
            if df.loc[i, 'suivi_parental'] == 'Fort':
                bonus += 1.2
            elif df.loc[i, 'suivi_parental'] == 'Faible':
                bonus -= 0.8
            
            # Impact du niveau d'études des parents
            if df.loc[i, 'niveau_etudes_parents'] == 'Supérieur':
                bonus += 1
            elif df.loc[i, 'niveau_etudes_parents'] == 'Primaire':
                bonus -= 0.8
            
            # Impact négatif du stress et du temps d'écrans
            bonus -= (df.loc[i, 'stress'] - 5) * 0.2
            bonus -= max(0, df.loc[i, 'temps_ecrans'] - 4) * 0.15
            
            # Impact négatif des absences
            bonus -= df.loc[i, 'absences'] * 0.05
            
            # Ajuster les notes
            df.loc[i, 'note_francais'] = np.clip(df.loc[i, 'note_francais'] + bonus, 0, 20)
            df.loc[i, 'note_maths'] = np.clip(df.loc[i, 'note_maths'] + bonus, 0, 20)
            df.loc[i, 'note_lecture'] = np.clip(df.loc[i, 'note_lecture'] + bonus, 0, 20)
        
        # Calculer la note moyenne
        df['note_moyenne'] = (df['note_francais'] + df['note_maths'] + df['note_lecture']) / 3
        
        # Créer des variables composites
        df['score_assiduite'] = 10 - (df['absences'] * 0.15 + df['retards'] * 0.25)
        df['score_assiduite'] = df['score_assiduite'].clip(0, 10)
        
        df['equilibre_vie'] = df['heures_sommeil'] - df['temps_ecrans'] * 0.3
        df['equilibre_vie'] = df['equilibre_vie'].clip(0, 10)
        
        self.donnees = df
        print(f"✓ Données synthétiques générées: {df.shape}")
        return df
    
    def exploration_donnees(self):
        """
        Effectue l'analyse exploratoire des données (EDA)
        """
        if self.donnees is None:
            print("✗ Aucune donnée chargée")
            return
        
        print("="*60)
        print("EXPLORATION ET ANALYSE DES DONNÉES")
        print("="*60)
        
        # Informations générales
        print(f"\n📊 APERÇU GÉNÉRAL")
        print(f"   • Nombre d'élèves: {len(self.donnees)}")
        print(f"   • Nombre de variables: {len(self.donnees.columns)}")
        print(f"   • Mémoire utilisée: {self.donnees.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Valeurs manquantes
        print(f"\n🔍 QUALITÉ DES DONNÉES")
        valeurs_manquantes = self.donnees.isnull().sum()
        if valeurs_manquantes.sum() > 0:
            print("   • Valeurs manquantes détectées:")
            for col, count in valeurs_manquantes[valeurs_manquantes > 0].items():
                pct = (count / len(self.donnees)) * 100
                print(f"     - {col}: {count} ({pct:.1f}%)")
        else:
            print("   • ✓ Aucune valeur manquante")
        
        # Statistiques descriptives
        print(f"\n📈 STATISTIQUES DESCRIPTIVES - PERFORMANCE")
        cols_notes = ['note_francais', 'note_maths', 'note_lecture', 'note_moyenne']
        stats_notes = self.donnees[cols_notes].describe()
        print(stats_notes.round(2))
        
        # Détection des valeurs aberrantes
        print(f"\n⚠️  DÉTECTION D'ANOMALIES")
        for col in ['note_moyenne', 'absences', 'heures_devoirs']:
            if col in self.donnees.columns:
                Q1 = self.donnees[col].quantile(0.25)
                Q3 = self.donnees[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = len(self.donnees[(self.donnees[col] < Q1 - 1.5*IQR) | 
                                          (self.donnees[col] > Q3 + 1.5*IQR)])
                print(f"   • {col}: {outliers} valeurs aberrantes potentielles")
    
    def nettoyer_donnees(self):
        """
        Nettoie et prépare les données pour la modélisation
        """
        print(f"\n🧹 NETTOYAGE DES DONNÉES")
        
        df = self.donnees.copy()
        
        # Gestion des valeurs manquantes
        colonnes_numeriques = df.select_dtypes(include=[np.number]).columns
        colonnes_categorielles = df.select_dtypes(include=['object']).columns
        
        # Imputation numérique (médiane)
        for col in colonnes_numeriques:
            if df[col].isnull().sum() > 0:
                mediane = df[col].median()
                df[col].fillna(mediane, inplace=True)
                print(f"   • {col}: {df[col].isnull().sum()} valeurs imputées (médiane)")
        
        # Imputation catégorielle (mode)
        for col in colonnes_categorielles:
            if df[col].isnull().sum() > 0:
                mode = df[col].mode()[0] if not df[col].mode().empty else 'Inconnu'
                df[col].fillna(mode, inplace=True)
                print(f"   • {col}: valeurs imputées (mode: {mode})")
        
        # Correction des types de données
        for col in df.columns:
            if 'age' in col.lower():
                df[col] = df[col].astype(int)
            elif any(word in col.lower() for word in ['note', 'score', 'heures']):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.donnees_nettoyees = df
        print(f"   ✓ Nettoyage terminé")
        
        return df
    
    def visualiser_donnees(self):
        """
        Crée des visualisations pour l'analyse exploratoire
        """
        if self.donnees_nettoyees is None:
            self.nettoyer_donnees()
        
        print(f"\n📊 GÉNÉRATION DES VISUALISATIONS")
        
        # Configuration
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Distribution des notes
        plt.subplot(2, 3, 1)
        self.donnees_nettoyees['note_moyenne'].hist(bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution des Notes Moyennes', fontsize=14, fontweight='bold')
        plt.xlabel('Note Moyenne')
        plt.ylabel('Fréquence')
        plt.axvline(self.donnees_nettoyees['note_moyenne'].mean(), color='red', linestyle='--', 
                   label=f'Moyenne: {self.donnees_nettoyees["note_moyenne"].mean():.1f}')
        plt.legend()
        
        # 2. Relation temps devoirs vs notes
        plt.subplot(2, 3, 2)
        plt.scatter(self.donnees_nettoyees['heures_devoirs'], self.donnees_nettoyees['note_moyenne'], 
                   alpha=0.6, color='green')
        plt.title('Temps de Devoirs vs Performance', fontsize=14, fontweight='bold')
        plt.xlabel('Heures de Devoirs/jour')
        plt.ylabel('Note Moyenne')
        
        # Ligne de tendance
        z = np.polyfit(self.donnees_nettoyees['heures_devoirs'], self.donnees_nettoyees['note_moyenne'], 1)
        p = np.poly1d(z)
        plt.plot(self.donnees_nettoyees['heures_devoirs'], p(self.donnees_nettoyees['heures_devoirs']), 
                "r--", alpha=0.8)
        
        # 3. Impact du suivi parental
        plt.subplot(2, 3, 3)
        suivi_notes = self.donnees_nettoyees.groupby('suivi_parental')['note_moyenne'].mean()
        suivi_notes.plot(kind='bar', color=['lightcoral', 'gold', 'lightgreen'])
        plt.title('Impact du Suivi Parental', fontsize=14, fontweight='bold')
        plt.xlabel('Niveau de Suivi Parental')
        plt.ylabel('Note Moyenne')
        plt.xticks(rotation=45)
        
        # 4. Corrélations principales
        plt.subplot(2, 3, 4)
        colonnes_correlation = ['note_moyenne', 'heures_devoirs', 'heures_sommeil', 
                              'motivation', 'absences', 'stress']
        corr_matrix = self.donnees_nettoyees[colonnes_correlation].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, cbar_kws={'shrink': .8})
        plt.title('Matrice de Corrélation', fontsize=14, fontweight='bold')
        
        # 5. Facteurs de risque
        plt.subplot(2, 3, 5)
        eleves_difficulte = self.donnees_nettoyees[self.donnees_nettoyees['note_moyenne'] < 
                                                 self.donnees_nettoyees['note_moyenne'].quantile(0.25)]
        facteurs_risque = {
            'Absences élevées': len(eleves_difficulte[eleves_difficulte['absences'] > 15]),
            'Sommeil insuffisant': len(eleves_difficulte[eleves_difficulte['heures_sommeil'] < 7]),
            'Temps écrans excessif': len(eleves_difficulte[eleves_difficulte['temps_ecrans'] > 6]),
            'Faible motivation': len(eleves_difficulte[eleves_difficulte['motivation'] < 5])
        }
        
        plt.bar(facteurs_risque.keys(), facteurs_risque.values(), color='orange', alpha=0.7)
        plt.title('Facteurs de Risque (25% élèves en difficulté)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Nombre d\'élèves')
        
        # 6. Performance par type d'établissement
        plt.subplot(2, 3, 6)
        perf_etablissement = self.donnees_nettoyees.groupby('type_etablissement')['note_moyenne'].mean()
        perf_etablissement.plot(kind='bar', color=['steelblue', 'darkorange'])
        plt.title('Performance par Type d\'Établissement', fontsize=14, fontweight='bold')
        plt.xlabel('Type d\'Établissement')
        plt.ylabel('Note Moyenne')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
        
        print("   ✓ Visualisations générées")
    
    def preparer_modelisation(self):
        """
        Prépare les données pour la modélisation (variables X et y)
        """
        if self.donnees_nettoyees is None:
            self.nettoyer_donnees()
        
        print(f"\n🔧 PRÉPARATION POUR LA MODÉLISATION")
        
        # Définir les variables explicatives (X) et la cible (y)
        colonnes_exclues = ['note_francais', 'note_maths', 'note_lecture', 'note_moyenne']
        X = self.donnees_nettoyees.drop(columns=colonnes_exclues)
        y = self.donnees_nettoyees['note_moyenne']
        
        # Identifier les colonnes numériques et catégorielles
        colonnes_numeriques = X.select_dtypes(include=[np.number]).columns.tolist()
        colonnes_categorielles = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"   • Variables numériques: {len(colonnes_numeriques)}")
        print(f"   • Variables catégorielles: {len(colonnes_categorielles)}")
        
        # Créer le preprocesseur
        preprocesseur = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), colonnes_numeriques),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), colonnes_categorielles)
            ])
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.preprocesseur = preprocesseur
        
        print(f"   • Échantillon d'entraînement: {len(X_train)} élèves")
        print(f"   • Échantillon de test: {len(X_test)} élèves")
        print(f"   ✓ Préparation terminée")
        
        return X_train, X_test, y_train, y_test
    
    def entrainer_modeles(self):
        """
        Entraîne et compare différents modèles de machine learning
        """
        if self.X_train is None:
            self.preparer_modelisation()
        
        print(f"\n🤖 ENTRAÎNEMENT DES MODÈLES")
        print("="*50)
        
        # Définir les modèles à tester
        modeles_config = {
            'Régression Linéaire': LinearRegression(),
            'Forêt Aléatoire': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }
        
        resultats = {}
        
        for nom, modele in modeles_config.items():
            print(f"\n🔄 Entraînement: {nom}")
            
            # Créer un pipeline avec preprocessing
            pipeline = Pipeline([
                ('preprocesseur', self.preprocesseur),
                ('modele', modele)
            ])
            
            # Validation croisée
            scores_cv = cross_val_score(pipeline, self.X_train, self.y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            rmse_cv = np.sqrt(-scores_cv)
            
            # Entraînement sur l'ensemble complet
            pipeline.fit(self.X_train, self.y_train)
            
            # Prédictions sur le test
            y_pred = pipeline.predict(self.X_test)
            
            # Métriques
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            resultats[nom] = {
                'pipeline': pipeline,
                'RMSE_CV': rmse_cv.mean(),
                'RMSE_Test': rmse,
                'MAE': mae,
                'R²': r2,
                'y_pred': y_pred
            }
            
            print(f"   • RMSE (Validation Croisée): {rmse_cv.mean():.3f} (±{rmse_cv.std():.3f})")
            print(f"   • RMSE (Test): {rmse:.3f}")
            print(f"   • MAE: {mae:.3f}")
            print(f"   • R²: {r2:.3f}")
        
        self.modeles = resultats
        
        # Sélectionner le meilleur modèle
        meilleur_nom = min(resultats.keys(), key=lambda x: resultats[x]['RMSE_Test'])
        self.meilleur_modele = resultats[meilleur_nom]['pipeline']
        
        print(f"\n🏆 MEILLEUR MODÈLE: {meilleur_nom}")
        print(f"   • RMSE: {resultats[meilleur_nom]['RMSE_Test']:.3f}")
        print(f"   • R²: {resultats[meilleur_nom]['R²']:.3f}")
        
        return resultats
    
    def analyser_importance_variables(self):
        """
        Analyse l'importance des variables du meilleur modèle
        """
        if self.meilleur_modele is None:
            print("✗ Aucun modèle entraîné")
            return
        
        print(f"\n🎯 ANALYSE DE L'IMPORTANCE DES VARIABLES")
        
        # Récupérer le modèle depuis le pipeline
        modele = self.meilleur_modele.named_steps['modele']
        preprocesseur = self.meilleur_modele.named_steps['preprocesseur']
        
        # Obtenir les noms des features après preprocessing
        feature_names = []
        
        # Features numériques
        num_features = preprocesseur.named_transformers_['num'].get_feature_names_out()
        feature_names.extend(num_features)
        
        # Features catégorielles
        if 'cat' in preprocesseur.named_transformers_:
            cat_features = preprocesseur.named_transformers_['cat'].get_feature_names_out()
            feature_names.extend(cat_features)
        
        # Extraire l'importance
        if hasattr(modele, 'feature_importances_'):
            importances = modele.feature_importances_
        elif hasattr(modele, 'coef_'):
            importances = np.abs(modele.coef_)
        else:
            print("   ⚠️ Impossible d'extraire l'importance des variables")
            return
        
        # Créer un DataFrame des importances
        importance_df = pd.DataFrame({
            'Variable': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Afficher le top 15
        print(f"\n📊 TOP 15 DES VARIABLES LES PLUS IMPORTANTES:")
        print("-" * 60)
        for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
            print(f"{i:2d}. {row['Variable']:<30} {row['Importance']:.4f}")
        
        # Visualisation
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        
        bars = plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue', alpha=0.8)
        plt.yticks(range(len(top_features)), top_features['Variable'])
        plt.xlabel('Importance')
        plt.title('Top 15 des Variables les Plus Importantes pour Prédire la Réussite Scolaire', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(top_features['Importance']) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def generer_rapport(self):
        """
        Génère un rapport de synthèse automatisé
        """
        print(f"\n📄 GÉNÉRATION DU RAPPORT DE SYNTHÈSE")
        print("="*60)
        
        if self.donnees is None:
            print("✗ Aucune donnée disponible")
            return
        
        rapport = []
        rapport.append("# RAPPORT D'ANALYSE - PERFORMANCES SCOLAIRES")
        rapport.append("=" * 60)
        rapport.append("")
        
        # Section 1: Résumé exécutif
        rapport.append("## 1. RÉSUMÉ EXÉCUTIF")
        rapport.append("")
        rapport.append(f"Cette analyse porte sur un échantillon de {len(self.donnees)} élèves.")
        
        if self.donnees_nettoyees is not None:
            note_moyenne = self.donnees_nettoyees['note_moyenne'].mean()
            note_mediane = self.donnees_nettoyees['note_moyenne'].median()
            note_std = self.donnees_nettoyees['note_moyenne'].std()
            
            rapport.append(f"- Note moyenne générale: {note_moyenne:.2f}/20")
            rapport.append(f"- Note médiane: {note_mediane:.2f}/20") 
            rapport.append(f"- Écart-type: {note_std:.2f}")
            
            # Identification des élèves en difficulté
            eleves_difficulte = len(self.donnees_nettoyees[self.donnees_nettoyees['note_moyenne'] < 10])
            pct_difficulte = (eleves_difficulte / len(self.donnees_nettoyees)) * 100
            rapport.append(f"- Élèves en difficulté (<10/20): {eleves_difficulte} ({pct_difficulte:.1f}%)")
        
        rapport.append("")
        
        # Section 2: Performance du modèle
        if self.modeles:
            rapport.append("## 2. PERFORMANCE DU MODÈLE PRÉDICTIF")
            rapport.append("")
            
            meilleur_nom = min(self.modeles.keys(), key=lambda x: self.modeles[x]['RMSE_Test'])
            meilleur_resultat = self.modeles[meilleur_nom]
            
            rapport.append(f"Le meilleur modèle identifié est: **{meilleur_nom}**")
            rapport.append("")
            rapport.append("### Métriques de performance:")
            rapport.append(f"- RMSE: {meilleur_resultat['RMSE_Test']:.3f} points")
            rapport.append(f"- MAE: {meilleur_resultat['MAE']:.3f} points")
            rapport.append(f"- R²: {meilleur_resultat['R²']:.3f} ({meilleur_resultat['R²']*100:.1f}% de variance expliquée)")
            rapport.append("")
            rapport.append("### Interprétation:")
            if meilleur_resultat['R²'] > 0.7:
                rapport.append("✓ Modèle très performant - prédictions fiables")
            elif meilleur_resultat['R²'] > 0.5:
                rapport.append("○ Modèle moyennement performant - prédictions modérément fiables")
            else:
                rapport.append("⚠ Modèle peu performant - prédictions à interpréter avec prudence")
        
        rapport.append("")
        
        # Section 3: Facteurs clés de réussite
        rapport.append("## 3. FACTEURS CLÉS DE RÉUSSITE")
        rapport.append("")
        
        if self.donnees_nettoyees is not None:
            # Corrélations importantes
            colonnes_correlation = ['note_moyenne', 'heures_devoirs', 'heures_sommeil', 
                                  'motivation', 'absences', 'stress']
            corr_avec_notes = self.donnees_nettoyees[colonnes_correlation].corr()['note_moyenne'].abs()
            top_correlations = corr_avec_notes.drop('note_moyenne').sort_values(ascending=False).head(5)
            
            rapport.append("### Variables les plus corrélées à la réussite:")
            for var, corr in top_correlations.items():
                rapport.append(f"- {var}: {corr:.3f}")
            
            rapport.append("")
            
            # Analyse par groupe
            rapport.append("### Analyse comparative par groupes:")
            
            # Suivi parental
            if 'suivi_parental' in self.donnees_nettoyees.columns:
                suivi_stats = self.donnees_nettoyees.groupby('suivi_parental')['note_moyenne'].agg(['mean', 'count'])
                rapport.append("")
                rapport.append("**Impact du suivi parental:**")
                for niveau, stats in suivi_stats.iterrows():
                    rapport.append(f"- {niveau}: {stats['mean']:.2f}/20 (n={stats['count']})")
            
            # Type d'établissement
            if 'type_etablissement' in self.donnees_nettoyees.columns:
                etab_stats = self.donnees_nettoyees.groupby('type_etablissement')['note_moyenne'].agg(['mean', 'count'])
                rapport.append("")
                rapport.append("**Performance par type d'établissement:**")
                for type_etab, stats in etab_stats.iterrows():
                    rapport.append(f"- {type_etab}: {stats['mean']:.2f}/20 (n={stats['count']})")
        
        rapport.append("")
        
        # Section 4: Recommandations
        rapport.append("## 4. RECOMMANDATIONS PÉDAGOGIQUES")
        rapport.append("")
        
        recommendations = [
            "### Actions prioritaires:",
            "",
            "1. **Optimiser le temps de travail personnel**",
            "   - Accompagner les élèves dans l'organisation de leurs devoirs",
            "   - Promouvoir des méthodes de travail efficaces",
            "   - Sensibiliser à l'importance d'un temps d'étude régulier",
            "",
            "2. **Améliorer l'hygiène de vie**",
            "   - Sensibiliser aux bienfaits d'un sommeil suffisant (7-9h)",
            "   - Réguler l'usage des écrans, particulièrement le soir",
            "   - Encourager un équilibre entre travail et loisirs",
            "",
            "3. **Renforcer l'engagement parental**",
            "   - Organiser des ateliers pour les parents sur l'accompagnement scolaire",
            "   - Faciliter la communication école-famille",
            "   - Proposer des ressources aux familles moins impliquées",
            "",
            "4. **Développer la motivation et la confiance**",
            "   - Mettre en place un système de reconnaissance des progrès",
            "   - Proposer un accompagnement personnalisé aux élèves en difficulté",
            "   - Organiser des activités valorisant les différents types de talents",
            "",
            "5. **Prévenir l'absentéisme**",
            "   - Identifier précocement les élèves à risque",
            "   - Mettre en place un suivi individualisé",
            "   - Développer des stratégies de remédiation rapide"
        ]
        
        rapport.extend(recommendations)
        rapport.append("")
        
        # Section 5: Méthodologie
        rapport.append("## 5. MÉTHODOLOGIE")
        rapport.append("")
        rapport.append("### Données analysées:")
        if self.donnees is not None:
            rapport.append(f"- Échantillon: {len(self.donnees)} élèves")
            rapport.append(f"- Variables: {len(self.donnees.columns)} indicateurs")
            rapport.append("- Catégories: facteurs individuels, familiaux, scolaires et contextuels")
        
        rapport.append("")
        rapport.append("### Modèles testés:")
        if self.modeles:
            for nom_modele in self.modeles.keys():
                rmse = self.modeles[nom_modele]['RMSE_Test']
                r2 = self.modeles[nom_modele]['R²']
                rapport.append(f"- {nom_modele}: RMSE={rmse:.3f}, R²={r2:.3f}")
        
        rapport.append("")
        rapport.append("### Validation:")
        rapport.append("- Validation croisée 5-folds sur l'ensemble d'entraînement")
        rapport.append("- Test final sur 20% des données (échantillon indépendant)")
        rapport.append("- Métriques: RMSE, MAE, R² pour évaluer la précision")
        
        rapport.append("")
        rapport.append("---")
        rapport.append("*Rapport généré automatiquement par le système d'analyse prédictive*")
        
        # Enregistrer le rapport
        contenu_rapport = "\n".join(rapport)
        
        try:
            with open('rapport_analyse_scolaire.md', 'w', encoding='utf-8') as f:
                f.write(contenu_rapport)
            print("   ✓ Rapport sauvegardé: rapport_analyse_scolaire.md")
        except Exception as e:
            print(f"   ⚠ Erreur lors de la sauvegarde: {e}")
        
        # Afficher un aperçu
        print(f"\n📋 APERÇU DU RAPPORT:")
        print("-" * 40)
        for ligne in rapport[:20]:  # Afficher les 20 premières lignes
            print(ligne)
        print("...")
        print(f"[Rapport complet: {len(rapport)} lignes]")
        
        return contenu_rapport
    
    def exporter_donnees_test(self, nom_fichier='test_synthetique.xlsx'):
        """
        Exporte un échantillon de données synthétiques pour les tests
        
        Args:
            nom_fichier (str): Nom du fichier de sortie
        """
        print(f"\n💾 EXPORT DES DONNÉES DE TEST")
        
        # Générer un petit échantillon de 30 élèves
        donnees_test = self.generer_donnees_synthetiques(30)
        
        try:
            donnees_test.to_excel(nom_fichier, index=False)
            print(f"   ✓ Fichier exporté: {nom_fichier}")
            print(f"   • {len(donnees_test)} élèves")
            print(f"   • {len(donnees_test.columns)} variables")
            
            # Aperçu des données
            print(f"\n📊 Aperçu des données exportées:")
            print(donnees_test[['age', 'note_moyenne', 'heures_devoirs', 'motivation', 'suivi_parental']].head())
            
        except Exception as e:
            print(f"   ✗ Erreur lors de l'export: {e}")
        
        return donnees_test
    
    def analyse_complete(self, generer_donnees=True):
        """
        Lance l'analyse complète de bout en bout
        
        Args:
            generer_donnees (bool): Si True, génère des données synthétiques
        """
        print("🎯 LANCEMENT DE L'ANALYSE COMPLÈTE")
        print("=" * 70)
        
        try:
            # 1. Données
            if generer_donnees:
                self.generer_donnees_synthetiques()
            
            # 2. Exploration
            self.exploration_donnees()
            
            # 3. Nettoyage et visualisation
            self.nettoyer_donnees()
            self.visualiser_donnees()
            
            # 4. Modélisation
            self.preparer_modelisation()
            self.entrainer_modeles()
            
            # 5. Analyse des résultats
            self.analyser_importance_variables()
            
            # 6. Rapport et export
            self.generer_rapport()
            self.exporter_donnees_test()
            
            print(f"\n🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
            print("=" * 50)
            print("📁 Fichiers générés:")
            print("   • rapport_analyse_scolaire.md")
            print("   • test_synthetique.xlsx")
            print("   • Graphiques affichés")
            
        except Exception as e:
            print(f"\n❌ ERREUR LORS DE L'ANALYSE: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Fonction principale pour exécuter l'analyse
    """
    print("🏫 SYSTÈME D'ANALYSE PRÉDICTIVE DES PERFORMANCES SCOLAIRES")
    print("=" * 70)
    print("Version 1.0 - Spécialisé pour l'éducation")
    print()
    
    # Créer l'analyseur
    analyseur = AnalyseurPerformanceScolaire()
    
    # Option 1: Charger des données existantes
    # analyseur.charger_donnees('joins.xlsx')
    
    # Option 2: Utiliser des données synthétiques (recommandé pour la démonstration)
    print("🔄 Démarrage de l'analyse avec des données synthétiques...")
    print()
    
    # Lancer l'analyse complète
    analyseur.analyse_complete(generer_donnees=True)
    
    print()
    print("📚 GUIDE D'UTILISATION:")
    print("1. Remplacez les données synthétiques par vos vraies données")
    print("2. Adaptez les variables selon votre contexte")
    print("3. Ajustez les seuils dans les recommandations")
    print("4. Personnalisez les visualisations selon vos besoins")
    print()
    print("💡 Pour plus d'aide, consultez la documentation dans le README.md")


if __name__ == "__main__":
    main()
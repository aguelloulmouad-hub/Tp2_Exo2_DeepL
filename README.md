# TP Deep Learning Avancé – Exercice 2 : Image Captioning avec CNN-LSTM et Attention

Ce dépôt contient le notebook Jupyter complet de l’exercice 2 du module **Deep Learning Avancé** (Master AIDC, FST Béni Mellal, Université Sultan Moulay Slimane).  
L’objectif est de construire un système d’**Image Captioning** capable de générer automatiquement une description textuelle d’une image, en combinant un encodeur CNN (ResNet50) et un décodeur LSTM, puis d’améliorer le modèle avec un **mécanisme d’attention visuelle de type Bahdanau**.

## 📋 Contenu

- `TP_Captioning_Ex2_kaggle.ipynb` : notebook exécuté sur **Kaggle** avec deux GPU Tesla T4.
- Toutes les étapes sont détaillées : chargement et prétraitement du dataset MS-COCO 2017, construction du vocabulaire, implémentation des modèles, entraînement, évaluation (BLEU, METEOR, CIDEr, perplexité) et visualisation des cartes d’attention.

## 🚀 Principales fonctionnalités

- **Dataset** : sous-ensemble de MS-COCO 2017 (5 000 images train, 500 images val, 5 légendes par image).
- **Vocabulaire** : construction avec seuil de fréquence 5 (taille finale 2 367 tokens).
- **Modèle de base** : encodeur ResNet50 (gelé partiel) + LSTM 2 couches (teacher forcing).
- **Modèle avec attention** : encodeur spatial (`7×7` grille de caractéristiques) + cellule LSTM avec attention additive de Bahdanau.
- **Génération** : greedy et beam search (beam size=3).
- **Métriques** : BLEU-1, BLEU-4, METEOR, CIDEr, perplexité.
- **Visualisation** : cartes d’attention superposées aux images originales.

## 📊 Résultats clés

| Modèle               | BLEU-1 | BLEU-4 | METEOR | CIDEr | Val PPL | Nb paramètres (M) |
|----------------------|--------|--------|--------|-------|---------|-------------------|
| CNN-LSTM (sans attention) | 0.5534 | 0.0000* | 0.1393 | 0.3261 | 37.1    | 29.5              |
| CNN-LSTM + Attention | **0.6735** | **0.2348** | **0.2152** | **0.7399** | **19.0** | 33.9              |

*Le BLEU-4 du modèle de base est quasi nul car aucun 4-gramme ne correspond exactement aux références humaines (score lissé ~0.006).*

L’ajout de l’attention améliore significativement toutes les métriques, avec une baisse de la perplexité de 49 % et un triplement du score CIDEr.

## 🛠 Prérequis

- Python 3.10+
- PyTorch ≥ 2.0, torchvision
- pycocotools, nltk, matplotlib, seaborn, tqdm, Pillow
- Connexion Internet pour télécharger les annotations et images COCO (ou les avoir localement)

## ▶️ Exécution

Le notebook est conçu pour être exécuté sur **Kaggle** (GPU T4 ×2) ou sur une machine avec GPU.  
Les principales cellules :

1. Installation des dépendances (`pycocotools`, `nltk`, etc.)
2. Téléchargement automatique des annotations COCO et d’un sous-ensemble d’images.
3. Construction du vocabulaire et visualisation des distributions.
4. Définition des modèles (encodeur, décodeur LSTM, attention).
5. Entraînement (20 époques pour le modèle de base, 15 pour le modèle avec attention).
6. Évaluation et génération des légendes.
7. Visualisation des cartes d’attention.

> ⚠️ Le téléchargement des images COCO peut prendre quelques minutes (environ 500 Mo).  
> Les poids des modèles (`best_captioning_basic.pth` et `best_captioning_attention.pth`) ne sont pas inclus dans le dépôt pour des raisons de taille.


## 📝 Licence & crédits

- TP réalisé dans le cadre du cours **Deep Learning Avancé** (Pr. Y. Saadi, FST Béni Mellal, USMS).
- Dataset : [MS-COCO 2017](https://cocodataset.org/).
- Code basé sur PyTorch et les architectures classiques d’image captioning.

## 🔗 Référence

- Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015.
- Vinyals et al., *Show and Tell: A Neural Image Caption Generator*, CVPR 2015.
- Xu et al., *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention*, ICML 2015.

---

*Notebook créé et exécuté par Aguelloul Mouad – Master AIDC 2025‑2026*  
*Dépôt GitHub : [https://github.com/aguelloulmouad-hub/Tp2_Exo2_DeepL](https://github.com/aguelloulmouad-hub/Tp2_Exo2_DeepL)*

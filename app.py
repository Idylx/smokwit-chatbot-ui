import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Smokwit - Votre Accompagnement pour Arrêter de Fumer", layout="centered")

# En-tête
st.title("🚬 Smokwit : L'Accompagnement Personnalisé pour l'Arrêt du Tabac")

# Introduction
st.markdown(
    """
    Bienvenue sur **Smokwit**, une plateforme innovante conçue pour accompagner les fumeurs dans leur démarche d'arrêt du tabac. 
    Grâce à une combinaison unique d'intelligence artificielle et de soutien humain, Smokwit offre un accompagnement personnalisé et adapté aux besoins de chacun.
    """
)

# Présentation des deux chatbots
st.header("🤖 Trois Chatbots Complémentaires : Expert, Paire et l'Oiseau")

## Chatbot Expert
st.subheader("📚 Expert")
st.markdown(
    """
    Le chatbot expert repose sur le modèle des **5A** (Ask, Advise, Assess, Assist, Arrange). Il vous aide à :
    - Évaluer votre niveau de dépendance
    - Recevoir des conseils fondés sur les meilleures pratiques médicales
    - Accéder à des ressources adaptées à votre profil
    """
)

## Chatbot Peer
st.subheader("🗣️ Peer")
st.markdown(
    """
    L'expérience des autres est une source précieuse de motivation ! Le chatbot Peer vous permet :
    - D'échanger avec un compagnon virtuel qui partage des témoignages inspirants
    - D'obtenir des astuces pratiques issues d'expériences réelles
    - De bénéficier d’un soutien émotionnel et motivant pour surmonter les difficultés
    """
)

## Chatbot Peer
st.subheader("🐦 L'oiseau")
st.markdown(
    """
    Un tout petit oiseau vient d’éclore… et il a déjà été exposé à la fumée !
    Dans cette partie gamifiée, le joueur doit s’occuper d’un oisillon fragile, lui offrir un environnement sain, l’aider à résister à l’appel de la cigarette et le guider vers une vie sans dépendance.
    Chaque action compte : attention, négligence ou rechute auront un impact direct sur sa santé, son humeur et son développement.
    """
)

# Pourquoi choisir Smokwit ?
st.header("✅ Pourquoi choisir Smokwit ?")
st.markdown(
    """
    - **Un accompagnement accessible 24/7** : Nos chatbots sont disponibles à tout moment pour vous aider.
    - **Une approche adaptée à votre rythme** : Que vous soyez prêt à arrêter aujourd’hui ou encore hésitant, nous adaptons nos conseils à votre progression.
    - **Des recommandations basées sur la science** : Notre chatbot expert utilise des stratégies validées scientifiquement pour maximiser vos chances de succès.
    - **Un soutien motivant et bienveillant** : Grâce au chatbot peer, vous bénéficiez d’un appui émotionnel pour rester motivé.
    """
)

# Call to Action
st.success("Rejoignez-nous sur **Smokwit** et faites le premier pas vers une vie sans tabac ! 🚀")

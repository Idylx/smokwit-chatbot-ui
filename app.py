import streamlit as st

# Configuration de la page
st.set_page_config(page_title="Smokwit - Votre Accompagnement pour Arrêter de Fumer", layout="centered")

# En-tête
st.title("🚬 Smokwit : L'Accompagnement Personnalisé pour l'Arrêt du Tabac")

# Introduction
st.markdown(
    """
    Bienvenue sur **Smokwit**, une plateforme conçue pour accompagner les fumeurs dans leur arrêt du tabac. 
    Grâce à une combinaison unique d'intelligence artificielle et de soutien humain, Smokwit vous accompagne afin de vous aider à arrêter de fumerr.
    """
)

# Présentation des deux chatbots
st.header("🤖 4 Chatbots  : Expert, un ex-fumeur, un fumeur et l'Oiseau (en test")

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
st.subheader("🗣️ Ex-fumeur")
st.markdown(
    """
    L'expérience des autres est une source précieuse de motivation ! Le chatbot Ex-fumeur vous permet :
    - D'échanger avec un compagnon virtuel qui partage des témoignages inspirants
    - D'obtenir des astuces pratiques issues d'expériences réelles
    - De bénéficier d’un soutien émotionnel et motivant pour surmonter les difficultés
    """
)

## Chatbot Fumeur
st.subheader("🗣️ Fumeur")
st.markdown(
    """
    Se sentir compris et ne pas être seul dans la même galère ! Le chatbot Fumeur vous permet :
    - D'échanger avec un fumeur qui souhaite arrêter, mais qui a encore des rechutes
    - Echanger des conseil et des astuces pratiques pour surmonter les envies
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


# Call to Action
st.success("Rejoignez-nous sur **Smokwit** et faites le premier pas vers une vie sans tabac ! 🚀")

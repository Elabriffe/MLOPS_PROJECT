import streamlit as st
import requests
import json  # en haut du fichier si ce n'est pas déjà fait

# Titre principal
st.set_page_config(page_title="Prédiction & Recommandations", layout="centered")
st.title("📤 Page de recommandations")

# === FORMULAIRE DE SOUMISSION ===
with st.form("upload_form"):
    id = st.number_input("ID", min_value=0, step=1)
    designation = st.text_input("Désignation")
    description = st.text_area("Description")
    productid = st.number_input("Product ID", min_value=0, step=1)
    imageid = st.number_input("Image ID", min_value=0, step=1)
    uploaded_file = st.file_uploader("Choisir un fichier", type=["png", "jpg", "jpeg", "pdf"])
    submit_button = st.form_submit_button("📨 Envoyer")

# === ENVOI VERS API ===
if submit_button:
    if uploaded_file is not None:
        files = {
            "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
        }
        data = {
            "id": str(id),
            "designation": designation,
            "description": description,
            "productid": str(productid),
            "imageid": str(imageid)
        }

        url = "http://host.docker.internal:8000/prediction"
        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            st.success("✅ Fichier envoyé avec succès !")
            result = response.json()

            # Affichage du score
            if "prediction" in result:
                classe = result["prediction"]
                st.markdown(
                    f"<h2 style='color:#4CAF50;'>🔮 Classe de prédiction : <strong>{classe}</strong></h2>",
                    unsafe_allow_html=True
                )

            # Affichage des recommandations
            if "recommandation" in result:
    # Convertir la chaîne JSON en liste d'objets Python
                if isinstance(result["recommandation"], str):
                    recs = json.loads(result["recommandation"])
                else:
                    recs = result["recommandation"]

                st.subheader("📦 Produits Recommandés")
                for rec in recs:
                    with st.container():
                        filename = f"image_{rec['imageid']}_product_{rec['productid']}.jpg"
                        st.markdown(f"### 🧸 {rec['designation']}")
                        if rec.get("description"):
                            st.markdown(f"**Description :** {rec['description'][:200]}{'...' if len(rec['description']) > 200 else ''}")
                        st.markdown(f"**ID produit :** `{rec['productid']}`")
                        st.markdown(f"**ID image produit :** `{rec['imageid']}`")
                        st.subheader("Image du produit : ")
                        st.image(f"../../src/data/raw/image_train/{filename}", caption=filename, width=500) # Changer le chemin du répertoire en fonction de la structure du projet (répertoire contenant toutes les photos)
                        st.markdown("---")
        else:
            st.error(f"❌ Erreur : {response.status_code}")
            st.text(response.text)
    else:
        st.warning("⚠️ Veuillez sélectionner un fichier avant d'envoyer.")

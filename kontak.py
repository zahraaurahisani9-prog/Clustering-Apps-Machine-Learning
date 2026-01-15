import streamlit as st


# ==================================================
# UI CARD HELPER
# ==================================================
def card(title, description=""):
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# CONTACT PAGE
# ==================================================
def contact_me():

    card(
        "📬 Kontak & Kolaborasi",
        "Jika Anda memiliki pertanyaan, masukan, atau ingin berdiskusi "
        "lebih lanjut terkait aplikasi ini, silakan hubungi saya melalui "
        "kanal berikut."
    )

    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 1.5rem 0;">
        <div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 8px; color: white;">
            <h4 style="margin: 0; color: white;">✉️ Email</h4>
            <p style="margin: 0.5rem 0 0 0;"><a href="mailto:zahraaurahisani9@gmail.com" style="text-decoration: none; color: white; font-weight: 500;">zahraaurahisani9@gmail.com</a></p>
        </div>
        <div class="card" style="background: linear-gradient(135deg, #0a66c2 0%, #005aa7 100%); padding: 1.5rem; border-radius: 8px; color: white;">
            <h4 style="margin: 0; color: white;">in LinkedIn</h4>
            <p style="margin: 0.5rem 0 0 0;"><a href="https://www.linkedin.com/in/zahra-aura-hisani" target="_blank" style="text-decoration: none; color: white; font-weight: 500;">Zahra Aura Hisani</a></p>
        </div>
        <div class="card" style="background: linear-gradient(135deg, #333333 0%, #1a1a1a 100%); padding: 1.5rem; border-radius: 8px; color: white;">
            <h4 style="margin: 0; color: white;">⚡ GitHub</h4>
            <p style="margin: 0.5rem 0 0 0;"><a href="https://github.com/zahraaurahisani9-prog" target="_blank" style="text-decoration: none; color: white; font-weight: 500;">zahraaurahisani9-prog</a></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    card(
        "🤝 Kolaborasi",
        "Saya terbuka untuk diskusi dan kolaborasi dalam bidang "
        "data science, machine learning, dan analisis data."
    )

    st.success(
        "Terima kasih telah menggunakan Clustering Analysis Platform."
    )

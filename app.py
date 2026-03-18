import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Sayfa ayarları
st.set_page_config(page_title="Dental AI", page_icon="🦷")
st.title("🦷 Dental Anomaly Detection")
st.write("Panoramik röntgen yükleyin, yapay zeka anomalileri tespit etsin.")

# Modeli yükle
@st.cache_resource
def load_model():
    # best.pt dosyasının app.py ile aynı klasörde olduğundan eminiz
    return YOLO("best.pt")

model = load_model()

# Dosya yükleyici
file = st.file_uploader("Röntgen görseli seçin...", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file)
    results = model(img) # Tahmin yap
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Orijinal Röntgen", use_container_width=True)
    with col2:
        # Sonuçları görselleştir ve ekrana bas
        res_plotted = results[0].plot() 
        st.image(res_plotted, caption="Yapay Zeka Analizi", use_container_width=True)
        
    st.success(f"Analiz tamamlandı! {len(results[0].boxes)} adet bulgu tespit edildi.")
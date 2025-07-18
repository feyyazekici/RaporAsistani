import os
import shutil
import tempfile
from typing import Optional

# LangChain ve Supabase tiplerini import ediyoruz.
from langchain_community.vectorstores import FAISS
from supabase import Client
import streamlit as st

# Kova adını (bucket name) burada sabit olarak tanımlıyoruz.
# Projenizdeki bucket adıyla eşleştiğinden emin olun.
BUCKET_NAME = "faissfiles" 

# --- Fonksiyon 1: FAISS'i Supabase'e Kaydetme ---
def save_faiss_to_supabase(
    vector_store: FAISS,
    storage_path: str,
    supabase_client: Client  # <-- DÜZELTME: Supabase istemcisini parametre olarak alıyoruz.
) -> Optional[str]:
    """
    FAISS indexini geçici bir klasöre kaydeder, bu klasörü zip'ler ve Supabase'e yükler.
    Bu yöntem, embedding modelini dahil etmeyerek dosya boyutunu küçültür.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Vektör deposunu geçici klasöre kaydet (model olmadan)
            vector_store.save_local(temp_dir)
            
            # Geçici klasörü zip dosyası haline getir (daha basit yöntem)
            zip_path_base = os.path.join(tempfile.gettempdir(), "faiss_temp_zip")
            shutil.make_archive(zip_path_base, 'zip', temp_dir)
            zip_file_path = f"{zip_path_base}.zip"

            # Zip dosyasını Supabase Storage'a yükle
            with open(zip_file_path, 'rb') as f:
                # Parametre olarak gelen supabase_client'ı kullanıyoruz
                supabase_client.storage.from_(BUCKET_NAME).upload(
                    file=f,
                    path=storage_path,
                    file_options={"content-type": "application/zip", "upsert": "true"}
                )
            
            # Oluşturulan geçici zip dosyasını temizle
            os.remove(zip_file_path)
            # st.toast("Vektör veritabanı başarıyla kaydedildi.") # İsteğe bağlı
            return storage_path

        except Exception as e:
            st.error(f"FAISS verisi Supabase Storage'a kaydedilemedi: {e}")
            return None

# --- Fonksiyon 2: FAISS'i Supabase'den Yükleme ---
def load_faiss_from_supabase(
    storage_path: str,
    supabase_client: Client,  # <-- DÜZELTME: Supabase istemcisini parametre olarak alıyoruz.
    embeddings_model       # <-- DÜZELTME: Embedding modelini parametre olarak alıyoruz.
) -> Optional[FAISS]:
    """
    Vektör deposunu Supabase Storage'dan indirir, zip'ten çıkarır ve yükler.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Parametre olarak gelen supabase_client'ı kullanıyoruz
            zip_bytes = supabase_client.storage.from_(BUCKET_NAME).download(storage_path)
            
            # İndirilen içeriği geçici bir zip dosyasına yaz
            temp_zip_path = os.path.join(temp_dir, "downloaded_faiss.zip")
            with open(temp_zip_path, "wb") as f:
                f.write(zip_bytes)
            
            # Zip dosyasını açmak için bir klasör oluştur
            extract_path = os.path.join(temp_dir, "faiss_index")
            os.makedirs(extract_path, exist_ok=True)
            shutil.unpack_archive(temp_zip_path, extract_path)

            # Yerel dosyalardan FAISS'i yüklerken, parametre olarak gelen modeli kullan
            return FAISS.load_local(
                extract_path, 
                embeddings_model,  # Dışarıdan gelen modeli kullan
                allow_dangerous_deserialization=True # Modern LangChain için gerekli
            )
            
        except Exception as e:
            # st.error(f"FAISS verisi Supabase Storage'dan yüklenemedi: {e}") # Ana app'te hata göstermek daha iyi
            print(f"FAISS verisi Supabase Storage'dan yüklenemedi: {storage_path}, Hata: {e}")
            return None
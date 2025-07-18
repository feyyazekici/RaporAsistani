# -*- coding: utf-8 -*-
"""
Kurumsal Rapor Sorgulama Asistanı - v7.0 (Yüksek Performanslı RAG)

Özellikler:
- Supabase-native Kullanıcı Yönetimi (Giriş/Kayıt).
- Multi-Modal RAG (Metin, Tablo, GÖRSEL) Analizi ile Üst Düzey Performans.
- Kaynak Gösterme (Source Citation) ile Güvenilir Cevaplar.
- Sohbet Yönetimi (Yeniden Adlandır/Sil).
- Stabil ve Ölçeklenebilir FAISS Depolama (.zip metodu).
"""
import os
import requests
import streamlit as st
import tempfile
import uuid
import bcrypt
import io
import shutil
import base64
import fitz  # PyMuPDF (Görsel analizi için)

# Yerel modülümüz ve Supabase istemcisini import ediyoruz
from faiss_storage import save_faiss_to_supabase, load_faiss_from_supabase
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

# Gerekli LangChain ve diğer kütüphaneler
from pypdf import PdfReader
import camelot
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 1. SAYFA YAPILANDIRMASI VE GİZLİ BİLGİLER ---
st.set_page_config(page_title="Akıllı Rapor Asistanı", layout="wide", initial_sidebar_state="expanded")

try:
    
    ABSTRACT_API_KEY = st.secrets["connections"]["abstract_api_key"]
    GOOGLE_API_KEY = st.secrets["connections"]["google_api_key"]
    N8N_WEBHOOK_URL = st.secrets["connections"]["n8n_webhook_url"]
    SUPABASE_URL = st.secrets["connections"]["supabase_url"]
    SUPABASE_KEY = st.secrets["connections"]["supabase_key"]
except (KeyError, FileNotFoundError):
    st.error("Lütfen `secrets.toml` dosyanızı ve içindeki tüm anahtarları kontrol edin.")
    st.stop()

# --- 2. SUPABASE BAĞLANTISI ---
try:
    
    options = ClientOptions(postgrest_client_timeout=120, storage_client_timeout=120)
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=options)
except Exception as e:
    st.error(f"Supabase bağlantısı kurulamadı: {e}"); st.stop()


# --- 3. YARDIMCI FONKSİYONLAR ---

@st.cache_resource(show_spinner="Embedding modeli yükleniyor...")
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def verify_email_realness(email: str) -> bool:
    """Verilen e-postanın gerçekliğini Abstract API ile kontrol eder."""
    try:
        response = requests.get(
            f"https://emailvalidation.abstractapi.com/v1/?api_key={ABSTRACT_API_KEY}&email={email}"
        )
        data = response.json()
        # API'den gelen 'DELIVERABLE' veya 'RISKY' sonuçlarını geçerli sayabiliriz.
        # 'UNDELIVERABLE' ise geçersizdir.
        if data.get("is_smtp_valid", {}).get("value") and data.get("deliverability") != "UNDELIVERABLE":
            return True
    except Exception as e:
        print(f"E-posta doğrulama API hatası: {e}")
        # API çalışmazsa, en azından format doğruysa geçmesine izin verelim.
        return "@" in email and "." in email.split('@')[1]
    return False

    
    # 2. Kullanıcı adı/e-posta veritabanında var mı diye kontrol et
    try:
        if supabase.table('users').select('id').or_(f"username.eq.{username},email.eq.{email}").execute().data:
            st.warning("Bu kullanıcı adı veya e-posta adresi zaten kullanımda."); return False
        
        # 3. Kullanıcıyı kaydet
        hashed_password = hash_password(password)
        supabase.table('users').insert({"name": name, "email": email, "username": username, "hashed_password": hashed_password}).execute(); return True
    except Exception as e: 
        st.error(f"Kayıt sırasında bir veritabanı hatası oluştu: {e}"); return False

# YÜKSEK PERFORMANSLI RAG İÇİN GÜNCELLENMİŞ FONKSİYON
def process_and_store_pdf(uploaded_file, username):
    with st.spinner(f"`{uploaded_file.name}` analiz ediliyor..."):
        file_bytes = uploaded_file.getvalue()
        documents = []
        # Metin ve Tablo çıkarma...
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for i, page in enumerate(reader.pages):
                if text := page.extract_text():
                    documents.append(Document(page_content=text, metadata={"page": i + 1}))
        except Exception as e: st.warning(f"PyPDF metin okuma hatası: {e}")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes); temp_path = tmp.name
            tables = camelot.read_pdf(temp_path, pages="all", flavor="lattice")
            for table in tables:
                documents.append(Document(page_content=f"--- Tablo (Sayfa {table.page}) ---\n{table.df.to_markdown(index=False)}\n---", metadata={"page": table.page}))
        except Exception: pass
        finally:
            if 'temp_path' in locals() and os.path.exists(temp_path): os.remove(temp_path)

        # GÖRSEL ÇIKARMA VE DEPOLAMA (PERFORMANS ARTIŞI)
        image_paths = []
        base_storage_path = f"{username}/{uuid.uuid4()}"
        try:
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    for img_index, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        image_storage_path = f"{base_storage_path}/image_{page_num + 1}_{img_index}.{image_ext}"
                        supabase.storage.from_("faissfiles").upload(file=image_bytes, path=image_storage_path, file_options={"content-type": f"image/{image_ext}"})
                        image_paths.append(image_storage_path)
        except Exception as e: st.warning(f"Görsel çıkarılırken hata oluştu: {e}")

        if not documents and not image_paths:
            st.error("Bu PDF'ten metin, tablo veya görsel çıkarılamadı."); return

        remote_filename = None
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
            chunks = text_splitter.split_documents(documents)
            embeddings = get_embeddings_model()
            vector_store = FAISS.from_documents(chunks, embeddings)
            faiss_zip_path = f"{base_storage_path}/faiss_index.zip"
            remote_filename = save_faiss_to_supabase(vector_store, faiss_zip_path, supabase)

        # DB kaydını güncelle (image_paths eklendi)
        supabase.table('conversations').insert({
            "username": username, "conversation_name": uploaded_file.name,
            "storage_path": remote_filename, "image_paths": image_paths
        }).execute()
        st.success(f"`{uploaded_file.name}` başarıyla analiz edildi.")
        st.rerun()

# DB fonksiyonlarını güncelle (image_paths eklendi)
def load_user_conversations(username):
    return supabase.table('conversations').select('id, conversation_name, storage_path, image_paths').eq('username', username).order('id', desc=True).execute().data

# ... diğer DB ve Auth fonksiyonları aynı kalıyor ...
def load_messages(conversation_id):
    response = supabase.table('messages').select('role, content').eq('conversation_id', conversation_id).order('id', desc=False).execute()
    return [{"role": row['role'], "content": row['content']} for row in response.data]
def save_message(conversation_id, role, content):
    supabase.table('messages').insert({"conversation_id": conversation_id, "role": role, "content": content}).execute()
def send_log_to_n8n(username, doc_name, question, answer):
    try: requests.post(N8N_WEBHOOK_URL, json={"username": username, "document_name": doc_name, "question": question, "answer": answer}, timeout=5)
    except Exception as e: print(f"n8n log hatası: {e}")
def rename_conversation(conversation_id, new_name):
    try: supabase.table('conversations').update({'conversation_name': new_name}).eq('id', conversation_id).execute(); st.toast("Sohbet yeniden adlandırıldı!"); return True
    except Exception as e: st.error(f"Sohbet yeniden adlandırılamadı: {e}"); return False
def delete_conversation(conversation_id, storage_path, image_paths):
    try:
        supabase.table('conversations').delete().eq('id', conversation_id).execute()
        files_to_delete = []
        if storage_path: files_to_delete.append(storage_path)
        if image_paths: files_to_delete.extend(image_paths)
        if files_to_delete: supabase.storage.from_("faissfiles").remove(files_to_delete)
        st.toast("Sohbet başarıyla silindi."); return True
    except Exception as e: st.error(f"Sohbet silinemedi: {e}"); return False
def hash_password(password: str): return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
def check_password(password: str, hashed: str): return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
def validate_login(username, password):
    try:
        user_data = supabase.table('users').select('name, hashed_password').eq('username', username).single().execute().data
        if user_data and check_password(password, user_data['hashed_password']):
            st.session_state.update({"authentication_status": True, "username": username, "name": user_data['name']}); return True
    except Exception: return False


def register_user(name, email, username, password):
    """Yeni kullanıcıyı Supabase'e kaydeder. (Kontroller dışarıda yapılır)"""
    try:
        if supabase.table('users').select('id').or_(f"username.eq.{username},email.eq.{email}").execute().data:
            st.warning("Bu kullanıcı adı veya e-posta adresi zaten kullanımda."); return False
        
        hashed_password = hash_password(password)
        supabase.table('users').insert({"name": name, "email": email, "username": username, "hashed_password": hashed_password}).execute(); return True
    except Exception as e: 
        st.error(f"Kayıt sırasında bir veritabanı hatası oluştu: {e}"); return False

# --- 5. ANA UYGULAMA AKIŞI ---
st.title("PDF RAG Sistemli Rapor Asistanı ")

if not st.session_state.get("authentication_status"):
    
    st.info("Lütfen devam etmek için giriş yapın veya yeni bir hesap oluşturun.")
    login_tab, register_tab = st.tabs(["Giriş Yap", "Yeni Hesap Oluştur"])
    with login_tab:
        with st.form("login_form", border=False):
            username_login = st.text_input("Kullanıcı Adı")
            password_login = st.text_input("Şifre", type="password")
            if st.form_submit_button("Giriş Yap"):
                if validate_login(username_login, password_login): st.rerun()
                else: st.error("Kullanıcı adı veya şifre yanlış")
    with register_tab:
        with st.form("register_form", border=False):
            st.write("Yeni bir hesap oluşturun.") 
            name_reg = st.text_input("Ad Soyad")
            email_reg = st.text_input("E-posta Adresi")
            username_reg = st.text_input("Kullanıcı Adı")
            password_reg = st.text_input("Şifre", type="password")
            
            # Form gönderim butonu
            submitted = st.form_submit_button("Hesap Oluştur")
            
            if submitted:
                # 1. Tüm alanların dolu olup olmadığını kontrol et
                if not all([name_reg, email_reg, username_reg, password_reg]):
                    st.warning("Lütfen tüm alanları doldurun.")
                else:
                    # 2. E-posta adresinin gerçekliğini kontrol et
                    with st.spinner(f"'{email_reg}' adresi kontrol ediliyor..."):
                        is_email_valid = verify_email_realness(email_reg)
                    
                    if not is_email_valid:
                        st.error("Lütfen geçerli ve ulaşılabilir bir e-posta adresi girin. (Örn: 'ornek@gmail.com')")
                    else:
                        # 3. Tüm kontroller başarılıysa, kullanıcıyı kaydetmeyi dene
                        st.info("E-posta geçerli. Kullanıcı kaydediliyor...")
                        if register_user(name_reg, email_reg, username_reg, password_reg):
                            st.success("Hesabınız başarıyla oluşturuldu! 'Giriş Yap' sekmesinden giriş yapabilirsiniz.")
                        # 'register_user' fonksiyonu zaten kendi hata/uyarı mesajını gösterecektir.
else:
    # --- GİRİŞ YAPILMIŞ ANA UYGULAMA ---
    name, username = st.session_state.get("name"), st.session_state.get("username")
    if 'last_known_user' not in st.session_state or st.session_state.last_known_user != username:
        st.session_state.clear(); st.session_state.update({"authentication_status": True, "username": username, "name": name, "last_known_user": username}); st.rerun()
    
    with st.sidebar:
        
        st.write(f'Hoş geldiniz, *{name}*');
        if st.button('Çıkış Yap'): st.session_state.clear(); st.rerun()
        st.markdown("---"); st.header("Yeni Rapor Yükle")
        uploaded_file = st.file_uploader("PDF dosyası seçin", type="pdf", label_visibility="collapsed")
        if uploaded_file and st.button("Raporu Analiz Et"): process_and_store_pdf(uploaded_file, username)
        st.markdown("---"); st.header("Geçmiş Sohbetler")
        user_conversations = load_user_conversations(username)
        if 'active_conversation_id' not in st.session_state: st.session_state.active_conversation_id = None
        
       
        for conv in user_conversations:
            conv_id, conv_name, conv_path, conv_images = conv['id'], conv['conversation_name'], conv.get('storage_path'), conv.get('image_paths', [])
            col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
            with col1:
                if st.button(conv_name, key=f"select_{conv_id}", use_container_width=True):
                    if st.session_state.active_conversation_id != conv_id:
                        st.session_state.active_conversation_id = conv_id; st.rerun()
            with col2:
                with st.popover("✏️"):
                    new_name = st.text_input("Yeni ad:", value=conv_name, key=f"rename_{conv_id}", label_visibility="collapsed")
                    if st.button("Kaydet", key=f"save_{conv_id}"):
                        if new_name and new_name != conv_name:
                            if rename_conversation(conv_id, new_name): st.rerun()
            with col3:
                if st.button("🗑️", key=f"delete_{conv_id}", help="Sohbeti kalıcı olarak sil"):
                    if delete_conversation(conv_id, conv_path, conv_images):
                        if st.session_state.active_conversation_id == conv_id:
                            st.session_state.active_conversation_id = None
                        st.rerun()
    
    active_conv = next((c for c in user_conversations if c['id'] == st.session_state.active_conversation_id), None)
    
    if not active_conv:
        st.info("Başlamak için kenar çubuğundan bir sohbet seçin veya yeni bir rapor yükleyin.")
    else:
        
        conv_id, storage_path, image_paths = active_conv['id'], active_conv.get('storage_path'), active_conv.get('image_paths', [])
        messages = load_messages(conv_id)
        for msg in messages:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        
        user_question = st.chat_input("Raporla ilgili sorunuzu yazın...")
        if user_question:
            save_message(conv_id, "user", user_question)
            with st.chat_message("user"): st.markdown(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Rapor analiz ediliyor ve yanıt oluşturuluyor..."):
                    try:
                        # 1. Metin Bağlamını Hazırla (Kaynak Göstererek)
                        context_text = "Bu soru için metin bağlamı bulunamadı."
                        if storage_path:
                            embeddings = get_embeddings_model()
                            vector_store = load_faiss_from_supabase(storage_path, supabase, embeddings)
                            if vector_store:
                                docs = vector_store.similarity_search(user_question, k=5)
                                # PERFORMANS ARTIŞI: Sayfa numarasını bağlama ekliyoruz
                                context_text = "\n\n".join([f"--- Kaynak (Sayfa: {doc.metadata.get('page', 'Bilinmiyor')}) ---\n{doc.page_content}" for doc in docs])
                        
                        # 2. Görsel Bağlamını Hazırla
                        prompt_images_b64 = []
                        if image_paths:
                            for img_path in image_paths:
                                try:
                                    img_bytes = supabase.storage.from_("faissfiles").download(img_path)
                                    prompt_images_b64.append(base64.b64encode(img_bytes).decode('utf-8'))
                                except Exception as e: print(f"Görsel indirilemedi: {img_path}, Hata: {e}")

                        # 3. LLM'i Doğrudan Çağır (Multi-modal için en temiz yol)
                        chat_history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in messages]
                        
                        
                        system_prompt = """Sen, sana verilen metin bağlamı ve görselleri analiz ederek soruları cevaplayan bir uzmansın.
- Cevabını oluştururken, kullandığın bilginin hangi metin kaynağından (Sayfa: X) geldiğini mutlaka belirt.
- Eğer bilgi bir görselde ise, 'rapordaki ilgili grafiğe/görsele göre...' gibi bir ifade kullan.
- Cevap bağlamda veya görsellerde yoksa, 'Bu bilgiye sağlanan rapor metinlerinde veya görsellerinde rastlayamadım.' de."""

                        # İnsan mesajının içeriğini dinamik olarak oluştur
                        human_message_content = [{"type": "text", "text": f"Metin Bağlamı:\n---\n{context_text}\n---\nSoru: {user_question}"}]
                        for img_b64 in prompt_images_b64:
                            human_message_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
                        
                        full_request_messages = [SystemMessage(content=system_prompt), *chat_history, HumanMessage(content=human_message_content)]
                        
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY, temperature=0.5, model_kwargs={"stream": True})
                        response = llm.invoke(full_request_messages)
                        full_response = response.content

                        st.markdown(full_response)
                        save_message(conv_id, "assistant", full_response)
                        send_log_to_n8n(username, active_conv['conversation_name'], user_question, full_response)
                        
                    except Exception as e: 
                        st.error(f"Yapay zeka yanıtı alınırken bir hata oluştu: {e}")
            
            st.rerun()
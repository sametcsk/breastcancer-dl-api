# Meme Kanseri Teşhis API 🎀

Bu proje, yapay zeka (PyTorch) destekli, sadece 5 temel özelliği (hücre çekirdeği özellikleri) kullanarak meme kanseri risk değerlendirmesi yapan bir web uygulamasıdır. Backend için **FastAPI**, frontend için modern ve duyarlı (responsive) Vanilla HTML/CSS kullanılmıştır.

## Özellikler ✨
- **Makine Öğrenmesi:** Wisconsin Meme Kanseri veri setinden Random Forest ile en önemli 5 özellik seçilerek eğitilmiş bir PyTorch sinir ağı.
- **Hızlı API:** Saniyeler içinde analiz yapabilen FastAPI sunucusu.
- **Modern Arayüz:** Kullanıcı dostu, animasyonlu ve koyu mod destekli (dark theme) şık tasarım.
- **Gerçek Zamanlı Sonuç:** Girilen 5 parametreye göre risk sınıfı ve yapay zeka güven (confidence) oranının anında gösterilmesi.

---

## Kurulum ve Çalıştırma 🚀

Projeyi sıfırdan bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### 1. Projeyi Klonlayın
```bash
git clone https://github.com/sametcsk/breastcancer-dl-api.git
cd breastcancer-dl-api
```

### 2. Sanal Ortam Oluşturun ve Bağımlılıkları Yükleyin
```bash
python -m venv .venv

# Windows için sanal ortamı aktif etme:
.\.venv\Scripts\activate

# Mac/Linux için sanal ortamı aktif etme:
source .venv/bin/activate

# Gerekli kütüphaneleri yükleyin:
pip install -r requirements.txt
```

### 3. Yapay Zeka Modelini Eğitin (Opsiyonel)
*(Not: `models` klasörü içinde eğitilmiş modeller zaten mevcuttur. Ancak modeli sıfırdan kendiniz eğitmek isterseniz bu adımı uygulayabilirsiniz.)*
```bash
python train_top5.py
```
Bu komut, veri setini indirip en önemli 5 özelliği seçecek, sinir ağını eğitecek ve güncel `.pth` ile `.pkl` dosyalarını `models` klasörüne kaydedecektir.

### 4. Web Uygulamasını Başlatın
Aşağıdaki komutla FastAPI sunucusunu çalıştırın:
```bash
uvicorn main:app --reload
```

### 5. Tarayıcıda Açın
Tarayıcınızı açın ve şu adrese gidin:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## Proje Yapısı 📁
- `main.py`: FastAPI backend kodları ve model entegrasyonu.
- `train_top5.py`: Modeli sıfırdan eğitmek için kullanılan script.
- `templates/index.html`: Kullanıcı arayüzü iskeleti.
- `static/style.css`: Arayüz tasarımı ve animasyonlar.
- `models/`: Eğitilmiş PyTorch ağırlıkları (`.pth`) ve scaler nesneleri (`.pkl`).

*Bu proje yalnızca eğitim ve bilgilendirme amaçlıdır. Tıbbi bir teşhis aracı yerine geçmez.*

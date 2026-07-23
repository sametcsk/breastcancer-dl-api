# Meme Kanseri Risk Sınıflandırma Demosu

Bu eğitim projesi, Wisconsin Breast Cancer veri setindeki beş özellik üzerinden
ikili sınıflandırma yapan bir PyTorch modeli ve FastAPI tabanlı web arayüzü içerir.

> Bu uygulama tıbbi teşhis aracı değildir. Model çıktısı klinik olasılık,
> uzman değerlendirmesi veya tıbbi güven ölçüsü olarak yorumlanamaz.

## Özellikler

- Random Forest ile özellik öneminin incelenmesi
- Seçilen beş özellik üzerinde sinir ağı eğitimi
- FastAPI ile tahmin uç noktası
- HTML ve CSS tabanlı örnek kullanıcı arayüzü

Yalnızca beş özelliğin kullanılması bilgi kaybına neden olabilir. Veri setinin
boyutu, temsil gücü ve deneysel veri bölme yöntemi de sonuçları sınırlar.

## Kurulum

```bash
git clone https://github.com/sametcsk/breastcancer-dl-api.git
cd breastcancer-dl-api
python -m venv .venv
pip install -r requirements.txt
python train_top5.py
uvicorn main:app --reload
```

Uygulama çalıştığında `http://127.0.0.1:8000` adresinden açılabilir.

## Dosya Yapısı

- `main.py`: FastAPI uygulaması ve model entegrasyonu
- `train_top5.py`: veri hazırlama, özellik seçimi ve model eğitimi
- `models/`: model ağırlıkları ve ön işleme nesneleri
- `templates/`: HTML şablonu
- `static/`: arayüz stilleri

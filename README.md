# Hazmat & C Sign Detection Project

Bu proje, video akışı üzerinde "C" işaretlerini ve yönlerini tespit ederken aynı zamanda Hazmat (Tehlikeli Madde) işaretlerini de tespit eden bir bilgisayarlı görü uygulamasıdır.

Proje, şablon eşleştirme (Template Matching) ve HOG (Histogram of Oriented Gradients) tekniklerini kullanarak tespit işlemini gerçekleştirir.

## Özellikler

- **C İşareti Tespiti:** Sağ, Sol, Yukarı, Aşağı yönlü C işaretlerini tespit eder.
- **Hazmat Tespiti:** 15 farklı tehlikeli madde işaretini tespit eder (Explosives, Flammable Gas, Radioactive, vb.).
- **Renk Doğrulama:** Hazmat işaretlerinin doğruluğunu artırmak için renk analizi yapar.
- **Takip (Tracking):** Tespit edilen objeleri takip eder ve aynı objeyi tekrar tekrar loglamaz.
- **Durdurma (Pause):** Yeni bir obje tespit edildiğinde videoyu otomatik olarak duraklatır.

## Gereksinimler

Projenin çalışması için aşağıdaki Python kütüphanelerine ihtiyaç vardır:

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Scikit-Image (`scikit-image`)

## Kurulum

1. **Projeyi Klonlayın veya İndirin:**
   Projeyi bilgisayarınıza indirin ve proje dizinine gidin.

2. **Sanal Ortam Oluşturun (Önerilen):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Bağımlılıkları Yükleyin:**
   Gerekli kütüphaneleri `requirements.txt` dosyasından yükleyin.
   ```bash
   pip install -r requirements.txt
   ```

## Kullanım

Projeyi çalıştırmak için `main.py` dosyasını kullanın. Video dosyasının yolunu argüman olarak vermeniz gerekmektedir.

### Temel Kullanım

```bash
python main.py data/odev1.mp4
```

### Gelişmiş Kullanım

Eğer şablon dosyalarınız farklı klasörlerdeyse veya farklı parametreler kullanmak istiyorsanız:

```bash
python main.py data/odev1.mp4 \
    --c-templates-root templates/c_sign \
    --hazmat-templates-root templates/hazmat \
    --threshold 0.6 \
    --pause-distance 80
```

### Kontroller

- **SPACE:** Video duraklatıldığında (yeni bir tespit yapıldığında) devam etmek için basın.
- **Q veya ESC:** Videodan çıkmak ve programı sonlandırmak için basın.

## Dosya Yapısı

- `main.py`: Ana program dosyası. Video akışını yönetir ve tespitleri ekrana çizer.
- `hazmat.py`: Hazmat tespiti yapan sınıf (HOG tabanlı).
- `c_sign_detector.py`: C işareti tespiti yapan sınıf (Template Matching tabanlı).
- `templates/`: Tespit için kullanılan referans görseller.
  - `c_sign/`: C işareti şablonları.
  - `hazmat/`: Hazmat işareti şablonları.
- `data/`: Test videolarının bulunduğu klasör.
- `requirements.txt`: Gerekli kütüphane listesi.

## Notlar

- Işık koşulları ve video kalitesi tespit başarısını etkileyebilir.
- Hazmat tespiti için renk doğrulama kullanıldığından, videodaki renklerin şablonlardaki renklerle uyumlu olması önemlidir.
- `hazmat.py` içerisindeki `CUSTOM_THRESHOLDS` değerleri, tespit hassasiyetini ayarlamak için değiştirilebilir.


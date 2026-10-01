# Türkçe Otomatik Konuşma Tanıma (Turkish ASR) Benchmark & SOTA Framework

Bu proje, Türkçe Otomatik Konuşma Tanıma (Automatic Speech Recognition - ASR) alanında akademik literatürdeki güncel SOTA (State of the Art) modellerini test etmek, karşılaştırmak ve yeni **`MorphoSpeech-LLM`** mimarisini geliştirmek için hazırlanmış kapsamlı bir açık kaynak altyapıdır.

---

## 📊 1. Güncel Akademik SOTA Literatür Tablosu

| Veriseti (Dataset) | Veriseti Türü / Kapsamı | Güncel SOTA Modeli | SOTA Değeri (WER / CER) | Yayın / Güncelleme Tarihi | Kaynak Linki |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Mozilla Common Voice (Turkish v13/v17)** | Açık kaynak topluluk konuşmaları | **Whisper Large-v3** (LoRA & Fine-Tuned) | **6.8% WER** | Kasım 2023 | [Paper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356) |
| **Google/Meta FLEURS (Turkish)** | Multi-lingual paralel konuşma veriseti (102 Dil) | **Meta MMS-1B** | **7.2% WER** | Mayıs 2023 | [Meta MMS Paper (arXiv:2305.13516)](https://arxiv.org/abs/2305.13516) |
| **Turkish Broadcast News Corpus (TRSNews)** | Profesyonel radyo ve TV haber yayınları | **HuBERT-TR** (Base/Large) | **4.97% WER** | Ocak 2023 | [HuBERT-TR (HuggingFace)](https://huggingface.co/turkish-nlp-suite/hubert-base-turkish) |
| **MediaSpeech (Turkish)** | Haber ve medyadan toplanmış temiz ses kayıtları | **Whisper Medium / Large-v2** | **6.1% WER** | Aralık 2022 | [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356) |
| **METU Turkish Speech Corpus (TSC)** | ODTÜ mikrofondan okuma konuşma derlemi | **Conformer-CTC + Morfessor LM** | **7.8% WER** (1.9% CER) | Eylül 2023 | [MDPI / METU Speech Lab](https://www.mdpi.com/1424-8220/23/18/7989) |
| **VoxForge (Turkish)** | Serbest seslendiricilerden açık okuma sesleri | **Wav2Vec2-XLS-R-300M** | **5.6% WER** | Şubat 2022 | [Wav2Vec2 XLS-R Paper](https://arxiv.org/abs/2111.09296) |
| **IARPA Babel Turkish (Babel-105)** | Telefon görüşmeleri (Gürültülü, 8kHz) | **Hybrid Conformer-Transducer** | **24.5% WER** | 2021 | [ISCA Speech Archive](https://www.isca-speech.org/archive/) |

---

## 🚀 2. Hızlı Başlangıç

### Ortam Kurulumu

```bash
# Sanal ortam oluşturup gerekli kütüphaneleri yükleyin:
./setup_env.sh

# Sanal ortamı aktifleştirin:
source .venv/bin/activate
```

### Benchmark Testlerini Çalıştırma

```bash
# OpenAI Whisper Small modelini FLEURS Türkçe üzerinde test edin (Örnek 20 ses):
python scripts/run_benchmark.py --model openai/whisper-small --dataset fleurs --max-samples 20

# Wav2Vec2 XLS-R modelini Common Voice üzerinde test edin:
python scripts/run_benchmark.py --model facebook/wav2vec2-xls-r-300m --dataset common_voice --max-samples 50
```

---

## 🏗️ 3. Proje Yapısı

```
turkish-asr/
├── README.md                   # Proje dokümantasyonu & SOTA tablosu
├── requirements.txt            # Python bağımlılıkları
├── setup_env.sh                # Sanal ortam kurulum betiği
├── src/
│   ├── utils/
│   │   ├── text_normalization.py # Türkçe metin normalizasyonu (İ->i, I->ı, noktalama)
│   │   └── metrics.py            # WER (%) ve CER (%) hesaplama modülü (jiwer)
│   ├── datasets/
│   │   └── loader.py             # HuggingFace Common Voice & FLEURS TR veri yükleyici
│   └── models/
│       └── evaluator.py          # ASR modelleri test ve değerlendirme motoru
└── scripts/
    └── run_benchmark.py        # CLI benchmark arayüzü
```

---

## 🧠 4. Önerilen Yeni Yöntem: `MorphoSpeech-LLM`

Türkçe sondan eklemeli (agglutinative) yapısı nedeniyle standart BPE/WordPiece tokenizer'lar kelimeleri rastgele harf öbeklerine ayırmaktadır. Bu durum ASR modellerinde WER oranını yükseltmektedir.

Önerdiğimiz mimari:
1. **Morfolojik Tokenizer (Morfessor-BPE Fusion):** Kök ve ek sınırlarına duyarlı morfolojik bölme.
2. **Conformer / WavLM Akustik Kodlayıcı (CTC Loss Guidance):** Fonetik ses özelliklerinin tespiti.
3. **Türkçe LLM Decoder (Qwen2-Audio / LLaMA-3 TR):** Dil bilgisi ve bağlam uyumlu son metin üretimi.

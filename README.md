# 🚚 TEKNOFEST 2026 - Lojistik Optimizasyon Sistemi

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![OR-Tools](https://img.shields.io/badge/OR--Tools-SCIP%2FCBC-blue)](https://developers.google.com/optimization)
[![Tests](https://github.com/Umutcagn/logistics-fleet-network-optimization/actions/workflows/tests.yml/badge.svg)](https://github.com/Umutcagn/logistics-fleet-network-optimization/actions/workflows/tests.yml)

> **Entegre Makine Öğrenmesi (LightGBM + PyTorch LSTM), FSMVRP Rota Optimizasyonu (Google OR-Tools ILP / ALNS), PyTorch GNN Darboğaz Erken Uyarısı ve Sub-Agent Yönetişim Paneli.**

---

## 📌 İçindekiler
- [Proje Hakkında](#-proje-hakkında)
- [Sistem Mimarisi](#-sistem-mimarisi)
- [Matematiksel Formülasyon](#-matematiksel-formülasyon)
- [Özellikler & Teknolojiler](#-özellikler--teknolojiler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Testler](#-testler)
- [Akademik Kaynakça](#-akademik-kaynakça)

---

## 📖 Proje Hakkında

Bu proje, Türkiye genelindeki Transfer Merkezleri (TM) ve lojistik ağı için geliştirilmiş **uçtan uca entegre karar destek ve rota optimizasyon sistemidir**. Geleneksel bağımsız modüller yerine, **tahmin çıktısını optimizasyona ve optimizasyon maliyetini tahmin katmanına besleyen** yaşayan bir pipeline mimarisine sahiptir.

---

## 🏗️ Sistem Mimarisi

```mermaid
graph TD
    A[Geçmiş Veri & Stok Değişkeni W_it] --> B[Makine Öğrenmesi Tahmin Katmanı]
    B -->|LightGBM & LSTM Ensemble| C[Tahmin & Belirsizlik Aralığı D_it]
    C --> D[Tahmin-Optimizasyon Entegrasyonu & WAPE Feedback]
    D --> F[Google OR-Tools ILP Solver - K1, K3, K6]
    F -->|K3 çözümsüzse| G[Uyarılı Sezgisel Yedek Plan]
    F --> H[Spot / Filo Araç Ataması]
    G --> H
    H --> I[PyTorch GNN Darboğaz Erken Uyarı Katmanı]
    I --> J[Sub-Agent Koordinasyon Sistemi]
    J --> K[Streamlit Karar Destek Dashboard & İnsan Onayı]
```

---

## 📐 Matematiksel Formülasyon

### 1. Amaç Fonksiyonu (Denklem 1.1)
$$\min Z = \sum_{(k,m)} F_m \cdot Y_{mk} + \sum_{(i,j,k)} D_{ij} \cdot C_k \cdot X_{ijk} + \sum_{(i,j,t)} S_{ij} \cdot Z_{spot\_ijt} + \sum_{(i,t)} \beta \cdot W_{it}$$

- $F_m$: $m$ tipi filo aracının günlük sabit maliyeti ($\text{Tır}=4500\text{TL}, \text{Kamyon}=2800\text{TL}, \text{Hafif Kamyon}=1800\text{TL}, \text{Kamyonet}=1100\text{TL}$)
- $D_{ij}$: Güzergah mesafesi (km)
- $C_k$: Km başına yakıt ve bakım maliyeti
- $S_{ij}$: Spot araç toplam kiralama maliyeti
- $\beta$: Birim desi başına Transfer Merkezi bekleme ve elleçleme maliyeti

### 2. Kısıtlar

> **Uygulama durumu:** ILP modeli şu an **K1, K3 ve K6**'yı kısıt olarak uygular. K3, her çıkış merkezinden günlük sevk edilen araç sayısını `rampa sayısı × 6 yükleme dalgası` ile sınırlar; tır rampası olmayan merkezlerde tır değişkenleri 0'a sabitlenir. **K4** kısıt olarak modelde yoktur, çözümden sonra merkez bazında kullanım oranı olarak raporlanır. **K2 ve K5** henüz uygulanmamıştır.

- **K1 (Talep Karşılama)**: $\sum_m (\text{cap}_m \cdot X_{ijk} \cdot Y_{mk}) + \text{cap}_{spot} \cdot Z_{spot\_it} \ge \hat{D}_{it} \quad \forall(i,j,t)$
- **K2 (Araç Kapasitesi)**: $l_{ij} \le \sum_m \text{cap}_m \cdot X_{ijk} \cdot Y_{mk} \quad \forall(i,j,k)$
- **K3 (Rampa Kısıtı)**: Tır rampası olmayan merkeze otomatik küçük araç ataması ($\sum_k W_{itk} \le R_i$).
- **K4 (Elleçleme Kapasitesi)**: $\text{Giden}(i,t) \le H_i \times \Delta t$.
- **K5 (Doluluk Toleransı)**: $l_{ij} \ge (1-\alpha) \cdot \text{cap}_m \cdot X_{ijk} \cdot Y_{mk}$ ($\alpha_{\text{tır}}=0.10, \alpha_{\text{kamyon}}=0.15, \alpha_{\text{kamyonet}}=0.20$).
- **K6 (Tamsayı Kısıtları)**: $X_{ijk}, Y_{mk} \in \{0,1\}, Z_{spot\_ijt} \in \mathbb{Z}^+$.

---

## 🛠️ Özellikler & Teknolojiler

- **Makine Öğrenmesi (ML)**: LightGBM (lag/rolling ve takvim özellikleri) + PyTorch LSTM (14 günlük kayan pencere, güzergah bazlı ölçekleme, bir gün sonrası tahmini) + Stacking MetaLearner (WAPE bazlı ağırlıklandırma). Bkz. [Tahmin Katmanı](#-tahmin-katmanı-değerlendirmesi).
- **Rota Optimizasyonu (FSMVRP)**: Google OR-Tools ILP (SCIP/CBC) ana çözücüdür. ALNS sezgisel çözücüsü karşılaştırma ve çok büyük ağlar için tutulur. Bkz. [Çözücü Karşılaştırması](#-çözücü-karşılaştırması-ilp-vs-alns).
- **Graf Sinir Ağı (GNN)**: Düz PyTorch ile yazılmış, ortalama-komşu mesaj iletimli Transfer Merkezleri darboğaz erken uyarı modeli (MARLHOSA prensibi). Sentetik ağ senaryolarıyla eğitilmiştir (bkz. [GNN Eğitimi](#-gnn-eğitimi-ve-değerlendirme)).
- **Sub-Agent & Human-in-the-Loop**: 3 uzman alt ajan (Talep, Optimizasyon, GNN) ve kritik kararlarda lojistik sorumlusu onay paneli.
- **Arayüz**: Streamlit etkileşimli kontrol paneli.

### ⚠️ Bilinen Sınırlamalar
- Veriler sentetiktir (`src/data/synthetic_data_generator.py`); sonuçlar gerçek operasyon verisiyle doğrulanmamıştır.
- `ortools` yüklü değilse ILP yerine sezgisel (heuristic) çözücüye, `lightgbm` yüklü değilse scikit-learn `HistGradientBoostingRegressor`'a sessizce geçilir.
- ALNS rampa kısıtını (K3) uygulamaz ve her güzergaha tek tip araç atar.
- Tahmin bandı (P10–P90), gerçek hata dağılımından değil, ±%12 standart sapma varsayımından üretilir.
- GNN, gerçek darboğaz kayıtlarıyla değil, açık bir kapasite/taşma kuralıyla etiketlenmiş sentetik senaryolarla eğitilmiştir; yani bu kuralı öğrenir, gerçek operasyondan öğrenilmiş bir model değildir.

---

## ⚡ Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/Umutcagn/logistics-fleet-network-optimization.git
cd logistics-fleet-network-optimization

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

---

## 🚀 Kullanım

### 1. Uçtan Uca Pipeline Çalıştırma (CLI)
```bash
python run_pipeline.py
```

### 2. Streamlit Web Dashboard'unu Başlatma
```bash
streamlit run app.py
```
Tarayıcınızda `http://localhost:8501` adresine gidin.

---

## 📈 Tahmin Katmanı Değerlendirmesi

Son 5 gün doğrulama verisidir. MetaLearner ağırlıkları ilk 3 günde seçilir, WAPE ise ağırlık seçimine katılmayan **son 2 günde** (180 güzergah-gün) ölçülür. Tüm modeller "bir gün sonrası" tahmini yapar: her gün için yalnızca önceki günlerin gerçekleşen değerleri kullanılır.

| Model (30 günlük veri, `python run_pipeline.py`) | Holdout WAPE |
|---|---|
| **Ensemble (w1=0.55 LightGBM, w2=0.45 LSTM)** | **%10.6** |
| LightGBM | %13.9 |
| Naif tahmin (dünün değeri) | %14.2 |
| LSTM | %15.0 |

Ensemble, tek başına modellerin ve naif tahminin önündedir. Bu tek bir holdout penceresi olduğundan sonuçlar gürültülüdür; daha uzun geçmişte (60–90 gün) LightGBM belirgin şekilde güçlenir ve MetaLearner LSTM ağırlığını düşürür.

---

## ⚖️ Çözücü Karşılaştırması (ILP vs ALNS)

İki çözücü aynı maliyet fonksiyonuyla, sentetik veriden alınan 5 farklı gün, 4 güzergah sayısı ve 3 talep seviyesinde karşılaştırıldı (toplam 60 örnek):

```bash
python -m benchmarks.solver_benchmark   # sonuçlar: benchmarks/solver_benchmark_results.json
```

| Güzergah | Talep | ILP optimal | ILP çözümsüz (K3) | ALNS ort. maliyet farkı | ALNS maks. fark | ILP süre (sn) | ALNS süre (sn) | ALNS K3 ihlali |
|---|---|---|---|---|---|---|---|---|
| 10 | 1x | 5/5 | 0/5 | +%1.81 | +%12.69 | 0.010 | 0.017 | 0/5 |
| 10 | 2x | 5/5 | 0/5 | +%17.40 | +%22.83 | 0.008 | 0.016 | 0/5 |
| 10 | 3x | 4/5 | 1/5 | +%7.16 | +%12.94 | 0.010 | 0.016 | 1/5 |
| 30 | 1x | 5/5 | 0/5 | +%6.22 | +%12.84 | 0.011 | 0.047 | 0/5 |
| 30 | 2x | 4/5 | 1/5 | +%14.85 | +%16.62 | 0.011 | 0.048 | 1/5 |
| 30 | 3x | 0/5 | 5/5 | — | — | 0.008 | 0.048 | 5/5 |
| 50 | 1x | 5/5 | 0/5 | +%4.64 | +%9.14 | 0.014 | 0.082 | 0/5 |
| 50 | 2x | 4/5 | 1/5 | +%12.82 | +%13.75 | 0.016 | 0.083 | 1/5 |
| 50 | 3x | 0/5 | 5/5 | — | — | 0.009 | 0.081 | 5/5 |
| 90 | 1x | 5/5 | 0/5 | +%6.51 | +%11.60 | 0.022 | 0.148 | 0/5 |
| 90 | 2x | 0/5 | 5/5 | — | — | 0.019 | 0.143 | 5/5 |
| 90 | 3x | 0/5 | 5/5 | — | — | 0.008 | 0.143 | 5/5 |

Örnek bazlı sonuçlar `benchmarks/solver_benchmark_results.json` dosyasındadır. ILP'nin çözümsüz olduğu satırlarda optimum maliyet tanımsız olduğundan fark hesaplanmaz. Çıkarımlar:
- ILP, 90 güzergahlı ağın tamamını **~0.02 saniyede kanıtlanmış optimumla** çözer; ALNS hem daha yavaş hem ortalama %2–17 daha pahalıdır. Bu yüzden eski "≥ 50 güzergahta ALNS" kuralı kaldırıldı ve ILP tüm ağ için kullanılır (`ILP_ROUTE_LIMIT`). Hedef gün için plan maliyeti bu değişiklikle 1.281.464 TL'den 1.188.264 TL'ye düştü (−%7,3).
- Talep 2 katına çıktığında tır rampası olmayan Antalya ve Trabzon'un, 3 katına çıktığında ise ek olarak az rampalı Adana ve Diyarbakır'ın rampa kapasitesi (K3) talebi taşıyamaz ve problem çözümsüz hale gelir. ILP bu durumda boş plan döndürmek yerine açık bir uyarıyla sezgisel yedek plana geçer. ALNS ise K3'ü bilmediği için bu durumda **uyarı vermeden ihlal eden planlar** üretir.

---

## 🧠 GNN Eğitimi ve Değerlendirme

Gerçek etiketli darboğaz verisi olmadığından etiketler açık bir kuraldan üretilir. Bir transfer merkezi, kendi 12 saatlik yük oranı ile aşırı yüklü komşularından taşan yükün toplamı kapasitesine ulaştığında darboğaz sayılır:

$$\text{load}_i + 0.5 \cdot \frac{1}{|N(i)|}\sum_{j \in N(i)} \max(0, \text{load}_j - 1) \ge 1$$

Model; gerçek 10 transfer merkezi üzerinde rastgele yük seviyeleri ve rastgele güzergah grafları içeren 6.000 eğitim, 1.500 doğrulama ve 2.000 test senaryosuyla eğitilir (~7 sn, CPU):

```bash
python -m src.gnn.train_gnn   # ağırlıklar: src/gnn/bottleneck_gnn_weights.pt, metrikler: src/gnn/training_metrics.json
```

Test seti (20.000 düğüm) sonuçları, komşuları görmeyen eşik kuralıyla (`load ≥ 1`) karşılaştırmalı:

| Model | Doğruluk | Kesinlik | Duyarlılık | F1 | Yalnızca komşu taşmasından doğan darboğazları yakalama (977 düğüm) |
|---|---|---|---|---|---|
| Düğüm eşiği (komşusuz) | 0.951 | 1.000 | 0.859 | 0.924 | %0 |
| **GNN** | **0.990** | **0.990** | **0.982** | **0.986** | **%90.3** |

GNN'in kazancı tamamen graf yapısından gelir: yalnızca kendi yüküne bakan bir kural, komşu merkezlerden taşan yükü göremez.

> Not: Pipeline'ın ürettiği mevcut sentetik talepte merkezlerin 12 saatlik elleçleme yük oranı %6–24 arasındadır; talep 3 katına çıksa bile elleçleme kapasitesi aşılmaz ve GNN alarm üretmez. Dashboard stres modunda darboğaz elleçlemede değil, rampa kapasitesinde (K3) ortaya çıkar; bu, optimizasyon katmanı tarafından uyarı olarak gösterilir.

---

## 🧪 Testler

Projedeki tüm alt modüller `pytest` ile test edilmiştir:
```bash
python -m pytest tests/
```

---

## 📚 Akademik Kaynakça

1. **Weng, S. et al. (2020)**. *Supply chain sales forecasting based on LightGBM and LSTM combination model*. Industrial Management & Data Systems.
2. **Ling, J. & Wu, W.B. (2025)**. *Comparative Analysis of Global and Local Probabilistic Time Series Forecasting*. arXiv:2509.08214.
3. **Fung, W. et al. (2025)**. *OpComm: A Reinforcement Learning Framework for Adaptive Buffer Control*. arXiv:2512.19738.
4. **Mandi, J. et al. (2024)**. *Decision-Focused Learning: Foundations, State of the Art*. JAIR.
5. **Fachini, R.F. et al. (2025)**. *A stochastic programming approach for the operational fleet composition problem*. ITOR.
6. **Salhi, S. et al. (2013)**. *The Fleet Size and Mix Vehicle Routing Problem with Backhauls*. Transportation Research Part E.
7. **Taherkhani, G. et al. (2025)**. *Exact Solution Method for Multi-Stakeholder Freight Transportation Systems Under Uncertainty*. Transportation Research Part B.
8. **Zhang, M. & Pan, C. (2024)**. *Hierarchical Optimization Scheduling Algorithm Based on Multi-Agent Reinforcement Learning*. IEEE T-ITS.
9. **Zhao, Y. & Jiang, Z. (2026)**. *Hub Location and Linehaul Route Planning with Truck Platooning*. Symmetry.

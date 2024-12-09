# Klasyfikacja Obrazów Psów i Kotów przy użyciu Sztucznych Sieci Neuronowych  

## Autorzy  
- Bartłomiej Kuk (nr albumu: 272497)  
- Daria Siwiak (nr albumu: 272510)  

## **Opis projektu**  
Projekt dotyczy klasyfikacji obrazów psów i kotów przy użyciu sztucznej sieci neuronowej (SSN). Celem było zbudowanie i przetestowanie różnych architektur sieci w celu osiągnięcia możliwie najlepszej dokładności klasyfikacji.
---
## **Technologie i Biblioteki**  
- **Python** – główny język programowania  
- **TensorFlow/Keras** – budowa i trenowanie sieci neuronowych  
- **Matplotlib, Seaborn** – wizualizacja wyników (wykresy, macierze pomyłek)  
- **Sklearn** – obliczanie metryk klasyfikacji  
- **PIL** – przetwarzanie obrazów  
---
## **Przygotowanie Danych**  
1. **Zbiory danych:**  
   - **Testowy (10%)** – ocena skuteczności modelu  
   - **Treningowy (89%)** – trening modelu  
   - **Walidacyjny (1%)** – monitorowanie postępów uczenia  
2. **Przetwarzanie danych:**  
   - Zdjęcia psów i kotów w formacie JPG, skalowane do rozmiaru 150x150px  
   - Normalizacja danych do zakresu [0,1]  
---
## **Architektury Sieci Neuronowych**  
1. **Fully Connected Network (FCN):**  
   - Każdy neuron łączy się z każdym neuronem w następnej warstwie.  
   - Użyte warstwy: 2 lub 3, liczba neuronów: 64, 128, 256, 512  
2. **Dimensional Reduction Network (DRN):**  
   - Liczba neuronów w kolejnych warstwach zmniejszana dwukrotnie.  
   - Konfiguracja podobna do FCN.  
---
## **Trenowanie Sieci**  
- **Metoda:** EarlyStopping – przerwanie uczenia po 3 epokach bez poprawy dokładności.  
- **Parametry:**  
  - Liczba epok: 50 (lub wcześniejsze zakończenie)  
  - Learning rate: 0.001  

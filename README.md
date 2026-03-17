# Waste Classification — Convolutional Neural Network (CNN)

## 🚀 Cel Projektu
Celem ćwiczenia było zaprojektowanie i wytrenowanie Konwolucyjnej Sieci Neuronowej (CNN) do binarnej klasyfikacji obrazów ze zbioru **Waste Classification Dataset**. Model dzieli odpady na dwie kategorie:
* **Organic** (organiczne)
* **Recyclable** (do recyklingu)

## 🛠️ Architektura Modelu (CNN)
Zastosowano autorską strukturę sieci konwolucyjnej, zapewniającą efektywną ekstrakcję cech przy optymalnej złożoności:
1.  **Wejście:** Obrazy $64\times64$ piksele w skali szarości (1 kanał).
2.  **Warstwa Konwolucyjna:** 32 filtry o rozmiarze $3\times3$ z aktywacją **ReLU**.
3.  **Warstwa Pooling:** MaxPooling $2\times2$.
4.  **Flatten:** Przekształcenie macierzy cech na wektor.
5.  **Warstwa Ukryta (Dense):** 64 neurony z aktywacją **ReLU**.
6.  **Warstwa Wyjściowa:** 1 neuron z aktywacją **Sigmoid**.

## ⚙️ Parametry Uczenia i Optymalizacja
Proces uczenia został skonfigurowany pod kątem stabilności i precyzji:
* **Funkcja błędu:** MSE (Mean Squared Error).
* **Optymalizator:** SGD z parametrem **Momentum = 0.9**.
* **Mini-batch:** 32 próbki.
* **Adaptacyjny Learning Rate:** Automatyczna redukcja współczynnika uczenia w przypadku stagnacji błędu (ReduceLROnPlateau).
* **Early Stopping:** Przerwanie treningu po 5 epokach bez poprawy wyniku na zbiorze walidacyjnym.

## 🧹 Przygotowanie Danych (Preprocessing)
* Zmiana rozdzielczości obrazów do formatu $64\times64$.
* Konwersja do skali szarości (redukcja szumu informacyjnego i kosztu obliczeń).
* Normalizacja wartości pikseli do przedziału [0, 1].
* Podział danych na zbiór uczący (TRAIN) oraz walidacyjny (TEST).

## 📈 Wyniki i Wnioski
Model osiągnął dokładność na poziomie ok. **78%**. 

Kluczowe obserwacje:
* **Zbieżność:** Zastosowanie momentum skutecznie wygładziło krzywe błędu MSE, przyspieszając zbieżność algorytmu.
* **Analiza Wag:** Histogramy wag po treningu wykazują przejście do rozkładu normalnego, co dowodzi, że sieć nauczyła się rozpoznawać specyficzne tekstury i kształty odpadów.
* **Geometria vs Kolor:** Skuteczna klasyfikacja w skali szarości potwierdza, że w przypadku odpadów kluczowe są cechy geometryczne i strukturalne, a kolor jest informacją drugorzędną.

# Leaf Health Backend – Model train, k-mea

Este projeto utiliza **redes neurais convolucionais (CNNs)** e **análise não supervisionada por cores (K-Means)** para identificar **doenças em folhas de plantas** e estimar o **percentual de dano** nas áreas infectadas.  
O sistema é dividido em dois módulos principais:

1. **Treinamento de modelo de classificação** (`train_a100.py`)
2. **Inferência e segmentação da folha** (`app.py` + `segmentation_kmeans.py`)

---

## 1. Treinamento do Modelo de Classificação

O modelo de classificação aprende a reconhecer **tipos de doenças em folhas** com base em um dataset de imagens (como o *PlantVillage*).

### Estrutura de dados

O dataset deve seguir o padrão do `ImageFolder` do PyTorch:

dataset/
train/
Tomato___Early_blight/
Tomato___Late_blight/
Corn___Healthy/
valid/
Tomato___Early_blight/

Cada subpasta representa uma **classe**, no formato `Planta___Doença`.

---

### Arquitetura e Backbone

O script suporta múltiplas arquiteturas de *backbone* (base da rede):

- `efficientnet_b3` (usado por padrão)
- `efficientnet_b4`
- `resnet50`
- `convnext_tiny`

O cabeçalho da rede é ajustado automaticamente para o número de classes do dataset.

---

### Conceitos Principais

#### **1. Transfer Learning**

O modelo começa com pesos pré-treinados em *ImageNet* e é ajustado (fine-tuned) para o problema de doenças em folhas.  
Isso acelera o treino e melhora a precisão, já que as primeiras camadas da rede já sabem detectar padrões visuais gerais (bordas, texturas, cores).

#### **2. OneCycleLR Scheduler**

Usa a política de aprendizado *One Cycle Learning Rate* que:

- aumenta a taxa de aprendizado no início;
- e depois diminui suavemente até o final do treino.

Isso acelera a convergência e reduz overfitting.

#### **3. Mixed Precision Training**

A técnica de **treinamento em precisão mista** (`float16` ou `bfloat16`) permite usar **menos memória** e **aumentar a velocidade**, especialmente em GPUs como a NVIDIA A100.  
O PyTorch faz isso automaticamente com `torch.cuda.amp.autocast`.

#### **4. Freeze / Unfreeze**

Durante os primeiros epochs é possível **congelar o backbone** (não treinar as camadas convolucionais base) e treinar apenas o *head* (camadas de saída).  
Depois disso, o modelo é “descongelado” e ajusta todas as camadas.

#### **5. GradScaler**

Durante o *mixed precision training*, pode ocorrer *underflow* dos gradientes.  
O `GradScaler` ajusta dinamicamente a escala dos gradientes para evitar perdas de precisão numérica.

---

### Hiperparâmetros usados

| Parâmetro              | Valor     | Descrição |
|------------------------|-----------|------------|
| `BATCH_SIZE`           | 128       | Tamanho do batch |
| `EPOCHS`               | 40        | Épocas de treino |
| `BASE_LR`              | 1e-3      | Taxa base de aprendizado |
| `REF_BATCH`            | 32        | Referência para *linear scaling* |
| `ACCUMULATION_STEPS`   | 1         | Acúmulo de gradientes |
| `WEIGHT_DECAY`         | 1e-4      | Regularização L2 |
| `CLIP_GRAD_NORM`       | 1.0       | Evita explosão de gradientes |
| `USE_AMP`              | True      | Usa mixed precision |
| `BACKBONE`             | EfficientNet-B3 | Arquitetura principal |

---

## 2. Inferência e Segmentação (Detecção de Lesões)

Após o treinamento, o modelo é usado no servidor FastAPI (`app.py`) para receber uma **imagem de folha**, identificar a **doença**, e gerar **máscaras de lesões**.

---

### Pipeline de inferência

1. A imagem é recebida via POST no endpoint `/predict`.
2. O modelo de classificação prevê:
   - a planta (`plant`)
   - a doença (`disease`)
   - e as probabilidades (`scores`)
3. O módulo `segmentation_kmeans.py` segmenta a folha e detecta **regiões de doença** com base nas cores.

---

## 3. Segmentação K-Means

A detecção de áreas lesionadas é feita por **clustering não supervisionado** em cores (sem necessidade de labels).  
O método assume que **as folhas saudáveis possuem tons de verde similares**, e que **as lesões têm cores distintas** (marrom, amarelo, preto etc).

### Etapas do algoritmo

1. **Segmentação da folha**
   - Usa HSV para detectar a cor verde predominante.
   - Cria uma *máscara binária* com os pixels da folha.

2. **Conversão para LAB**
   - O espaço de cor LAB é usado porque separa **luminosidade (L)** da **cor (a,b)**.
   - Isso facilita identificar variações sutis de cor causadas por doenças.

3. **Clusterização com K-Means**
   - Os pixels da folha são agrupados em *k* clusters (tipicamente `k=3` ou `k=4`).
   - O maior cluster é considerado **folha saudável**.
   - Os demais representam **regiões lesionadas**.

4. **Pós-processamento**
   - Operações morfológicas (`open` e `close`) limpam ruídos e suavizam bordas.
   - Mantém apenas a maior área conectada, se desejado.

5. **Cálculo de dano**
   - O percentual de dano é obtido pela razão entre a área lesionada e a área total da folha:
     \[
     \text{Dano (\%)} = \frac{\text{Pixels da Lesão}}{\text{Pixels da Folha}} \times 100
     \]

---

### 3.1. Módulo `build_centers.py` — Centros Globais de Cor (Pré-Treinamento do K-Means)

O módulo `build_centers.py` serve para gerar um **modelo de cores representativas das folhas e lesões**.  
Em vez de rodar o *K-Means* do zero em cada imagem (o que é mais lento e inconsistente),  
foram criados **centros de cor globais** (`centers_lab_global.npy`) com base em **várias imagens** do dataset.

Esses centros são depois reutilizados no `segmentation_kmeans.py` para classificar cores rapidamente, mantendo consistência entre execuções.

---

#### **Objetivo**

Gerar um arquivo contendo os **centros médios de cor** (em espaço LAB) representando:

- Tons típicos de **verde saudável**
- Tons amarelados, marrons ou escuros de **lesões**

---

#### **Como funciona**

1. O script percorre um conjunto de imagens do dataset (geralmente algumas centenas).
2. Cada imagem é:
   - Convertida para o espaço de cores **LAB** (`cv2.COLOR_RGB2LAB`)
   - Segmentada para obter apenas a área da folha (usando HSV)
3. Os pixels de todas as folhas são amostrados e combinados em um único array global.
4. É aplicado o algoritmo **MiniBatchKMeans** (`k=6`, por padrão), que encontra os *clusters de cor mais frequentes*.
5. Os centros (médias das cores LAB de cada cluster) são salvos em um arquivo `.npy`:

---

### Saídas geradas pelo módulo

- `lesion_mask.png` → Máscara binária da área doente  
- `leaf_mask.png` → Máscara da folha  
- `cluster_vis.png` → Visualização dos clusters coloridos  
- `damage_amount` → Percentual estimado de dano  

---

## 4. Integração com o Aplicativo Flutter

Repositório do app: <https://github.com/Enzolinn/leaf_health_app/tree/main>\
\
O app deve enviar um `POST` para `/predict` com o seguinte formato:

### **Requisição**

POST /predict
Content-Type: multipart/form-data
file=<imagem_da_folha>

### **Resposta**

```json
{
  "plant": "Tomato",
  "disease": "Late_blight",
  "scores": {"Tomato___Late_blight": 0.92, ...},
  "leaf_mask_b64": "...",
  "leaf_cluster_vis_b64": "...",
  "damage_amount": 23.7
}
```

---

### Tecnologias e Bibliotecas

- `PyTorch` → treinamento do modelo CNN
- `Torchvision` → datasets e backbones
- `FastAPI` → API de inferência
- `OpenCV` → manipulação e segmentação de imagens
- `scikit-learn` → algoritmo K-Means
- `NumPy` → processamento numérico

---

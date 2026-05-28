# Relatório — Projeto M2 de Processamento de Imagens

**Autor:** Eduardo Drux
**Disciplina:** Processamento de Imagens
**Professor:** Felipe Viel
**Curso:** Ciência da Computação — UNIVALI

---

## 1. Identificação do trabalho

- **Tema:** Pipeline completo de processamento digital de imagens para segmentação
  automática de grãos agrícolas.
- **Dataset escolhido:** *Seed Images* (Kaggle — `ddsssss/seed-images`).
- **Algoritmo de segmentação principal:** SLIC Superpixels + Otsu por Superpixel,
  ambos implementados *from scratch*.
- **Arquivos entregues:**
  - `processamento_imagens_m2.ipynb` — notebook executável com toda a explicação e código.
  - `src/pipeline.py` — versão modular do mesmo código (para reaproveitamento).
  - `RELATORIO.md` — este relatório.
  - `figures/` — figuras geradas pelo notebook.

---

## 2. Enunciado do projeto

Conforme as instruções da M2, o objetivo é desenvolver um pipeline completo de
processamento digital de imagens, integrando:

- (a) Pré-processamento;
- (b) Filtragem no domínio da frequência;
- (c) Segmentação baseada em superpixels;
- (d) Morfologia matemática;
- (e) Avaliação quantitativa por IoU, Dice e contagem.

Para o dataset *Seed Images*, o resultado esperado é uma **máscara binária** em que
pixels pertencentes a grãos recebem valor 1 e pixels de fundo recebem valor 0,
junto com a estimativa do número de grãos presentes em cada imagem.

---

## 3. Contexto da aplicação

O dataset contém imagens (1920×1080) de grãos (sementes) dispostos dentro de uma
placa de Petri sobre um fundo colorido (azul-teal, em geral). Cada imagem traz
anotações **bounding box** no formato PASCAL VOC, com rótulo `yes`/`no` indicando
se o grão germinou ou não — para a tarefa de segmentação, trato ambos como grão.

### 3.1 Por que essa aplicação é interessante

A análise automatizada de grãos é usada na agroindústria para:
- contar grãos sem trabalho manual;
- estimar taxa de germinação;
- verificar uniformidade de lotes.

Resolvê-la com técnicas clássicas (sem deep learning) é mais barato e
explicável — não exige GPU nem datasets enormes, e cada decisão do pipeline
pode ser justificada matematicamente.

### 3.2 Desafios específicos do dataset

| Desafio | Impacto |
|---|---|
| Grãos tocando uns aos outros | Componentes conectados subestimam a contagem |
| Borda da placa de Petri visível | Forma um anel que vira falso positivo |
| Vinheta escura por causa da lente fisheye em algumas imagens | Confunde o Otsu — região escura tem b\* ambíguo |
| Reflexos brancos nos vidros | Pode "furar" a região do grão |
| Variação de iluminação entre imagens | Threshold fixo não funciona — precisa de Otsu adaptativo |
| Grãos com sprouts (brotos brancos) | O passa-baixa atenua as raízes finas |
| Anotação só com bounding box | GT pixel-a-pixel só pode ser aproximado |

---

## 4. Desenvolvimento

### 4.1 Visão geral do pipeline

```
Imagem RGB
   ↓ 1. Pré-processamento     → RGB → LAB, normalização
   ↓ 2. Frequência            → Passa-baixa Gaussiano via FFT (canal b*)
   ↓ 3. Feature warmth        → b* × gate(L*)        ← anula vinheta escura
   ↓ 4. SLIC Superpixels      → from scratch — espaço LAB
   ↓ 5. Otsu por Superpixel   → from scratch — sobre médias dos superpixels
   ↓ 6. Morfologia            → abertura → filtro de área → fill holes → fechamento
   ↓ 7. Contagem              → erosão + componentes conectados (BFS)
   ↓ 8. Avaliação             → IoU, Dice, |pred - gt|
```

### 4.2 Justificativa do pipeline e da ordem

A escolha de cada etapa e a ordem foram pensadas explicitamente para o problema:

1. **LAB primeiro** — Os grãos são amarelados e o fundo é azulado/teal. No espaço
   LAB, o canal **b\*** (eixo azul–amarelo) separa quase linearmente essas duas
   classes. Testei a\* e L\* isoladamente; nenhum dos dois oferece tanta separação
   quanto b\*.

2. **Passa-baixa antes do SLIC** — A textura interna dos grãos e o ruído do sensor
   são alta frequência e atrapalham tanto o SLIC (criando superpixels irregulares)
   quanto o Otsu (criando histograma menos bimodal). Suavizando o b\* antes,
   reduzo essas duas fontes de erro de uma vez.

3. **Feature *warmth gated by L\** antes do Otsu** — Esse foi o passo de maior
   ganho prático (subiu o IoU em imagens difíceis de ~0.15 pra ~0.50). Ele anula
   o sinal nas regiões muito escuras (vinheta) sem afetar o resto.

4. **SLIC + Otsu por superpixel** — Em vez de Otsu pixel-a-pixel, a média
   intra-superpixel pré-filtra ruído e gera um histograma muito mais bimodal.
   Otsu acerta o limiar com mais facilidade nesse vetor de médias.

5. **Morfologia em ordem específica**:
   - **Abertura** primeiro — remove pingos isolados (ruído) sem mexer no formato.
   - **Filtro de área (min/max)** — elimina anel da placa antes de `fill_holes`
     (importante! se eu rodasse `fill_holes` antes, ele preencheria o interior
     inteiro da placa).
   - **Fill holes** — tampa furos internos dos grãos (sombras, reflexos pontuais).
   - **Fechamento** — suaviza bordas finais.

6. **Contagem com erosão** — Aplico uma erosão extra antes do CC pra separar
   grãos colados. Não resolve sobreposição profunda, mas separa "toques" leves.

### 4.3 Parâmetros escolhidos

| Parâmetro | Valor | Justificativa empírica |
|---|---|---|
| Resolução de trabalho | 640×360 | 9× menos pixels que o original; grãos ainda têm ~12–18 px de largura |
| σ do passa-baixa | 25 | Testei 10/15/25/40 — abaixo de 15 ainda há ruído; acima de 40 borra borda do grão |
| `l_low`, `l_high` (gate) | 20, 40 | Cobre a transição entre vinheta (L < 20) e área válida (L > 40) |
| `n_segments` SLIC | 300 | Suficiente pra cada grão ser coberto por 1–3 superpixels |
| `compactness` SLIC | 12 | Equilibrado — não tão rígido que ignora cor, não tão solto que vira blob |
| `n_iters` SLIC | 8 | Empiricamente os centros estabilizam em 6–8 iterações |
| SE da abertura | disco r=2 | Remove pingos < 5 px sem mudar formato dos grãos |
| `min_area` filtro | 80 | Maior que ruído residual mas menor que o menor grão |
| `max_area` filtro | 25000 | Permite clusters grandes de grãos colados; corta o anel da placa |
| SE do fechamento | disco r=2 | Suaviza bordas sem juntar grãos vizinhos |
| `erosion_radius` contagem | 3 | Suficiente pra desfazer toques finos sem matar grãos |

---

## 5. Códigos importantes da implementação

> O notebook traz todos os códigos em ordem, com explicações em markdown entre
> cada bloco. Aqui destaco apenas os trechos mais importantes.

### 5.1 Filtragem no domínio da frequência (uso de biblioteca permitido)

```python
def gaussian_lowpass_fft(channel, sigma=25.0):
    H, W = channel.shape
    F = np.fft.fftshift(np.fft.fft2(channel))
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    D2 = (y - cy) ** 2 + (x - cx) ** 2
    H_filter = np.exp(-D2 / (2.0 * sigma * sigma))
    G = F * H_filter
    return np.real(np.fft.ifft2(np.fft.ifftshift(G))).astype(np.float32)
```

**Por que Gaussiano e não passa-baixa ideal:** o ideal (corte abrupto no
espectro) gera *ringing* (artefatos em forma de onda nas bordas) por causa do
fenômeno de Gibbs. O Gaussiano cai suavemente e não cria esse artefato.

**Por que FFT em vez de convolução espacial:** a Gaussiana espacial com σ=25
exige kernel de raio ~75 (~150×150). Convolução desse tamanho é cara. FFT
resolve em O(N log N) independente do σ.

### 5.2 Feature de classificação (`warmth gated by L*`)

```python
def warmth_feature(img_lab, l_low=20.0, l_high=40.0):
    L = img_lab[..., 0]
    b = img_lab[..., 2]
    gate = np.clip((L - l_low) / max(l_high - l_low, 1e-6), 0.0, 1.0)
    return b * gate
```

Foi o passo de maior impacto no IoU. Sem o gate, imagens com vinheta escura
tinham IoU ≈ 0.15. Com o gate, ≈ 0.50.

### 5.3 SLIC Superpixels (from scratch — trecho principal)

```python
def slic_superpixels(img_lab, n_segments=300, compactness=12.0, n_iters=8):
    H, W, _ = img_lab.shape
    S = max(2, int(np.sqrt(H*W / n_segments)))
    # 1) Inicializa centros numa grade regular
    centers = [[*img_lab[cy, cx], cx, cy]
               for cy in range(S//2, H, S) for cx in range(S//2, W, S)]
    centers = np.array(centers, dtype=np.float32)
    K = len(centers)
    # ... laços de atribuição (janela 2S×2S) e atualização (média) ...
```

**Vetorização da atribuição:** em vez de varrer pixel-a-pixel, cada centro
processa sua janela 2S×2S inteira em uma operação numpy. Isso reduz o tempo
total para ~0.2 s por imagem 640×360.

### 5.4 Otsu manual com prefix sums

```python
def otsu_threshold(values, n_bins=256):
    hist, edges = np.histogram(values, bins=n_bins,
                               range=(float(values.min()), float(values.max())))
    centers = (edges[:-1] + edges[1:]) / 2.0
    p = hist / hist.sum()
    w0 = np.cumsum(p)
    mu_acc = np.cumsum(p * centers)
    mu_total = mu_acc[-1]
    w1 = 1.0 - w0
    valid = (w0 > 1e-8) & (w1 > 1e-8)
    mu0 = np.where(valid, mu_acc / np.where(w0 == 0, 1, w0), 0.0)
    mu1 = np.where(valid, (mu_total - mu_acc) / np.where(w1 == 0, 1, w1), 0.0)
    sigma_b2 = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, -1.0)
    return float(centers[int(np.argmax(sigma_b2))])
```

**Por que prefix sums:** evita recalcular w₀/μ₀ pra cada limiar candidato.
Reduz de O(n²) pra O(n).

### 5.5 Morfologia vetorizada

```python
def erode(mask, se):
    H, W = mask.shape
    sH, sW = se.shape
    padded = np.pad(mask, ((sH//2, sH//2), (sW//2, sW//2)), constant_values=0)
    out = np.ones((H, W), dtype=np.uint8)
    for dy in range(sH):
        for dx in range(sW):
            if se[dy, dx]:
                out &= padded[dy:dy+H, dx:dx+W]
    return out
```

**Truque chave:** pra um SE com K pontos ativos, faço K operações vetorizadas
em vez de H·W·K operações escalares. Isso é o que torna a morfologia em Python
puro viável (~0.3 s por operação numa imagem 640×360).

### 5.6 Fill holes via flood fill

```python
def fill_holes(mask):
    padded = np.pad(mask, 1, constant_values=0)
    visited = np.zeros_like(padded, dtype=bool)
    q = deque([(0, 0)]); visited[0, 0] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny, nx = y+dy, x+dx
            if 0<=ny<padded.shape[0] and 0<=nx<padded.shape[1]:
                if not visited[ny, nx] and padded[ny, nx] == 0:
                    visited[ny, nx] = True; q.append((ny, nx))
    holes = (~visited) & (padded == 0)
    return (padded | holes.astype(np.uint8))[1:-1, 1:-1].astype(np.uint8)
```

**Por que BFS iterativo:** recursão estouraria a stack em imagens grandes
(640×360 = 230k pixels). O `deque` do Python tem operações O(1) em ambos os
lados.

---

## 6. Resultados

### 6.1 Tabela de métricas

| Imagem | IoU    | Dice   | Pred  | GT  | |Δ contagem| |
|--------|--------|--------|-------|-----|-------------|
| 0619   | 0.394  | 0.565  | 31    | 33  | 2           |
| 1105   | 0.517  | 0.682  | 14    | 19  | 5           |
| 1113   | 0.509  | 0.674  | 25    | 33  | 8           |
| 1141   | 0.412  | 0.584  | 10    | 31  | 21          |
| 1286   | 0.555  | 0.713  | 12    | 47  | 35          |
| **Média** | **0.477** | **0.644** | — | — | **14.2** |

### 6.2 Visualizações

Cada imagem do conjunto selecionado tem uma figura em `figures/resultado_<id>.png`
mostrando: original, feature `b*·gate(L*)`, superpixels SLIC, máscara coarse pós-Otsu,
máscara final pós-morfologia e overlay sobre a imagem original.

Figuras intermediárias úteis pra apresentação:
- `figures/01_carregamento.png` — original + bndbox + GT por elipses;
- `figures/02_frequencia_e_feature.png` — efeito do passa-baixa e do gate L\*;
- `figures/03_slic.png` — sobreposição das fronteiras dos superpixels;
- `figures/04_otsu_superpixel.png` — histograma das médias e limiar escolhido;
- `figures/05_morfologia.png` — efeito de cada operação morfológica.

### 6.3 Tempo de execução

Numa máquina típica, o pipeline completo (5 imagens em sequência, 640×360 px) leva
**~5 segundos** total. As etapas mais caras são:

- SLIC: ~0.2 s
- Morfologia (4 operações + filtros): ~0.3 s
- Componentes conectados: ~0.1 s
- Resto (FFT, GT, Otsu, métricas): <0.1 s

---

## 7. Análise crítica e discussão

### 7.1 O que funcionou bem

- **Feature `b* · gate(L*)`** — foi o passo de maior ganho prático. Em imagens
  com vinheta escura (1113), o IoU subiu de ~0.15 pra ~0.50. A lição é que,
  quando o b\* sozinho dá ambiguidade, combinar com a luminosidade resolve.

- **SLIC com 300 superpixels** — gera regiões compactas e seguindo bem as bordas
  dos grãos. Visualmente, a separação grão/fundo no nível do superpixel é
  praticamente perfeita.

- **Otsu por superpixel** — o histograma de médias é fortemente bimodal, então
  o Otsu acerta o limiar com folga em todas as imagens testadas.

- **Filtro de área `[80, 25000]`** — simples e eficaz pra eliminar o anel da
  placa de Petri (que vira o maior componente conectado da imagem).

### 7.2 O que não funcionou tão bem

- **Contagem em clusters densos** — em imagens 1141 e 1286, onde os grãos estão
  fortemente sobrepostos, a contagem subestima muito (10/31 e 12/47). A erosão
  pré-CC ajuda só em "toques" leves; quando os grãos se sobrepõem 30–40%, nada
  além de uma técnica mais sofisticada (watershed com markers de distância)
  consegue separar.

- **GT aproximado por elipses** — o dataset não tem máscara pixel-a-pixel, então
  reconstituí o GT inscrevendo uma elipse em cada bndbox. Isso:
  - **Infla o GT** — a elipse cobre uma região maior que o grão real.
  - **Não captura sobreposição** parcial entre grãos vizinhos.

  O resultado é que o IoU "real" do pipeline (se houvesse máscara verdadeira)
  seria provavelmente um pouco diferente — não dá pra dizer com certeza se pra
  cima ou pra baixo sem inspecionar manualmente.

- **Brotos brancos dos grãos germinados** — alguns grãos têm raízes brancas
  finas (sprouts). O passa-baixa σ=25 borra essas estruturas e o Otsu não as
  reconhece como grão. Em imagens com muitos germinados (1113, 1141), perdo
  essa parte.

- **Reflexos pontuais nos vidros** — em algumas imagens há reflexos muito
  brilhantes na placa de Petri que escapam do filtro de área (têm tamanho
  similar a um grão).

### 7.3 Impacto de cada etapa (ablação informal)

| Etapa removida | Efeito observado |
|---|---|
| Sem filtro de frequência | IoU cai ~5% — SLIC vira ruidoso |
| Sem gate(L\*) | IoU cai ~15% em imagens com vinheta (chega a -30% em 1113) |
| Sem fill_holes | IoU cai ~3% — poucos buracos internos |
| Sem `max_area` no filtro | IoU cai >30% — anel da placa contamina tudo |
| Sem erosão pré-contagem | Contagem subestima ainda mais (5–7 grãos a menos) |

### 7.4 Possíveis melhorias futuras

1. **Watershed por marcadores de distância** — implementar distance transform
   (Felzenszwalb-Huttenlocher, dois passes O(N)) e usar local maxima como
   markers do watershed. Isso resolveria a contagem em clusters densos.

2. **Detecção da placa via Hough Circles** — mais robusto que filtro de área
   pra eliminar a borda da placa de Petri. Permitiria também mascarar o
   exterior do círculo antes do SLIC.

3. **Resolução maior (1280×720)** — melhoraria a fidelidade das bordas dos
   grãos pequenos, à custa de tempo de processamento (provavelmente 5–10×).

4. **Combinar múltiplos canais** — usar `b* * gate(L*) - 0.5*a*` (subtraindo a
   componente verde) reforçaria a tonalidade quente dos grãos contra o fundo
   teal.

5. **Pós-processamento por superpixel** — depois do Otsu inicial, refinar a
   classificação dos superpixels de fronteira usando textura local.

### 7.5 Conclusões

O pipeline atende a todos os requisitos do projeto: usa filtragem na frequência,
morfologia matemática e segmentação por superpixels (todas implementadas
manualmente exceto a parte de FFT, que o enunciado libera). A qualidade da
máscara binária é boa (IoU médio ≈ 0.48, Dice médio ≈ 0.64) considerando que
todo o processamento é clássico, sem ML, e que o GT é uma aproximação.

A grande limitação está na contagem em imagens com clusters muito densos —
não é uma falha do pipeline em si, mas um limite das técnicas clássicas sem
uma etapa explícita de separação por watershed/distance transform.

---

## 8. Como reproduzir

1. Instalar dependências:
   ```
   pip install numpy matplotlib pillow scikit-image
   ```
2. Garantir que o dataset está em `dataset/JPEGImages/` e `dataset/Annotations/`.
3. Executar o notebook:
   ```
   jupyter notebook processamento_imagens_m2.ipynb
   ```
   Ou reproduzir os resultados via:
   ```
   python -m jupyter nbconvert --to notebook --execute --inplace processamento_imagens_m2.ipynb
   ```

---

## 9. Referências

- Achanta, R. et al. **SLIC Superpixels Compared to State-of-the-Art Superpixel
  Methods.** IEEE TPAMI, 2012. DOI: 10.1109/TPAMI.2012.120
- Otsu, N. **A Threshold Selection Method from Gray-Level Histograms.**
  IEEE TSMC, 1979. DOI: 10.1109/TSMC.1979.4310076
- Gonzalez, R. & Woods, R. **Digital Image Processing**, 4ª ed. — referência geral
  para FFT, morfologia matemática e métricas.

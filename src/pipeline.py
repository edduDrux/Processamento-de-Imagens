"""
Pipeline de segmentação de grãos agrícolas.

Esse arquivo guarda toda a lógica que eu uso no notebook. Deixei tudo
separado em funções pra ficar mais fácil de testar e pra não poluir a
parte de visualização do notebook.

Convenções que eu segui:
- Toda imagem entra como uint8 RGB (vinda do PIL).
- Trabalho a maior parte do tempo no espaço LAB porque o canal b* (eixo
  azul–amarelo) separa muito bem o grão (amarelado) do fundo (azulado/verde).
- Máscaras são uint8 com valores 0 ou 1 — assim posso usar &, |, ~ direto.

Bibliotecas externas usadas e por quê:
- numpy: contas matriciais — base de tudo.
- PIL: só pra carregar o JPG e redimensionar.
- xml.etree: pra ler as bndbox do XML do dataset.
- skimage.color.rgb2lab: conversão RGB→LAB. O enunciado proíbe funções
  prontas de morfologia e segmentação, mas conversão de espaço de cor é
  pré-processamento — implementar a fórmula CIE LAB do zero não agrega
  nada ao trabalho e a skimage é a referência canônica.
- numpy.fft: filtragem no domínio da frequência (explicitamente liberada
  pelo enunciado).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


# =============================================================================
# Bloco 1 — Carregamento de imagens e anotações
# =============================================================================

def load_image(images_path: str, image_id: str, resize_to=(640, 360)) -> np.ndarray:
    """Carrega uma imagem do dataset, converte pra RGB e redimensiona.

    Redimensiono pra (640, 360) porque o original é 1920x1080. Em pixels
    isso é ~8x menos dados — o SLIC e as operações morfológicas em Python
    puro ficam viáveis sem perder a estrutura dos grãos (que ainda ocupam
    dezenas de pixels nessa resolução).
    """
    path = os.path.join(images_path, f"{image_id}.jpg")
    img = Image.open(path).convert("RGB")
    img = img.resize(resize_to, Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def parse_annotation(annotations_path: str, image_id: str, target_w: int, target_h: int):
    """Lê o XML do PASCAL VOC e devolve as bndbox já escaladas pro tamanho do redimensionamento.

    Retorna lista de tuplas (xmin, ymin, xmax, ymax, label) onde label é "yes" ou "no"
    (o dataset usa essa convenção pra dizer se o grão germinou ou não — pra
    segmentação eu trato todos como grão).
    """
    path = os.path.join(annotations_path, f"{image_id}.xml")
    tree = ET.parse(path)
    root = tree.getroot()

    orig_w = int(root.find("size/width").text)
    orig_h = int(root.find("size/height").text)

    sx = target_w / orig_w
    sy = target_h / orig_h

    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(int(bbox.find("xmin").text) * sx)
        ymin = int(int(bbox.find("ymin").text) * sy)
        xmax = int(int(bbox.find("xmax").text) * sx)
        ymax = int(int(bbox.find("ymax").text) * sy)
        label = obj.find("name").text
        boxes.append((xmin, ymin, xmax, ymax, label))
    return boxes


def build_gt_mask(boxes, shape) -> np.ndarray:
    """Monta uma máscara binária de ground truth a partir das bndbox.

    O dataset só fornece bounding box, então pra calcular IoU/Dice eu
    preciso converter cada box numa máscara aproximada. Em vez de
    preencher o retângulo inteiro (que daria um GT bem inflado), aproximo
    cada grão por uma elipse inscrita na bndbox — fica bem mais próximo
    da forma alongada que os grãos têm.

    Isso obviamente não é um ground truth perfeito (ainda não é a forma
    real do grão), mas serve como referência consistente pra comparar
    diferentes configurações do pipeline entre si.
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    y_idx = np.arange(H).reshape(-1, 1)
    x_idx = np.arange(W).reshape(1, -1)

    for (xmin, ymin, xmax, ymax, _label) in boxes:
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        rx = max(1.0, (xmax - xmin) / 2.0)
        ry = max(1.0, (ymax - ymin) / 2.0)
        ellipse = ((x_idx - cx) / rx) ** 2 + ((y_idx - cy) / ry) ** 2 <= 1.0
        mask |= ellipse.astype(np.uint8)
    return mask


# =============================================================================
# Bloco 2 — Pré-processamento
# =============================================================================

def to_lab(img_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 [0,255] -> LAB float32.

    L vai de 0 a 100 (luminosidade), a* e b* tipicamente entre -128 e +127.
    No nosso caso o canal b* é o mais útil: grãos amarelados ficam com b*
    positivo alto e o fundo azulado/esverdeado fica com b* baixo (ou
    negativo). É praticamente uma separação linear no histograma.
    """
    return rgb2lab(img_rgb.astype(np.float32) / 255.0).astype(np.float32)


def normalize01(channel: np.ndarray) -> np.ndarray:
    """Normaliza um canal pra [0,1] usando min/max.

    Uso isso pra visualização e pra rodar Otsu sem precisar saber o
    intervalo real do canal.
    """
    cmin, cmax = float(channel.min()), float(channel.max())
    if cmax - cmin < 1e-8:
        return np.zeros_like(channel)
    return (channel - cmin) / (cmax - cmin)


# =============================================================================
# Bloco 3 — Domínio da frequência (uso de biblioteca permitido)
# =============================================================================

def gaussian_lowpass_fft(channel: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Filtra um canal com passa-baixa Gaussiana no domínio da frequência.

    Por que passa-baixa: a textura interna dos grãos e o ruído do
    micrótomo/câmera viram alta frequência. Suavizando antes do SLIC,
    os superpixels saem mais "limpos" e o canal b* fica mais uniforme
    dentro de cada grão.

    Por que Gaussiana e não box/ideal: o passa-baixa ideal (corte abrupto
    no espectro) introduz ringing (anéis no entorno das bordas) por causa
    do Gibbs. O Gaussiano cai suave e não cria esse artefato.

    sigma controla a largura — quanto maior, mais frequências altas a
    Gaussiana atenua, logo a imagem fica mais borrada. sigma=25 num
    domínio 640x360 corta detalhes < ~12 px, que é menor que um grão.
    """
    H, W = channel.shape
    # FFT centralizada — fftshift coloca DC no meio, fica mais intuitivo
    F = np.fft.fftshift(np.fft.fft2(channel))

    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    D2 = (y - cy) ** 2 + (x - cx) ** 2  # distância ao quadrado do centro
    H_filter = np.exp(-D2 / (2.0 * sigma * sigma))

    G = F * H_filter
    out = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    return out.astype(np.float32)


# =============================================================================
# Bloco 4 — SLIC Superpixels (from scratch)
# =============================================================================

def slic_superpixels(img_lab: np.ndarray,
                     n_segments: int = 300,
                     compactness: float = 12.0,
                     n_iters: int = 8) -> np.ndarray:
    """SLIC implementado do zero — só numpy.

    Resumo do algoritmo (Achanta et al., TPAMI 2012):
      1. Inicializa K centros distribuídos numa grade regular sobre a imagem.
         Cada centro carrega [L, a, b, x, y] — cor LAB + posição.
      2. Calcula o espaçamento S = sqrt(N/K). Cada centro só "enxerga" pixels
         dentro de uma janela 2S x 2S — isso é o que faz o SLIC ser O(N) e
         não O(N*K) como o k-means clássico.
      3. Atribui cada pixel ao centro mais próximo, usando a distância
         híbrida D = sqrt(dc^2 + (m/S)^2 * ds^2), onde dc é distância de cor
         e ds é distância espacial. m é o parâmetro "compactness": quanto
         maior, mais os superpixels viram quadrados regulares.
      4. Recalcula cada centro como a média dos pixels atribuídos.
      5. Repete 3-4 por n_iters iterações.

    Por que SLIC e não outro algoritmo de superpixel:
      - Felzenszwalb produz regiões muito irregulares e instáveis com a
        textura do fundo.
      - Watershed sem marcadores explode em milhares de bacias.
      - SLIC é o mais "comportado" pra superfícies relativamente uniformes
        como a nossa (fundo liso + objetos compactos), e é simples de
        explicar passo a passo (que o trabalho exige).

    Por que LAB e não RGB: a distância euclidiana em LAB é perceptualmente
    uniforme — ΔE ≈ percepção de diferença de cor. Em RGB, diferenças
    iguais em valor numérico não correspondem a diferenças iguais
    perceptuais, então o SLIC erra mais.
    """
    H, W, _ = img_lab.shape
    N = H * W
    S = int(np.sqrt(N / n_segments))
    S = max(S, 2)

    # --- 1) Inicialização dos centros numa grade regular ---
    cy_grid = np.arange(S // 2, H, S)
    cx_grid = np.arange(S // 2, W, S)
    centers = []
    for cy in cy_grid:
        for cx in cx_grid:
            l, a, b = img_lab[cy, cx]
            centers.append([float(l), float(a), float(b), float(cx), float(cy)])
    centers = np.array(centers, dtype=np.float32)
    K = len(centers)

    labels = -np.ones((H, W), dtype=np.int32)
    distances = np.full((H, W), np.inf, dtype=np.float32)

    # Pré-calculo das coords de todos os pixels — economiza tempo dentro do loop
    ys_full, xs_full = np.indices((H, W), dtype=np.int32)

    m_over_S = compactness / float(S)

    for _it in range(n_iters):
        labels.fill(-1)
        distances.fill(np.inf)

        # --- 2-3) Atribuição ---
        for k in range(K):
            cl, ca, cb, cx, cy = centers[k]
            icx, icy = int(cx), int(cy)
            y0 = max(0, icy - S); y1 = min(H, icy + S + 1)
            x0 = max(0, icx - S); x1 = min(W, icx + S + 1)

            window = img_lab[y0:y1, x0:x1]
            dl = window[..., 0] - cl
            da = window[..., 1] - ca
            db = window[..., 2] - cb
            d_color = dl * dl + da * da + db * db

            ys = ys_full[y0:y1, x0:x1] - cy
            xs = xs_full[y0:y1, x0:x1] - cx
            d_space = xs * xs + ys * ys

            # D ao quadrado — não preciso da raiz só pra comparar
            D = d_color + (m_over_S ** 2) * d_space

            local_d = distances[y0:y1, x0:x1]
            update = D < local_d
            local_d[update] = D[update]
            labels[y0:y1, x0:x1] = np.where(update, k, labels[y0:y1, x0:x1])

        # --- 4) Atualização dos centros (média dos pixels atribuídos) ---
        flat = labels.ravel()
        counts = np.bincount(flat, minlength=K).astype(np.float32)
        counts_safe = np.where(counts == 0, 1.0, counts)

        new_centers = np.empty_like(centers)
        for ch in range(3):
            new_centers[:, ch] = np.bincount(flat, weights=img_lab[..., ch].ravel(), minlength=K) / counts_safe
        new_centers[:, 3] = np.bincount(flat, weights=xs_full.ravel().astype(np.float32), minlength=K) / counts_safe
        new_centers[:, 4] = np.bincount(flat, weights=ys_full.ravel().astype(np.float32), minlength=K) / counts_safe
        centers = new_centers

    # Em casos extremos algum pixel pode ficar sem label se a janela do
    # centro não cobrir ele (não acontece com S correto, mas vou garantir)
    if (labels < 0).any():
        # Vizinho mais próximo bobo: copia do pixel acima/à esquerda
        for y in range(H):
            for x in range(W):
                if labels[y, x] < 0:
                    if y > 0 and labels[y - 1, x] >= 0:
                        labels[y, x] = labels[y - 1, x]
                    elif x > 0 and labels[y, x - 1] >= 0:
                        labels[y, x] = labels[y, x - 1]
                    else:
                        labels[y, x] = 0

    return labels


# =============================================================================
# Bloco 5 — Otsu por superpixel (from scratch)
# =============================================================================

def superpixel_mean(labels: np.ndarray, channel: np.ndarray) -> np.ndarray:
    """Média do canal dentro de cada superpixel.

    Faço isso com bincount em vez de loop — fica O(N) em vez de O(N*K).
    """
    K = int(labels.max()) + 1
    flat = labels.ravel()
    weights = channel.ravel().astype(np.float64)
    sums = np.bincount(flat, weights=weights, minlength=K)
    counts = np.bincount(flat, minlength=K).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)
    return sums / counts


def otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu manual: maximiza variância inter-classes em 1D.

    Passos:
      1. Monta histograma com n_bins bins.
      2. Para cada threshold candidato t, separa em duas classes.
      3. Calcula sigma_b^2 = w0*w1*(mu0-mu1)^2.
      4. Pega o t que dá maior sigma_b^2.

    Por que Otsu e não threshold fixo: o b* varia bastante entre imagens
    (iluminação diferente, vidro com reflexo, etc.). Um valor fixo de b*
    funcionaria pra uma imagem e falharia em outras. Otsu se adapta
    automaticamente porque escolhe o ponto que melhor separa os dois
    modos do histograma.

    Por que aplicar Otsu sobre superpixels e não sobre pixels: a média
    por superpixel já filtra muito do ruído de pixel-a-pixel, então o
    histograma de médias é bimodal de forma muito mais clara que o
    histograma de pixels individuais — Otsu acerta o limiar com mais
    facilidade.
    """
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-8:
        return vmin

    hist, edges = np.histogram(values, bins=n_bins, range=(vmin, vmax))
    centers = (edges[:-1] + edges[1:]) / 2.0
    p = hist.astype(np.float64) / hist.sum()

    # Acumulados — somas prefix permitem calcular w0/mu0 sem refazer somas
    w0 = np.cumsum(p)
    mu_acc = np.cumsum(p * centers)
    mu_total = mu_acc[-1]

    # Evita div por zero nas pontas
    w1 = 1.0 - w0
    valid = (w0 > 1e-8) & (w1 > 1e-8)

    mu0 = np.where(valid, mu_acc / np.where(w0 == 0, 1, w0), 0.0)
    mu1 = np.where(valid, (mu_total - mu_acc) / np.where(w1 == 0, 1, w1), 0.0)
    sigma_b2 = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, -1.0)

    best_idx = int(np.argmax(sigma_b2))
    return float(centers[best_idx])


def otsu_per_superpixel(labels: np.ndarray, feature: np.ndarray) -> np.ndarray:
    """Devolve uma máscara binária no tamanho da imagem.

    Etapas:
      1. Calcula a média do canal dentro de cada superpixel.
      2. Aplica Otsu sobre o vetor de médias.
      3. Classifica cada superpixel acima do limiar como "grão" (1).
      4. "Pinta" a máscara final substituindo cada label pelo veredito.
    """
    sp_mean = superpixel_mean(labels, feature)
    t = otsu_threshold(sp_mean)
    sp_class = (sp_mean > t).astype(np.uint8)
    return sp_class[labels]


def warmth_feature(img_lab: np.ndarray, l_low: float = 20.0, l_high: float = 40.0) -> np.ndarray:
    """Feature de "amarelado iluminado" — combina b* e L*.

    Por que combinar: o canal b* sozinho funciona bem quando a iluminação
    é uniforme, mas algumas imagens do dataset têm vinheta escura na borda
    (lente fisheye + iluminação ruim). Nessas regiões escuras o b* fica
    próximo de zero ou levemente positivo, e o Otsu pode classificar elas
    como grão por engano.

    A solução: faço um "portão" (gate) baseado em L*. Quando L* < l_low
    (regiões escuras), o sinal vai a zero. Quando L* > l_high, passa
    completamente. Entre os dois, transição linear. Isso anula a
    contribuição da vinheta sem afetar o resto.

    O resultado é um canal onde só pixels que são amarelados E
    suficientemente iluminados (= grãos de verdade) têm valor alto.
    """
    L = img_lab[..., 0]
    b = img_lab[..., 2]
    gate = np.clip((L - l_low) / max(l_high - l_low, 1e-6), 0.0, 1.0)
    return b * gate


# =============================================================================
# Bloco 6 — Morfologia matemática (from scratch)
# =============================================================================

def disk_se(radius: int) -> np.ndarray:
    """Elemento estruturante em forma de disco (raio inteiro).

    Uso disco em vez de quadrado porque os grãos não têm direção
    preferencial — um SE quadrado deixa "cantos" na máscara e enviesa
    a forma; o disco trata todas as direções igual.
    """
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def erode(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Erosão binária: pixel só fica 1 se TODOS os vizinhos do SE forem 1.

    Implementação vetorizada — em vez de um loop pixel-a-pixel que
    seria horroroso em Python, eu desloco a máscara pra cada offset do
    SE e faço AND lógico de tudo. Pra um SE com k pontos ativos, são k
    operações vetorizadas em vez de H*W*k operações escalares.
    """
    H, W = mask.shape
    sH, sW = se.shape
    pad_h, pad_w = sH // 2, sW // 2
    padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)
    out = np.ones((H, W), dtype=np.uint8)
    for dy in range(sH):
        for dx in range(sW):
            if se[dy, dx]:
                out &= padded[dy:dy + H, dx:dx + W]
    return out


def dilate(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Dilatação binária: pixel vira 1 se ALGUM vizinho do SE for 1.

    Mesma ideia da erosão, só que OR no lugar de AND.
    """
    H, W = mask.shape
    sH, sW = se.shape
    pad_h, pad_w = sH // 2, sW // 2
    padded = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)
    out = np.zeros((H, W), dtype=np.uint8)
    for dy in range(sH):
        for dx in range(sW):
            if se[dy, dx]:
                out |= padded[dy:dy + H, dx:dx + W]
    return out


def opening(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Abertura = erode depois dilata. Tira ruído pequeno preservando forma."""
    return dilate(erode(mask, se), se)


def closing(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Fechamento = dilate depois erode. Tampa buracos pequenos e cola
    objetos quase-conectados."""
    return erode(dilate(mask, se), se)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Preenche buracos internos da máscara.

    Truque clássico: faço flood fill do fundo a partir das bordas da
    imagem. Tudo que NÃO foi alcançado e era zero originalmente é um
    buraco fechado dentro de algum objeto. Inverto isso e somo na
    máscara original.

    Por que precisamos disso: o canal b* dentro do grão pode ter alguns
    pixels mais escuros (sombra, reflexo, defeito) que o Otsu rejeita —
    isso vira buraco. Preencher recupera o grão inteiro.

    Implementação: BFS iterativo a partir do pixel (0,0) num padding de
    fundo. Recursão pura estouraria stack em imagens grandes.
    """
    H, W = mask.shape
    padded = np.pad(mask, 1, mode='constant', constant_values=0)
    visited = np.zeros_like(padded, dtype=bool)

    q = deque()
    q.append((0, 0))
    visited[0, 0] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < padded.shape[0] and 0 <= nx < padded.shape[1]:
                if not visited[ny, nx] and padded[ny, nx] == 0:
                    visited[ny, nx] = True
                    q.append((ny, nx))

    holes = (~visited) & (padded == 0)
    filled = padded | holes.astype(np.uint8)
    return filled[1:-1, 1:-1].astype(np.uint8)


# =============================================================================
# Bloco 7 — Componentes conectados (from scratch)
# =============================================================================

def count_grains_eroded(mask: np.ndarray, erosion_radius: int = 5, min_seed_area: int = 20) -> int:
    """Conta grãos aplicando uma erosão forte antes de rotular CCs.

    Problema: dois grãos colados aparecem como um componente só, então
    a contagem direta na máscara final subestima o total.

    Truque: erodo a máscara com um SE maior que a "ponte" entre grãos
    encostados. Cada grão individual ainda mantém um núcleo (porque o
    grão sozinho é mais grosso que a região de toque), mas a junção
    fica fina e some na erosão. Aí conto componentes nesse núcleo.

    erosion_radius=5 é empírico: grãos têm largura ~12-18px nessa
    resolução, então erodir 5px deixa um núcleo de ~2-8px de cada grão
    mas zera regiões de toque que tinham só ~2-3px de espessura.
    """
    eroded = erode(mask, disk_se(erosion_radius))
    # Remove restos minúsculos que sobraram da erosão (ruído)
    if min_seed_area > 0:
        eroded = filter_components_by_area(eroded, min_area=min_seed_area, max_area=None)
    _labels, n = connected_components(eroded)
    return n


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Rotula CCs com 4-conectividade usando BFS.

    Devolve (mapa_de_labels, n_componentes). Label 0 = fundo.

    Uso 4-conectividade porque com 8-conectividade dois grãos só
    "tocando" diagonalmente seriam contados como um só. Em 4-conectividade
    precisa ter pixels efetivamente colados, o que separa melhor os
    grãos que apenas se beijam.
    """
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    next_label = 0

    for y in range(H):
        for x in range(W):
            if mask[y, x] and labels[y, x] == 0:
                next_label += 1
                q = deque()
                q.append((y, x))
                labels[y, x] = next_label
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = next_label
                            q.append((ny, nx))
    return labels, next_label


def filter_components_by_area(mask: np.ndarray, min_area: int, max_area: int | None = None) -> np.ndarray:
    """Tira componentes fora do intervalo [min_area, max_area].

    Útil em dois cenários:
      - min_area: elimina pingos isolados que sobraram do Otsu (reflexos
        pontuais, ruído alto que escapou do passa-baixa).
      - max_area: elimina coisas grandes que não são grão. No nosso dataset
        a borda da placa de Petri vira um anel imenso e o Otsu deixa ele
        passar — descarto qualquer componente acima de max_area pra
        eliminar essa contaminação.
    """
    labels, n = connected_components(mask)
    if n == 0:
        return mask.copy()
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    keep = areas >= min_area
    if max_area is not None:
        keep &= areas <= max_area
    keep[0] = False  # fundo sempre fora
    return keep[labels].astype(np.uint8)


# Mantém o nome antigo pra retrocompatibilidade caso eu ainda chame em algum lugar
def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    return filter_components_by_area(mask, min_area, None)


# =============================================================================
# Bloco 8 — Métricas de avaliação (from scratch)
# =============================================================================

def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection over Union — clássico."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice = 2*|A ∩ B| / (|A| + |B|). É mais "perdoador" que IoU pra
    objetos pequenos — penaliza menos os erros nas bordas."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    s = pred_b.sum() + gt_b.sum()
    if s == 0:
        return 1.0
    return 2.0 * float(inter) / float(s)

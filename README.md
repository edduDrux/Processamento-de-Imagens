# Trabalho da M2 de Processamento de Imagens

**Autores:** Eduardo Sartori e Candido Fachini
**Disciplina:** Processamento de Imagens — UNIVALI
**Professor:** Felipe Viel

## Dataset escolhido

[Seed Images (Kaggle — ddsssss/seed-images)](https://www.kaggle.com/datasets/ddsssss/seed-images) — segmentação de grãos agrícolas.

## Fluxo de segmentação

O algoritmo de segmentação é o **Otsu por Superpixel** (_from scratch_): tiro a média da
feature dentro de cada superpixel e separo grão de fundo pelo limiar do Otsu. Os
superpixels quem gera é o **SLIC**, que também fiz na mão.

O SLIC não conta como um segundo algoritmo de segmentação — ele só monta os superpixels
que o Otsu por Superpixel usa de entrada (sozinho ele só dá oversegmentação, não uma
máscara grão/fundo).

```
RGB → LAB → FFT passa-baixa no b* → feature warmth = b*·gate(L*)
    → SLIC superpixels → Otsu por superpixel
    → morfologia (abertura + filtro de área + fill holes + fechamento)
    → componentes conectados → IoU/Dice/contagem
```

## Imagens selecionadas

`0619, 1105, 1113, 1141, 1286` — cobrindo casos fáceis (placa centralizada,
fundo limpo) e difíceis (vinheta escura, clusters densos de grãos).

## Como reproduzir

```bash
pip install numpy matplotlib pillow scikit-image
python -m jupyter nbconvert --to notebook --execute --inplace processamento_imagens_m2.ipynb
```

Ou abrir `processamento_imagens_m2.ipynb` no Jupyter/VS Code/Colab e rodar as células em ordem.

## Estrutura do repositório

```
.
├── processamento_imagens_m2.ipynb   ← notebook principal (executável)
├── RELATORIO.md                     ← relatório completo do projeto
├── src/pipeline.py                  ← versão modular do código (re-uso opcional)
├── dataset/
│   ├── JPEGImages/                  ← imagens originais (.jpg)
│   └── Annotations/                 ← anotações PASCAL VOC (.xml)
└── figures/                         ← figuras geradas pelo notebook
```

## Restrições do enunciado e como atendidas

| Restrição                                  | Atendimento                                                                    |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| Domínio da frequência pode usar biblioteca | Uso `numpy.fft` no filtro Gaussiano                                            |
| Morfologia: from scratch                   | Erosão, dilatação, abertura, fechamento e fill_holes implementados manualmente |
| Segmentação: from scratch                  | SLIC e Otsu por superpixel implementados manualmente                           |
| Sem Deep Learning / U-Net / YOLO           | OK — apenas técnicas clássicas                                                 |
| Métricas: IoU, Dice, contagem              | Todas implementadas (`iou_score`, `dice_score`, `count_grains_eroded`)         |

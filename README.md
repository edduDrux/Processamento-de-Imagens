# Trabalho da M2 de Processamento de Imagens

**Autor:** Eduardo Drux e Candido Neto
**Disciplina:** Processamento de Imagens — UNIVALI
**Professor:** Felipe Viel

## Dataset escolhido

[Seed Images (Kaggle — ddsssss/seed-images)](https://www.kaggle.com/datasets/ddsssss/seed-images) — segmentação de grãos agrícolas.

## Fluxo de segmentação

**SLIC Superpixels + Otsu por Superpixel** (ambos implementados _from scratch_).

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

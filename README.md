# Trabalho da M2 de processamento de imagens

### Dataset escolhido:

- "Segmentação de Grãos Agrículas"

### Fluxo de segmentação escolhido:

**SLIC + Otsu por Superpixel**

**Justificativa:**
- Fundo uniforme do dataset → histograma bimodal → Otsu encontra limiar ótimo entre grão e fundo
- SLIC opera no espaço LAB → resistente à variação de brilho entre imagens
- Otsu por superpixel (regional) é mais robusto que Otsu global em cenas com grãos tocando-se
- Cobre o critério de "Segmentação com superpixels" (20% da nota)

**Pipeline:**
1. Pré-processamento: RGB → LAB + Grayscale, normalização
2. Filtro passa-baixa Gaussiano (domínio da frequência / FFT)
3. SLIC Superpixels — from scratch
4. Classificação Otsu por Superpixel — from scratch
5. Morfologia: Abertura + Fechamento + Fill Holes — from scratch
6. Componentes conectados BFS (contagem de grãos) — from scratch
7. Avaliação: IoU + Dice Coefficient — from scratch

### Escolher de 3 a 5 imagens;

- Imagens selecionadas:
  - 0619
  - 1105
  - 1113
  - 1141
  - 1286

> Observação: O único código pronto que pode ser utilizado é o de Domínio da frequência, o restante eu preciso implementar a lógica manualmente

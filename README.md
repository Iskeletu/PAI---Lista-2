# Processamento e Análise de Imagens - Lista 2  
Este repositório contém a implementação técnica das **questões 1 e 2** da Lista 2 de *Processamento e Análise de Imagens*.  
O objetivo é **comparar dois algoritmos de segmentação**: **Limiarização Global (Otsu)** e **K-Means**, aplicados a diferentes tipos de imagens.  

---

## 🧩 Estrutura do Projeto
```python
PAI---Lista-2/
├── img/                        # Imagens de entrada
├── output/                        # Diretório de saída de resultados
│ ├── kmeans/                   # Máscaras obtidas por K-means
│ ├── otsu/                     # Máscaras obtidas por Otsu
│ ├── overlay/                  # Imagens originais com sobreposição das máscaras
│ └── metrics.csv               # Tabela de métricas (tempo e proporção de foreground)
├── src/                        # Código-fonte
│ ├── global_thresholding.py    # Implementação do algorítmo Otsu.
│ ├── k_means.py                # Implementação do algorítmo K-Means.
│ └── script.py                 # Script primário do projeto.
├── config.ini                  # Arquivo de confiugração depreciado.
├── LICENSE                     # Licença para utilização do projeto, uso livre com base na licença MIT.
├── README.md                   # Este arquivo.
└── requirements.txt            # Dependências do projeto
```
---

## ⚙️ Requisitos
- Python >= **3.13.7**
- Bibliotecas principais:
  ```bash
  pip install -r requirements.txt
  ```

### Principais dependências:
- numpy
- scikit-image
---

## 🚀 Execução
1. **Coloque as imagens de teste na pasta img/.**
2. **Execute o script principal:**

```bash
python src/main.py
````
3. **Os resultados serão gerados automaticamente em `./output/`, incluindo:**
- Máscaras segmentadas (`./output/otsu/`, `./output/kmeans/`)
- Imagens com sobreposição de máscara (`./output/overlay/`)
- Métricas numéricas (`./output/metrics.csv`)
---

## 📊 Resultados
Cada imagem é segmentada pelos dois métodos:
- **Otsu (global thresholding):** abordagem simples e rápida baseada no histograma global.
- **K-means:** segmentação baseada em agrupamento de pixels, com opção de termo espacial.

Os resultados são salvos para comparação visual e numérica (tempo e proporção de foreground).

## ✉️ Autor
**Fábio Gandini**  
**Disciplina:** *Processamento e Análise de Imagens – 2025*  
**Instituição:** *Pontifícia Universidade Católica de Minas Gerais*
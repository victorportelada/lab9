# Laboratório 09 — Arquitetura RAG Avançada (HNSW, HyDE e Cross-Encoders)

## Pipeline

1. **Indexação HNSW (FAISS)**: 20 fragmentos de manuais médicos → `all-MiniLM-L6-v2` → `IndexHNSWFlat`
2. **HyDE (Query Transformation)**: Dicionário coloquial→técnico + template de prontuário → documento hipotético → vetor-âncora
3. **Bi-Encoder Retrieve**: Busca top-10 por similaridade de cosseno no grafo HNSW
4. **Cross-Encoder Re-rank**: `ms-marco-MiniLM-L-6-v2` reavalia top-10 → top-3 finais

## Hiperparâmetros HNSW (M e ef_construction) vs. KNN Exato

| Aspecto | KNN Exato (IndexFlatIP) | HNSW (IndexHNSWFlat) |
|---------|------------------------|----------------------|
| **Memória RAM** | O(n × d) para os vetores | O(n × d + n × M) — grafo adicional com M arestas por nó |
| **Construção** | O(1) — sem índice | O(n × log(n) × M × ef_construction) |
| **Busca** | O(n × d) — varredura completa | O(log(n) × M × ef_search) — sublinear |
| **Precisão** | 100% (exata) | ~99% (aproximada, ajustável) |

- **M** (default=16, usado=32): Número de vizinhos por nó no grafo. Cada nó armazena M arestas de saída. Aumentar M melhora a precisão e velocidade de busca, mas cada aresta consome 4 bytes por dimensão extra. Para 20 documentos de 384d com M=32: overhead ≈ 20 × 32 × 4 × 384 ≈ 983 KB extra vs. KNN.
- **ef_construction** (default=40, usado=200): Largura do beam search durante a construção do grafo. Valores maiores produzem um grafo de melhor qualidade (rotas mais curtas), mas aumentam o tempo de build. Não afeta RAM.

## Como executar

```bash
pip install -r requirements.txt
python main.py
```

## Atribuição de IA
Este projeto utilizou ferramentas de inteligência artificial (OpenCode/Claude) para brainstorming da arquitetura do pipeline RAG, refinamento dos hiperparâmetros HNSW e geração assistida do código-fonte.

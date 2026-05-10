#!/usr/bin/env python3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


# ===== PASSO 1 =====
documentos_medicos = [
    "Cefaleia pulsátil unilateral associada a náuseas e fotofobia caracteriza enxaqueca clássica",
    "AVC isquêmico: perda súbita de função neurológica por oclusão vascular; janela terapêutica de 4h para trombólise",
    "Crise epiléptica tônico-clônica generalizada com duração maior que 5 minutos define estado de mal epiléptico",
    "Infarto agudo do miocárdio com supradesnivelamento de ST requer angioplastia primária em até 90 minutos",
    "Fibrilação atrial: ritmo irregularmente irregular; CHA2DS2-VASc para estratificação de risco tromboembólico",
    "Hipertensão arterial sistêmica: pressão alvo < 130/80 mmHg em pacientes de alto risco cardiovascular",
    "Asma: obstrução reversível das vias aéreas; corticoide inalatório é base do tratamento de manutenção",
    "DPOC exacerbação: aumento da dispneia, volume de escarro e purificação; antibiótico + corticoide sistêmico",
    "Pneumonia adquirida na comunidade: CURB-65 guia decisão de internação; antibioticoterapia empírica com beta-lactâmico",
    "Tromboembolia pulmonar: disfunção ventricular direita ao ecocardiograma indica trombólise em pacientes instáveis",
    "Diabetes mellitus tipo 2: hemoglobina glicada alvo < 7%; metformina é primeira linha terapêutica",
    "Insuficiência cardíaca com fração de ejeção reduzida: betabloqueador + IECA + espironolactona reduzem mortalidade",
    "Doença renal crônica: taxa de filtração glomerular < 60 mL/min/1,73m² por mais de 3 meses define diagnóstico",
    "Hepatite B crônica: DNA viral > 2000 UI/mL e ALT elevada indicam necessidade de tratamento antiviral",
    "Meningite bacteriana: análise do LCR com pleocitose neutrofílica, proteínas elevadas e glicose baixa",
    "Sepse: qSOFA ≥ 2 pontos (TAS < 100, FR ≥ 22, rebaixamento mental) sugere mau prognóstico",
    "Síndrome coronariana aguda sem supradesnivelamento de ST: troponina de alta sensibilidade para diagnóstico",
    "Acidente vascular cerebral hemorrágico: hematoma intraparenquimatoso espontâneo associado a HAS crônica",
    "Crise hipertensiva: PAS > 180 ou PAD > 120 mmHg com lesão de órgão alvo requer redução gradual da PA",
    "Edema agudo de pulmão cardiogênico: congestão pulmonar com crepitações bilaterais e B3 à ausculta cardíaca",
]


# ===== PASSO 2: HyDE com LLM =====
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("LLM_API_BASE", "https://api.xiaomimimo.com/v1"),
    api_key=os.getenv("LLM_API_KEY"),
)


def gerar_documento_hipotetico(query_coloquial: str) -> str:
    prompt = (
        "Você é um médico especialista. Um paciente leigo descreveu os seguintes sintomas "
        "usando linguagem coloquial. Gere um pequeno parágrafo de prontuário médico técnico, "
        "no mesmo estilo e jargão de um manual de neurologia, que poderia corresponder a esses sintomas. "
        "NÃO responda à pergunta do paciente — apenas alucine um trecho de manual que contenha "
        "termos técnicos associados.\n\n"
        f"Sintomas relatados pelo paciente: {query_coloquial}\n\n"
        "Trecho de manual médico:"
    )
    resposta = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "mimo-v2-flash"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7,
    )
    return resposta.choices[0].message.content.strip()


query_teste = "dor de cabeça latejante e luz incomodando"


# ===== PASSO 3 =====
modelo = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = modelo.encode(documentos_medicos)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


# ===== PASSO 4 =====
index = faiss.IndexHNSWFlat(384, 32)
index.hnsw.efConstruction = 200
index.add(embeddings)


# ===== PASSO 5: Busca HNSW via HyDE =====
documento_hipotetico = gerar_documento_hipotetico(query_teste)
vetor_hyde = modelo.encode([documento_hipotetico])
vetor_hyde = vetor_hyde / np.linalg.norm(vetor_hyde)
distancias, indices = index.search(vetor_hyde, 10)


# ===== PASSO 6: Re-ranking com Cross-Encoder =====
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query_original = query_teste
pares = [(query_original, documentos_medicos[idx]) for idx in indices[0]]
scores_ce = cross_encoder.predict(pares)

indices_ordenados_ce = np.argsort(scores_ce)[::-1]


# ===== PASSO 7 =====
if __name__ == "__main__":
    print(f"Total de documentos indexados: {index.ntotal}")
    print(f"Dimensionalidade: {index.d}")
    print(f"\nDocumento hipotético gerado:\n{documento_hipotetico}")
    print("\n--- TOP-10 (Busca Bi-Encoder via HyDE) ---")
    for i, (score, idx) in enumerate(zip(distancias[0], indices[0])):
        print(f"#{i + 1} [score={score:.4f}] {documentos_medicos[idx]}")

    print("\n--- TOP-3 APÓS CROSS-ENCODER ---")
    for rank, pos in enumerate(indices_ordenados_ce[:3]):
        idx = indices[0][pos]
        print(f"#{rank + 1} [score={scores_ce[pos]:.4f}] {documentos_medicos[idx]}")

    print("\n--- COMPARAÇÃO BI-ENCODER vs CROSS-ENCODER ---")
    print(f"{'#':>3} {'Pos BE':>6} {'Score BE':>8} {'Pos CE':>6} {'Score CE':>8}")
    for i in range(10):
        novo_rank = int(np.where(indices_ordenados_ce == i)[0][0]) + 1
        print(
            f"{i + 1:>3} {i + 1:>6} {distancias[0][i]:>8.4f} {novo_rank:>6} {scores_ce[i]:>8.4f}"
        )

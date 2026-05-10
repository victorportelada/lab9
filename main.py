import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


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


# ===== PASSO 2: HyDE Determinístico =====
DICIONARIO_COLOQUIAL = {
    "dor de cabeça": "cefaleia",
    "luz incomodando": "fotofobia",
    "vista embaçada": "turvação visual",
    "tontura": "vertigem",
    "falta de ar": "dispneia",
    "cansaço": "fadiga",
    "aperto no peito": "angina pectoris",
    "coração acelerado": "taquicardia",
    "tosse com catarro": "expectoração purulenta",
    "chiado no peito": "sibilância",
    "desmaio": "síncope",
    "formigamento": "parestesia",
    "boca torta": "paralisia facial",
    "convulsão": "crise epiléptica",
    "dor nas juntas": "artralgia",
    "inchaço": "edema",
    "febre alta": "hipertermia",
    "suor frio": "diaforese",
}


def gerar_documento_hipotetico(query_coloquial: str) -> str:
    query_modificada = query_coloquial
    termos_encontrados = []
    for coloquial, tecnico in DICIONARIO_COLOQUIAL.items():
        if coloquial in query_modificada:
            query_modificada = query_modificada.replace(coloquial, tecnico)
            termos_encontrados.append(tecnico)
    if not termos_encontrados:
        return query_coloquial
    termos_str = ", ".join(termos_encontrados)
    return (
        f"Paciente relata quadro clínico compatível com {termos_str}. "
        "Ao exame neurológico, observa-se sinais focais ausentes. "
        "A hipótese diagnóstica principal envolve enxaqueca clássica, "
        "devendo-se considerar diagnósticos diferenciais como "
        "cefaleia tensional e cefaleia em salvas."
    )


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


# ===== PASSO 6 =====
if __name__ == "__main__":
    print(f"Total de documentos indexados: {index.ntotal}")
    print(f"Dimensionalidade: {index.d}")
    print(f"\nDocumento hipotético gerado:\n{documento_hipotetico}")
    print("\n--- TOP-10 (Busca Bi-Encoder via HyDE) ---")
    for i, (score, idx) in enumerate(zip(distancias[0], indices[0])):
        print(f"#{i + 1} [score={score:.4f}] {documentos_medicos[idx]}")

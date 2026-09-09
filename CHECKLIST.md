# RAG against the machine — checklist

Legenda: `[ ]` a fazer · `[~]` em andamento · `[x]` feito
Enunciado: `docs/en.subject.pdf` (v2.0) · Fatos fixados: `CLAUDE.md` ·
Guia: `docs/GUIA.md`

## Fase 0 — Setup e regras do jogo
- [x] Corpus, datasets e corretor no lugar (`data/`, `./moulinette`)
- [x] Descobrir na prática como o corretor pontua (IoU >= 0.05, limite de 2000 chars)
- [x] Medir o teto de recall do chunking fixo (>= 0.96 em todos os tamanhos)
- [x] Ler o enunciado e fixar os requisitos duros no `CLAUDE.md`
- [ ] `pyproject.toml` + `.flake8` + `Makefile` (install/run/debug/clean/lint) + `.gitignore` + `README.md`
- [ ] `uv sync` gerando o `.venv/` e o `uv.lock`
- [ ] Primeiro commit
- [ ] **Capítulo 01 — a métrica**: ler e responder as 6 perguntas

## Fase 1 — Fundações
- [ ] Capítulo 02 — o que é RAG e o que este RAG tem que fazer
- [ ] Capítulo 03 — contratos de dados com pydantic
- [ ] `src/models.py` — os 8 modelos exigidos, com esses nomes exatos
- [ ] Capítulo 04 — o CLI com Python Fire
- [ ] `src/__main__.py` — os 6 comandos, ainda vazios, com todos os caminhos configuráveis
- [ ] Entrada degenerada tratada: query vazia, `k=0`, arquivo faltando, JSON inválido

## Fase 2 — Indexing
- [ ] Capítulo 05 — o corpus
- [ ] `src/corpus.py` — percorrer `data/raw/`, ler o texto, decidir o que entra
- [ ] Capítulo 06 — chunking I: a ideia
- [ ] Capítulo 07 — chunking II: as duas estratégias exigidas
- [ ] `src/chunking.py` — chunker de Python (`ast`) e chunker de Markdown/texto
- [ ] Capítulo 08 — tokenização
- [ ] `src/tokenizer.py`
- [ ] Capítulo 09 — o índice invertido
- [ ] `src/indexer.py` — construir e persistir em `data/processed/`
- [ ] Comando `index --max_chunk_size` rodando em **menos de 5 minutos**

## Fase 3 — Retrieval e chegar nos thresholds
- [ ] Capítulo 10 — TF-IDF
- [ ] Capítulo 11 — BM25
- [ ] `src/retriever.py` — top-k para uma query
- [ ] `search_dataset` escrevendo em `data/output/search_results/<Scope>/`
- [ ] Baseline medido nos dois datasets
- [ ] Capítulo 12 — avaliar e melhorar
- [ ] Iterar tokenizer, chunking e parâmetros; anotar o efeito no recall (vai pro README)
- [ ] **Recall@5 >= 80% (docs)** e **>= 50% (code)**
- [ ] **<= 90 s para 200 perguntas**

## Fase 4 — Augmenting e Generation
- [ ] Capítulo 13 — escolher o contexto
- [ ] Capítulo 14 — o modelo e o prompt
- [ ] Capítulo 15 — gerar e escrever a saída
- [ ] `src/generator.py` + comandos `answer` e `answer_dataset`

## Fase 5 — Qualidade e entrega
- [ ] `src/evaluation.py` + comando `evaluate` (o corretor nunca é importado)
- [ ] Capítulo 16 — performance: medir e otimizar
- [ ] Capítulo 17 — testes `pytest` das peças isoladas
- [ ] `make lint` limpo (flake8 + mypy com as flags do enunciado)
- [ ] Capítulo 18 — README em inglês com todas as seções exigidas
- [ ] Preparar a *recode* da defesa

## Fase 6 — Bônus (só contam com a parte obrigatória toda validada)
- [ ] Capítulo 19 — embeddings semânticos (`all-MiniLM-L6-v2`)
- [ ] Capítulo 20 — retrieval híbrido
- [ ] Capítulo 21 — indexação incremental · caching · API HTTP local

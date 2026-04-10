import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make sibling src modules importable when run from the project root
sys.path.insert(0, str(Path(__file__).parent))

# Load ANTHROPIC_API_KEY (and any other vars) from .env before importing clients
load_dotenv()

from rag_pipeline import ask_claude, retrieve_chunks

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama, OllamaEmbeddings

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
OLLAMA_MODEL = "qwen2.5:0.5b"

# ---------------------------------------------------------------------------
# Test dataset — 5 question / ground-truth pairs about Apollo Hospital policies
# ---------------------------------------------------------------------------
TEST_DATASET = [
    {
        "question": "What is the ICU discharge procedure at Apollo Hospitals?",
        "ground_truth": (
            "The ICU discharge procedure at Apollo Hospitals involves a clinical "
            "assessment by the treating physician to confirm the patient is stable, "
            "followed by a transfer to a general ward or direct discharge home with "
            "written care instructions and a follow-up appointment."
        ),
    },
    {
        "question": "What documents are required for patient admission at Apollo Hospitals?",
        "ground_truth": (
            "Patients admitted to Apollo Hospitals are required to present a valid "
            "government-issued photo ID, health insurance card or pre-authorisation "
            "letter, previous medical records, and a physician referral letter where applicable."
        ),
    },
    {
        "question": "What is the visiting hours policy at Apollo Hospitals?",
        "ground_truth": (
            "Apollo Hospitals permits visitors during designated morning and evening "
            "windows. ICU and critical care units have restricted visiting hours and "
            "generally allow only immediate family members to enter."
        ),
    },
    {
        "question": "How does Apollo Hospitals handle patient data confidentiality?",
        "ground_truth": (
            "Apollo Hospitals protects patient data in accordance with applicable "
            "healthcare regulations. Medical records and personal information are "
            "accessible only to authorised clinical and administrative staff, and "
            "are never shared with third parties without patient consent."
        ),
    },
    {
        "question": "What is the billing and payment policy at Apollo Hospitals?",
        "ground_truth": (
            "Apollo Hospitals accepts cash, credit and debit cards, and direct "
            "insurance billing. An admission deposit is collected at registration, "
            "and an itemised bill is provided at the time of discharge."
        ),
    },
]


def build_ragas_dataset() -> Dataset:
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_DATASET:
        question = item["question"]
        print(f"\n  Q: {question}")

        chunks = retrieve_chunks(question)
        answer = ask_claude(question, chunks)

        questions.append(question)
        answers.append(answer)
        contexts.append([c["text"] for c in chunks])
        ground_truths.append(item["ground_truth"])

        print(f"     → {len(chunks)} chunks retrieved, answer generated.")

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def main() -> None:
    print("=" * 60)
    print("Apollo RAG — RAGAS Evaluation")
    print("=" * 60)

    print("\n[1/3] Running all questions through the RAG pipeline...")
    dataset = build_ragas_dataset()

    print("\n[2/3] Setting up evaluation LLMs (Claude Haiku for faithfulness/relevancy, Ollama for precision/recall)...")
    claude_llm = LangchainLLMWrapper(ChatAnthropic(model=CLAUDE_MODEL, temperature=0))
    ollama_llm = LangchainLLMWrapper(ChatOllama(model=OLLAMA_MODEL, temperature=0))
    ollama_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=OLLAMA_MODEL))

    metrics = [
        Faithfulness(llm=claude_llm),
        AnswerRelevancy(llm=claude_llm, embeddings=ollama_embeddings),
        ContextPrecision(llm=ollama_llm),
        ContextRecall(llm=ollama_llm),
    ]

    print("\n[3/3] Running RAGAS evaluation (this may take a few minutes)...\n")
    result = evaluate(dataset=dataset, metrics=metrics)

    scores = result.to_pandas()

    print("\n" + "=" * 60)
    print("Evaluation Results  (0 = worst, 1 = best)")
    print("=" * 60)
    print(f"  faithfulness        {scores['faithfulness'].mean():.4f}")
    print(f"  answer_relevancy    {scores['answer_relevancy'].mean():.4f}")
    print(f"  context_precision   {scores['context_precision'].mean():.4f}")
    print(f"  context_recall      {scores['context_recall'].mean():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

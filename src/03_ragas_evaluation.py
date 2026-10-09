"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import argparse
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
# ⚠️ Cả 2 phải chứa {context}, ví dụ kết thúc bằng "...\n\nContext:\n{context}"
#    Thiếu {context} → LLM không thấy tài liệu, không báo lỗi, faithfulness/context_* rất thấp.
SYSTEM_V1 = (
    "Bạn là trợ lý AI hữu ích. Chỉ trả lời bằng thông tin có trong context. "
    "Trả lời cùng ngôn ngữ với câu hỏi, trực tiếp và ngắn gọn trong 2-4 câu. "
    "Nếu context không đủ, hãy nói "
    "rằng bạn không có đủ thông tin; không suy đoán.\n\nContext:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia AI cẩn trọng. Hãy đọc kỹ context, chọn các dữ kiện liên quan "
    "và trình bày câu trả lời cùng ngôn ngữ với câu hỏi, rõ ràng, có cấu trúc trong "
    "3-5 câu. Chỉ đưa ra kết luận "
    "được context hỗ trợ; nếu thiếu dữ kiện, nêu rõ giới hạn thay vì suy đoán.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    contexts = [doc.page_content for doc in docs]

    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    cache_path = (
        Path(__file__).parent.parent / "data" / f"ragas_{prompt_version}_outputs.json"
    )
    results = []
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and len(cached) <= len(QA_PAIRS):
                results = cached
                print(f"♻️  Tiếp tục từ cache: {len(results)}/50 câu đã có")
        except (json.JSONDecodeError, OSError):
            print("⚠️  Cache không hợp lệ; chạy lại từ đầu.")

    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS[len(results):], len(results) + 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        cache_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RunConfig(
            timeout=180,
            max_retries=10,
            max_wait=120,
            # OpenRouter giới hạn chi phí của các request đang chạy đồng thời.
            # Chạy tuần tự chậm hơn nhưng tránh lỗi 402 in_flight_budget_exhausted.
            max_workers=1,
        ),
        batch_size=1,
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    incomplete_metrics = []
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = np.asarray(result[key], dtype=float)
        valid = raw[np.isfinite(raw)]
        if valid.size == 0:
            raise RuntimeError(f"Không có điểm RAGAS hợp lệ cho metric '{key}'")
        if valid.size != len(rag_results):
            incomplete_metrics.append(f"{key}: {valid.size}/{len(rag_results)}")
        scores[key] = float(np.mean(valid))

    if incomplete_metrics:
        details = ", ".join(incomplete_metrics)
        raise RuntimeError(
            "RAGAS không hoàn tất đủ mẫu; không ghi đè báo cáo chính. "
            f"Số điểm hợp lệ: {details}"
        )

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Chạy đánh giá RAGAS cho hai prompt.")
    parser.add_argument(
        "--only",
        choices=("v1", "v2"),
        help="Chỉ đánh giá lại một version và giữ điểm version còn lại từ report hiện có.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    scores_by_version = {}

    if args.only:
        if not report_path.exists():
            parser.error("--only cần data/ragas_report.json từ một lần chạy đầy đủ trước đó")
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        other_version = "v2" if args.only == "v1" else "v1"
        other_key = f"prompt_{other_version}_scores"
        if other_key not in existing_report:
            parser.error(f"Report hiện có thiếu '{other_key}'")
        scores_by_version[other_version] = existing_report[other_key]

    vectorstore = setup_vectorstore()
    versions = [args.only] if args.only else ["v1", "v2"]
    for version in versions:
        rag_results = collect_rag_outputs(vectorstore, version)
        scores_by_version[version] = run_ragas_eval(rag_results, version)

    v1_scores = scores_by_version["v1"]
    v2_scores = scores_by_version["v2"]

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "evaluation_complete": True,
        "samples_per_prompt": len(QA_PAIRS),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()

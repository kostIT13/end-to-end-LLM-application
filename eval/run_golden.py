import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from src.services.document_chunks.document_chunks_service import DocumentChunksService
from src.core.database.db import get_db
from eval.metrics import recall_top_k, mrr, evaluate_with_ci


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    return dataset


async def main():
    dataset_path = Path("eval/golden_dataset.jsonl")
    top_k = 5
    threshold = 0.7
    use_hybrid = True
    
    if len(sys.argv) > 1:
        threshold = float(sys.argv[1])
    if len(sys.argv) > 2:
        top_k = int(sys.argv[2])
    
    if not dataset_path.exists():
        logger.error("Dataset not found: {}", dataset_path)
        sys.exit(1)
    
    dataset = load_dataset(dataset_path)
    logger.info("Loaded {} queries", len(dataset))
    
    per_query_recalls = []
    per_query_mrrs = []
    
    async for session in get_db():
        service = DocumentChunksService(session)
        
        for item in dataset:
            query = item["query"]
            ground_truth = [{"doc_id": str(d)} for d in item["relevant_doc_ids"]]
            
            chunks = await service.search(
                query=query,
                top_k=top_k,
                use_hybrid=use_hybrid,
            )
            
            per_query_recalls.append(recall_top_k(chunks, ground_truth, top_k))
            per_query_mrrs.append(mrr(chunks, ground_truth))
    
    avg_recall = sum(per_query_recalls) / len(per_query_recalls)
    avg_mrr = sum(per_query_mrrs) / len(per_query_mrrs)
    
    avg_recall, lower_ci, upper_ci, passed = evaluate_with_ci(
        per_query_recalls, threshold=threshold
    )
    
    logger.info("=" * 50)
    logger.info("Recall@{}: {:.4f} ({:.2f}%)", top_k, avg_recall, avg_recall * 100)
    logger.info("95% CI:     [{:.4f}, {:.4f}]", lower_ci, upper_ci)
    logger.info("MRR:        {:.4f} ({:.2f}%)", avg_mrr, avg_mrr * 100)
    logger.info("Threshold:  {}", threshold)
    logger.info("=" * 50)
    
    if passed:
        logger.info("PASSED")
        sys.exit(0)
    else:
        logger.error("FAILED: lower CI < threshold")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
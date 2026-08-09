"""Retrieval + answering harness.

Runs each gold question as its gold principal, so ACL correctness is measured,
not assumed. Three question classes, scored separately:

  answerable    gold passage ids exist; score recall@k, MRR, nDCG, citation validity
  unanswerable  ~15% of the set; the only correct behaviour is abstention
  restricted    answerable for user A, invisible to user B; any leak is a hard
                failure that fails the whole run regardless of other metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field

from isc.eval.metrics import mrr, ndcg_at_k, recall_at_k


@dataclass
class QuestionOutcome:
    question_id: str
    question_class: str
    retrieved_ids: list[str] = field(default_factory=list)
    gold_ids: set[str] = field(default_factory=set)
    abstained: bool = False
    citations_valid: bool = True
    leaked_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RetrievalReport:
    outcomes: list[QuestionOutcome] = field(default_factory=list)

    def recall_at(self, k: int) -> float:
        rows = [o for o in self.outcomes if o.question_class == "answerable"]
        return sum(recall_at_k(o.retrieved_ids, o.gold_ids, k) for o in rows) / max(len(rows), 1)

    def mean_mrr(self) -> float:
        rows = [o for o in self.outcomes if o.question_class == "answerable"]
        return sum(mrr(o.retrieved_ids, o.gold_ids) for o in rows) / max(len(rows), 1)

    def ndcg(self, k: int = 8) -> float:
        rows = [o for o in self.outcomes if o.question_class == "answerable"]
        return sum(ndcg_at_k(o.retrieved_ids, o.gold_ids, k) for o in rows) / max(len(rows), 1)

    def abstention_precision(self) -> float:
        """Of the questions we abstained on, how many were genuinely unanswerable?"""
        abstained = [o for o in self.outcomes if o.abstained]
        if not abstained:
            return 0.0
        return sum(o.question_class == "unanswerable" for o in abstained) / len(abstained)

    def abstention_recall(self) -> float:
        """Of the genuinely unanswerable questions, how many did we abstain on?"""
        unanswerable = [o for o in self.outcomes if o.question_class == "unanswerable"]
        if not unanswerable:
            return 1.0
        return sum(o.abstained for o in unanswerable) / len(unanswerable)

    def leaks(self) -> list[QuestionOutcome]:
        return [o for o in self.outcomes if o.leaked_chunk_ids]

    def passed(self) -> bool:
        """A single leak fails the run. This is not a tunable metric."""
        return not self.leaks()

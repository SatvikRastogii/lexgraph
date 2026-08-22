"""Shared result type for every retriever."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Hit:
    """One retrieved chunk, with enough provenance to cite it in an answer."""

    chunk_id: str
    doc_id: str
    text: str
    score: float = 0.0
    title: str = ""
    year: str = "unknown"
    para_label: str = ""
    # How each stage scored this hit; useful for debugging a fused ranking.
    components: dict[str, float] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """``Sunil Batra vs Delhi Administration (1979), para 12-15``."""
        label = self.title or self.doc_id
        if self.year and self.year != "unknown":
            label += f" ({self.year})"
        if self.para_label:
            label += f", {self.para_label}"
        return label

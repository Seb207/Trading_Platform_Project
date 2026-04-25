import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from Research_LLM.arxiv_client import ArxivToolClient


SAMPLE_MD = """\
# Momentum Factor in Equity Markets

## Abstract

This paper investigates the momentum factor across equity markets.
We find strong evidence for cross-sectional momentum persistence.

## Introduction

Momentum strategies have been documented in the literature since Jegadeesh (1993).
Prior work shows 12-1 month returns predict future performance.

## Methodology

We use a long-short portfolio ranked on trailing 12-month returns.
The rebalancing frequency is monthly with transaction cost adjustments.

## Results

The strategy yields a Sharpe ratio of 1.2 over the sample period.
Maximum drawdown is limited to 15% using volatility scaling.

## Conclusion

Momentum remains a robust and exploitable factor in modern markets.
"""

SAMPLE_METADATA_ENTRY = {
    "arxiv_id": "1234.5678",
    "title": "Momentum Factor in Equity Markets",
    "authors": ["Author A", "Author B"],
    "published": "2024-01-15T00:00:00Z",
    "summary": "This paper investigates the momentum factor across equity markets.",
    "category": "q-fin.CP",
    "format": "md",
    "relative_path": "q-fin.CP/1234.5678.md",
    "indexed_at": "2024-01-20T10:00:00",
}


class TestLocalPaperTools(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.download_dir = Path(self.temp_dir.name) / "papers" / "arXiv"
        (self.download_dir / "q-fin.CP").mkdir(parents=True, exist_ok=True)
        (self.download_dir / "cs.AI").mkdir(parents=True, exist_ok=True)

        (self.download_dir / "q-fin.CP" / "1234.5678.md").write_text(SAMPLE_MD, encoding="utf-8")
        (self.download_dir / "cs.AI" / "9999.0001.md").write_text(
            "# AI Paper\n\n## Abstract\n\nTransformer summary content.", encoding="utf-8"
        )
        (self.download_dir / "q-fin.CP" / "1111.2222.pdf").write_bytes(b"%PDF-1.4")

        self.client = ArxivToolClient(download_dir=str(self.download_dir))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # ==================================================================
    # list_local_papers
    # ==================================================================

    def test_list_local_papers_returns_records(self) -> None:
        result = self.client.list_local_papers(limit=10)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 3)
        self.assertTrue(any(p["relative_path"].endswith("1234.5678.md") for p in result["papers"]))

    def test_list_local_papers_category_filter(self) -> None:
        result = self.client.list_local_papers(category="q-fin.CP", limit=10)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["count"] >= 1)
        self.assertTrue(all(p["category"] == "q-fin.CP" for p in result["papers"]))

    # ==================================================================
    # read_local_paper
    # ==================================================================

    def test_read_local_paper_markdown_success(self) -> None:
        result = self.client.read_local_paper("q-fin.CP/1234.5678.md", max_chars=100)
        self.assertEqual(result["status"], "success")
        self.assertIn("Momentum", result["content"])
        self.assertGreater(result["returned_chars"], 0)

    def test_read_local_paper_default_max_chars_is_50000(self) -> None:
        import inspect
        sig = inspect.signature(self.client.read_local_paper)
        self.assertEqual(sig.parameters["max_chars"].default, 50000)

    def test_read_local_paper_pdf_not_supported(self) -> None:
        result = self.client.read_local_paper("q-fin.CP/1111.2222.pdf")
        self.assertEqual(result["status"], "error")
        self.assertIn("PDF text extraction is not supported", result["message"])

    def test_read_local_paper_blocks_path_traversal(self) -> None:
        result = self.client.read_local_paper("../../etc/passwd")
        self.assertEqual(result["status"], "error")
        self.assertIn("Path traversal", result["message"])

    # ==================================================================
    # analyze_local_paper
    # ==================================================================

    def test_analyze_local_paper_success(self) -> None:
        result = self.client.analyze_local_paper("q-fin.CP/1234.5678.md")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["arxiv_id"], "1234.5678")
        self.assertEqual(result["category"], "q-fin.CP")
        self.assertGreater(result["total_chars"], 0)
        self.assertIn("full_content", result)
        self.assertIn("Momentum", result["full_content"])

    def test_analyze_local_paper_extracts_sections(self) -> None:
        result = self.client.analyze_local_paper("q-fin.CP/1234.5678.md")
        section_titles = [s["title"] for s in result["sections"]]
        self.assertIn("Abstract", section_titles)
        self.assertIn("Methodology", section_titles)
        self.assertIn("Results", section_titles)

    def test_analyze_local_paper_section_map_has_no_content(self) -> None:
        result = self.client.analyze_local_paper("q-fin.CP/1234.5678.md")
        for entry in result["section_map"]:
            self.assertIn("title", entry)
            self.assertIn("char_count", entry)
            self.assertNotIn("content", entry)

    def test_analyze_local_paper_section_content_correct(self) -> None:
        result = self.client.analyze_local_paper("q-fin.CP/1234.5678.md")
        methodology = next((s for s in result["sections"] if s["title"] == "Methodology"), None)
        self.assertIsNotNone(methodology)
        self.assertIn("long-short portfolio", methodology["content"])

    def test_analyze_local_paper_missing_file(self) -> None:
        result = self.client.analyze_local_paper("q-fin.CP/nonexistent.md")
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["message"].lower())

    def test_analyze_local_paper_pdf_not_supported(self) -> None:
        result = self.client.analyze_local_paper("q-fin.CP/1111.2222.pdf")
        self.assertEqual(result["status"], "error")
        self.assertIn(".md", result["message"])

    def test_analyze_local_paper_blocks_path_traversal(self) -> None:
        result = self.client.analyze_local_paper("../../etc/passwd")
        self.assertEqual(result["status"], "error")
        self.assertIn("Path traversal", result["message"])

    # ==================================================================
    # _normalize_date
    # ==================================================================

    def test_normalize_date_with_dashes(self) -> None:
        self.assertEqual(ArxivToolClient._normalize_date("2024-01-15"), "20240115")

    def test_normalize_date_already_compact(self) -> None:
        self.assertEqual(ArxivToolClient._normalize_date("20240115"), "20240115")

    # ==================================================================
    # Phase 3 — Metadata helpers
    # ==================================================================

    def test_metadata_save_and_load(self) -> None:
        self.client._save_metadata({"2401.99999": SAMPLE_METADATA_ENTRY})
        loaded = self.client._load_metadata()
        self.assertIn("2401.99999", loaded)
        self.assertEqual(loaded["2401.99999"]["title"], SAMPLE_METADATA_ENTRY["title"])

    def test_load_metadata_returns_empty_dict_when_missing(self) -> None:
        result = self.client._load_metadata()
        self.assertIsInstance(result, dict)

    def test_upsert_paper_metadata_creates_entry(self) -> None:
        self.client._upsert_paper_metadata(
            arxiv_id="2401.11111",
            title="Test Paper",
            authors=["Alice"],
            published="2024-01-01T00:00:00Z",
            summary="A test abstract.",
            category="q-fin.ST",
            fmt="md",
            relative_path="q-fin.ST/2401.11111.md",
        )
        metadata = self.client._load_metadata()
        self.assertIn("2401.11111", metadata)
        self.assertEqual(metadata["2401.11111"]["title"], "Test Paper")
        self.assertIn("indexed_at", metadata["2401.11111"])

    def test_upsert_paper_metadata_overwrites_existing(self) -> None:
        self.client._upsert_paper_metadata(
            arxiv_id="2401.11111", title="Old Title", authors=[],
            published="", summary="", category="q-fin.ST", fmt="md", relative_path=""
        )
        self.client._upsert_paper_metadata(
            arxiv_id="2401.11111", title="New Title", authors=[],
            published="", summary="", category="q-fin.ST", fmt="md", relative_path=""
        )
        metadata = self.client._load_metadata()
        self.assertEqual(metadata["2401.11111"]["title"], "New Title")

    # ==================================================================
    # Phase 3 — bulk_download_papers
    # ==================================================================

    def test_bulk_download_papers_calls_download_for_each(self) -> None:
        mock_result = {"status": "success", "format": "markdown", "file_path": "/tmp/x.md",
                       "category": "q-fin.CP", "message": "ok"}
        with patch.object(self.client, "download_paper", return_value=mock_result) as mock_dl:
            results = self.client.bulk_download_papers(
                arxiv_ids=["2401.00001", "2401.00002"], category="q-fin.CP"
            )

        self.assertEqual(mock_dl.call_count, 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["status"] == "success" for r in results))

    def test_bulk_download_papers_returns_arxiv_id_in_each_result(self) -> None:
        mock_result = {"status": "success", "format": "markdown", "file_path": "/tmp/x.md",
                       "category": "q-fin.CP", "message": "ok"}
        with patch.object(self.client, "download_paper", return_value=mock_result):
            results = self.client.bulk_download_papers(["2401.00001"], "q-fin.CP")

        self.assertEqual(results[0]["arxiv_id"], "2401.00001")

    # ==================================================================
    # Phase 3 — backfill_metadata
    # ==================================================================

    def test_backfill_metadata_populates_missing_entries(self) -> None:
        api_papers = [
            {"arxiv_id": "1234.5678", "title": "Momentum Paper", "authors": ["A"],
             "published": "2024-01-01T00:00:00Z", "summary": "Momentum abstract.", "category": "q-fin.CP"},
            {"arxiv_id": "9999.0001", "title": "AI Paper", "authors": ["B"],
             "published": "2024-02-01T00:00:00Z", "summary": "AI abstract.", "category": "cs.AI"},
            {"arxiv_id": "1111.2222", "title": "PDF Paper", "authors": ["C"],
             "published": "2024-03-01T00:00:00Z", "summary": "PDF abstract.", "category": "q-fin.CP"},
        ]
        with patch.object(self.client, "fetch_paper_metadata_batch", return_value=api_papers):
            result = self.client.backfill_metadata()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["backfilled"], 3)
        metadata = self.client._load_metadata()
        self.assertIn("1234.5678", metadata)
        self.assertIn("9999.0001", metadata)
        self.assertEqual(metadata["1234.5678"]["title"], "Momentum Paper")

    def test_backfill_metadata_skips_already_indexed(self) -> None:
        self.client._upsert_paper_metadata(
            arxiv_id="1234.5678", title="Already indexed", authors=[],
            published="", summary="existing", category="q-fin.CP", fmt="md",
            relative_path="q-fin.CP/1234.5678.md"
        )
        api_papers = [
            {"arxiv_id": "9999.0001", "title": "AI Paper", "authors": [],
             "published": "", "summary": "AI abstract.", "category": "cs.AI"},
            {"arxiv_id": "1111.2222", "title": "PDF Paper", "authors": [],
             "published": "", "summary": "", "category": "q-fin.CP"},
        ]
        with patch.object(self.client, "fetch_paper_metadata_batch", return_value=api_papers):
            result = self.client.backfill_metadata()

        self.assertEqual(result["backfilled"], 2)
        metadata = self.client._load_metadata()
        self.assertEqual(metadata["1234.5678"]["title"], "Already indexed")

    def test_backfill_metadata_nothing_to_do(self) -> None:
        for arxiv_id in ["1234.5678", "9999.0001", "1111.2222"]:
            self.client._upsert_paper_metadata(
                arxiv_id=arxiv_id, title="x", authors=[], published="",
                summary="", category="q-fin.CP", fmt="md", relative_path=""
            )
        with patch.object(self.client, "fetch_paper_metadata_batch") as mock_api:
            result = self.client.backfill_metadata()

        mock_api.assert_not_called()
        self.assertEqual(result["backfilled"], 0)
        self.assertIn("All papers already have metadata", result["message"])

    # ==================================================================
    # Phase 4 — build_search_index
    # ==================================================================

    def test_build_search_index_no_metadata_returns_error(self) -> None:
        mock_collection = MagicMock()
        with patch.object(self.client, "_get_chroma_collection", return_value=mock_collection):
            result = self.client.build_search_index()
        self.assertEqual(result["status"], "error")
        self.assertIn("metadata", result["message"].lower())

    def test_build_search_index_success(self) -> None:
        self.client._upsert_paper_metadata(
            arxiv_id="1234.5678", title="Momentum Paper", authors=["A"],
            published="2024-01-01T00:00:00Z",
            summary="Cross-sectional momentum factor in equity markets.",
            category="q-fin.CP", fmt="md", relative_path="q-fin.CP/1234.5678.md"
        )
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1

        with patch.object(self.client, "_get_chroma_collection", return_value=mock_collection):
            result = self.client.build_search_index()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["indexed"], 1)
        mock_collection.upsert.assert_called_once()

    def test_build_search_index_skips_entries_without_summary(self) -> None:
        self.client._upsert_paper_metadata(
            arxiv_id="1234.5678", title="With Summary", authors=[],
            published="", summary="Has abstract content.", category="q-fin.CP",
            fmt="md", relative_path="q-fin.CP/1234.5678.md"
        )
        self.client._upsert_paper_metadata(
            arxiv_id="9999.0001", title="No Summary", authors=[],
            published="", summary="", category="cs.AI",
            fmt="md", relative_path="cs.AI/9999.0001.md"
        )
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1

        with patch.object(self.client, "_get_chroma_collection", return_value=mock_collection):
            result = self.client.build_search_index()

        self.assertEqual(result["indexed"], 1)

    # ==================================================================
    # Phase 4 — search_local_papers_by_topic
    # ==================================================================

    def _make_mock_collection(self, papers: list[dict]) -> MagicMock:
        """Helper: build a mock ChromaDB collection returning given papers."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(papers)
        mock_collection.query.return_value = {
            "ids": [[p["arxiv_id"] for p in papers]],
            "documents": [[p["abstract"] for p in papers]],
            "metadatas": [[{
                "title": p["title"],
                "category": p["category"],
                "published": p.get("published", ""),
                "relative_path": p.get("relative_path", ""),
                "format": p.get("format", "md"),
            } for p in papers]],
            "distances": [[p["distance"] for p in papers]],
        }
        return mock_collection

    def test_search_local_papers_by_topic_success(self) -> None:
        papers = [
            {"arxiv_id": "1234.5678", "title": "Momentum Paper", "category": "q-fin.CP",
             "abstract": "Cross-sectional momentum.", "distance": 0.1,
             "relative_path": "q-fin.CP/1234.5678.md", "format": "md"},
        ]
        mock_collection = self._make_mock_collection(papers)

        with patch.object(self.client, "_get_chroma_collection", return_value=mock_collection):
            result = self.client.search_local_papers_by_topic("momentum equity factor")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["arxiv_id"], "1234.5678")
        self.assertAlmostEqual(result["results"][0]["similarity_score"], 0.9)

    def test_search_local_papers_empty_index_returns_error(self) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0

        with patch.object(self.client, "_get_chroma_collection", return_value=mock_collection):
            result = self.client.search_local_papers_by_topic("momentum")

        self.assertEqual(result["status"], "error")
        self.assertIn("build_search_index", result["message"])

    def test_search_local_papers_empty_query_returns_error(self) -> None:
        result = self.client.search_local_papers_by_topic("")
        self.assertEqual(result["status"], "error")
        self.assertIn("query", result["message"].lower())

    def test_search_local_papers_similarity_score_range(self) -> None:
        papers = [
            {"arxiv_id": "1234.5678", "title": "P1", "category": "q-fin.CP",
             "abstract": "abstract 1", "distance": 0.0,
             "relative_path": "q-fin.CP/1234.5678.md", "format": "md"},
            {"arxiv_id": "9999.0001", "title": "P2", "category": "q-fin.CP",
             "abstract": "abstract 2", "distance": 1.0,
             "relative_path": "q-fin.CP/9999.0001.md", "format": "md"},
        ]
        mock_collection = self._make_mock_collection(papers)

        with patch.object(self.client, "_get_chroma_collection", return_value=mock_collection):
            result = self.client.search_local_papers_by_topic("test query", n_results=2)

        scores = [r["similarity_score"] for r in result["results"]]
        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[1], 0.0)


if __name__ == "__main__":
    unittest.main()

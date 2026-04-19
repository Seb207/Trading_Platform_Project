import tempfile
import unittest
from pathlib import Path

from Research_LLM.arxiv_client import ArxivToolClient


class TestLocalPaperTools(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.download_dir = Path(self.temp_dir.name) / "papers" / "arXiv"
        (self.download_dir / "q-fin.CP").mkdir(parents=True, exist_ok=True)
        (self.download_dir / "cs.AI").mkdir(parents=True, exist_ok=True)

        (self.download_dir / "q-fin.CP" / "1234.5678.md").write_text(
            "# Title\n\nThis is a sample quant paper.", encoding="utf-8"
        )
        (self.download_dir / "cs.AI" / "9999.0001.md").write_text(
            "# AI Paper\n\nTransformer summary content.", encoding="utf-8"
        )
        (self.download_dir / "q-fin.CP" / "1111.2222.pdf").write_bytes(b"%PDF-1.4")

        self.client = ArxivToolClient(download_dir=str(self.download_dir))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

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

    def test_read_local_paper_markdown_success(self) -> None:
        result = self.client.read_local_paper("q-fin.CP/1234.5678.md", max_chars=100)

        self.assertEqual(result["status"], "success")
        self.assertIn("sample quant paper", result["content"])
        self.assertGreater(result["returned_chars"], 0)

    def test_read_local_paper_pdf_not_supported(self) -> None:
        result = self.client.read_local_paper("q-fin.CP/1111.2222.pdf")

        self.assertEqual(result["status"], "error")
        self.assertIn("PDF text extraction is not supported", result["message"])

    def test_read_local_paper_blocks_path_traversal(self) -> None:
        result = self.client.read_local_paper("../../etc/passwd")

        self.assertEqual(result["status"], "error")
        self.assertIn("Path traversal", result["message"])


if __name__ == "__main__":
    unittest.main()



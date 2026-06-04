// ── Paper ──────────────────────────────────────────────────────────────
export interface Paper {
  arxiv_id: string;
  title: string;
  category: string;
  published: string;        // ISO date string
  abstract: string;
  similarity_score?: number; // 0–1 from ChromaDB
  relative_path?: string;
  format?: "md" | "pdf";
}

export interface PaperSection {
  arxiv_id: string;
  paper_title: string;
  category: string;
  section_name: string;
  section_index: number;
  char_count: number;
  relative_path: string;
  preview: string;
  similarity_score: number;
}

// ── Search ─────────────────────────────────────────────────────────────
export type SearchMode = "abstract" | "section" | "arxiv";

export interface SearchFilters {
  query: string;
  mode: SearchMode;
  category: string; // "" = all
}

// ── LLM ───────────────────────────────────────────────────────────────
export type LLMProvider = "claude" | "ollama";

export interface LLMConfig {
  provider: LLMProvider;
  model: string;
  apiKey?: string;         // Claude only
  ollamaUrl?: string;      // Ollama only, default localhost:11434
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  paperRefs?: string[];    // arxiv_ids cited
  modelName?: string;      // e.g. "claude-opus-4-5" or "gemma4:latest"
}

// ── API responses ──────────────────────────────────────────────────────
export interface PaperListResponse {
  status: "success" | "error";
  papers: Paper[];
  count: number;
}

export interface SearchResponse {
  status: "success" | "error";
  query: string;
  results: Paper[];
  count: number;
}

export interface StatusResponse {
  papers_count: number;
  sections_count: number;
  metadata_count: number;
  abstract_index_built: boolean;
  section_index_built: boolean;
}

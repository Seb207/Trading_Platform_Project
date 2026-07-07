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
export type LLMProvider = "claude" | "ollama" | "openrouter";

export interface LLMConfig {
  provider: LLMProvider;
  model: string;
  apiKey?: string;            // Claude only
  ollamaUrl?: string;         // Ollama only, default localhost:11434
  openRouterApiKey?: string;  // OpenRouter only
}

export type CriticVerification = "verifying" | "verified" | "revised";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  paperRefs?: string[];    // arxiv_ids cited
  modelName?: string;      // e.g. "claude-opus-4-5" or "gemma4:latest"
  verification?: CriticVerification;  // critic status of this draft (assistant messages only)
  revisionOf?: string;     // set on a critic-driven replacement message: id of the draft it revises
  revisionIssues?: string[]; // issues the critic flagged (only present on revision messages)
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

// ── Market Regime ────────────────────────────────────────────────────
// Context-retrieval tool: find_analogs() is the core, always-on result;
// validation (KS test / sensitivity sweep) is opt-in, never auto-called.

export interface RegimeFactor {
  key: string;
  name: string;
  theme: string;
  start: string | null;
  end: string | null;
}

export interface RegimeFactorsResponse {
  factors: RegimeFactor[];
  themes: string[];
}

export interface EventPoint {
  offset_weeks: number;
  return_pct: number | null;
}

export interface RegimeAnalog {
  date: string;
  distance: number;
  contributions: Record<string, number>;
  // Per merged dimension: each original factor's own share within that
  // group (sums to 1 within the group). Informational only — not a
  // decomposition of the group's `contributions` percentage above, since
  // the group score is based on the averaged z-score, not summed per-member
  // squared differences. Absent/empty for dimensions that aren't merged.
  member_contributions: Record<string, Record<string, number>>;
  event_series: EventPoint[];
  forward_returns: Record<string, number | null>;
}

export interface RegimeAnalogsResponse {
  query_date: string;
  factors_used: string[];
  vector_dimensions: string[];
  groups: Record<string, string[]>;
  analogs: RegimeAnalog[];
  query_event_series: EventPoint[];
}

export interface RegimeValidationHorizonStats {
  n_conditioned: number;
  n_conditioned_unique?: number;
  n_unconditional?: number;
  conditioned_mean?: number;
  conditioned_unique_mean?: number;
  unconditional_mean?: number;
  conditioned_std?: number;
  unconditional_std?: number;
  ks_statistic?: number;
  ks_pvalue?: number;
  significant_at_5pct?: boolean;
  ks_statistic_unique?: number;
  ks_pvalue_unique?: number;
  significant_at_5pct_unique?: boolean;
  note?: string;
}

export interface RegimeValidateResponse {
  mode: "validate" | "sweep";
  n_query_dates?: number;
  k?: number;
  query_interval_weeks?: number;
  horizons?: Record<string, RegimeValidationHorizonStats>;
  rows?: {
    k: number;
    exclude_weeks: number;
    n_query_dates: number;
    n_unique_analogs: number | null;
    ks_pvalue_unique: number | null;
    significant: boolean | null;
  }[];
}

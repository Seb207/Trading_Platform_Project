/**
 * Typed API client for the FastAPI backend (localhost:8000).
 * All functions throw on HTTP errors so callers can catch and display them.
 */

import type {
  Paper,
  PaperSection,
  StatusResponse,
} from "@/lib/types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ── Generic fetch helper ───────────────────────────────────────────────
async function apiFetch<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `API error ${res.status}`);
  }
  return res.json() as Promise<T>;
}

// ── Status ─────────────────────────────────────────────────────────────
export async function fetchStatus(): Promise<StatusResponse & {
  abstract_index_count: number;
}> {
  return apiFetch("/api/status");
}

// ── Paper list ─────────────────────────────────────────────────────────
export interface PaperListResult {
  status: string;
  papers: Paper[];
  count: number;
}

export async function listPapers(
  category = "",
  limit = 200,
): Promise<PaperListResult> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (category) params.set("category", category);
  return apiFetch(`/api/papers/list?${params}`);
}

// ── Abstract search ────────────────────────────────────────────────────
export interface SearchResult {
  status: string;
  query: string;
  results: Paper[];
  count: number;
}

export async function searchAbstract(
  query: string,
  nResults = 20,
  category = "",
): Promise<SearchResult> {
  return apiFetch("/api/papers/search/abstract", {
    method: "POST",
    body: JSON.stringify({ query, n_results: nResults, category }),
  });
}

// ── Section search ─────────────────────────────────────────────────────
export interface SectionSearchResult {
  status: string;
  query: string;
  results: PaperSection[];
  count: number;
}

export async function searchSections(opts: {
  query: string;
  nResults?: number;
  category?: string;
  arxivId?: string;
  sectionFilter?: string;
}): Promise<SectionSearchResult> {
  return apiFetch("/api/papers/search/section", {
    method: "POST",
    body: JSON.stringify({
      query:          opts.query,
      n_results:      opts.nResults ?? 10,
      category:       opts.category ?? "",
      arxiv_id:       opts.arxivId ?? "",
      section_filter: opts.sectionFilter ?? "",
    }),
  });
}

// ── Analyze paper ──────────────────────────────────────────────────────
export interface AnalyzeResult {
  status: string;
  arxiv_id: string;
  category: string;
  total_chars: number;
  section_count: number;
  section_map: { title: string; char_count: number }[];
  full_content: string;
}

export async function analyzePaper(
  relativePath: string,
): Promise<AnalyzeResult> {
  return apiFetch("/api/papers/analyze", {
    method: "POST",
    body: JSON.stringify({ relative_path: relativePath }),
  });
}

// ── arXiv live search ──────────────────────────────────────────────────
export async function searchArxiv(opts: {
  query?: string;
  maxResults?: number;
  dateFrom?: string;
  dateTo?: string;
  category?: string;
}): Promise<SearchResult> {
  return apiFetch("/api/arxiv/search", {
    method: "POST",
    body: JSON.stringify({
      query:       opts.query ?? "",
      max_results: opts.maxResults ?? 10,
      date_from:   opts.dateFrom ?? "",
      date_to:     opts.dateTo ?? "",
      category:    opts.category ?? "",
    }),
  });
}

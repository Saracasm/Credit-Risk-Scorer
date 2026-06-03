/* ------------------------------------------------------------------ */
/* API client + shared TypeScript types for the Credit Risk Scorer.   */
/* ------------------------------------------------------------------ */

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

/* ---------- Types ---------- */

export type ShapItem = { feature: string; value: number };

export type PredictResult = {
  probability: number;
  predicted_class: number;
  risk_label: "HIGH" | "MEDIUM" | "LOW";
  shap: ShapItem[];
  shap_base_value: number;
};

export type ApplicantInput = {
  age: number;
  monthly_income: number;
  debt_ratio: number;
  revolving_util: number;
  n_30_59_late: number;
  n_60_89_late: number;
  n_90_late: number;
  n_open_lines: number;
  n_real_estate: number;
  n_dependents: number;
};

export type ScenarioResult = {
  probability: number;
  risk_label: string;
  change: number;
  shap: ShapItem[];
};

export type ScenarioResponse = {
  baseline_probability: number;
  baseline_risk_label: string;
  results: ScenarioResult[];
};

export type BatchResultRow = {
  index: number;
  probability: number;
  predicted_class: number;
  risk_label: string;
};

export type BatchResponse = {
  total: number;
  avg_risk: number;
  high_risk_count: number;
  low_risk_count: number;
  rows: BatchResultRow[];
};

export type DashboardData = {
  total: number;
  avg_prob: number;
  high_count: number;
  medium_count: number;
  low_count: number;
  recent: Record<string, unknown>[];
  histogram: number[];
};

export type AdvisorResponse = {
  reply: string;
  session_id: string;
};

export type KnowledgeInfo = {
  available: boolean;
  doc_count: number;
  chunk_count: number;
  docs: string[];
};

/* ---------- Defaults ---------- */

export const DEFAULT_APPLICANT: ApplicantInput = {
  age: 45,
  monthly_income: 5000,
  debt_ratio: 0.3,
  revolving_util: 0.2,
  n_30_59_late: 0,
  n_60_89_late: 0,
  n_90_late: 0,
  n_open_lines: 5,
  n_real_estate: 1,
  n_dependents: 0,
};

export const TEMPLATES: Record<string, ApplicantInput | null> = {
  Custom: null,
  "🟢 Perfect Applicant": {
    age: 42, monthly_income: 12000, debt_ratio: 0.15, revolving_util: 0.12,
    n_30_59_late: 0, n_60_89_late: 0, n_90_late: 0, n_open_lines: 8, n_real_estate: 2, n_dependents: 1,
  },
  "🟡 Young Professional": {
    age: 25, monthly_income: 3500, debt_ratio: 0.35, revolving_util: 0.45,
    n_30_59_late: 0, n_60_89_late: 0, n_90_late: 0, n_open_lines: 3, n_real_estate: 0, n_dependents: 0,
  },
  "🔴 Financial Stress": {
    age: 38, monthly_income: 2800, debt_ratio: 0.85, revolving_util: 0.92,
    n_30_59_late: 3, n_60_89_late: 2, n_90_late: 1, n_open_lines: 12, n_real_estate: 1, n_dependents: 3,
  },
  "🟡 Retiree": {
    age: 67, monthly_income: 3200, debt_ratio: 0.25, revolving_util: 0.15,
    n_30_59_late: 0, n_60_89_late: 0, n_90_late: 0, n_open_lines: 4, n_real_estate: 1, n_dependents: 0,
  },
};

/* ---------- API calls ---------- */

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail =
      typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data);
    throw new Error(detail);
  }
  return data as T;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(JSON.stringify(data));
  return data as T;
}

export async function predictRisk(a: ApplicantInput): Promise<PredictResult> {
  return post<PredictResult>("/api/predict", a);
}

export async function runScenarios(
  baseline: ApplicantInput,
  scenarios: ApplicantInput[],
): Promise<ScenarioResponse> {
  return post<ScenarioResponse>("/api/scenarios", { baseline, scenarios });
}

export async function batchScore(applicants: Record<string, unknown>[]): Promise<BatchResponse> {
  return post<BatchResponse>("/api/batch", { applicants });
}

export async function getDashboard(): Promise<DashboardData> {
  return get<DashboardData>("/api/dashboard");
}

export async function askAdvisor(
  message: string,
  applicant: ApplicantInput | null,
  provider: string = "gemini",
  model_name: string = "gemini-2.0-flash",
  apiKey?: string,
  sessionId?: string | null,
): Promise<AdvisorResponse> {
  return post<AdvisorResponse>("/api/advisor", {
    message,
    applicant,
    provider,
    model_name,
    api_key: apiKey || undefined,
    session_id: sessionId || undefined,
  });
}

export async function downloadReport(applicant: ApplicantInput): Promise<Blob> {
  const res = await fetch(`${API}/api/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ applicant }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Report generation failed");
    throw new Error(text);
  }
  return res.blob();
}

export function checkHealth(): Promise<{ ok: boolean; model_loaded: boolean; error?: string }> {
  return get("/health");
}

export function getKnowledge(): Promise<KnowledgeInfo> {
  return get<KnowledgeInfo>("/api/knowledge");
}

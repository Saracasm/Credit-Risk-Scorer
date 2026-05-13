"use client";

import { ApplicantInput, DEFAULT_APPLICANT, TEMPLATES } from "@/lib/api";

type Props = {
  form: ApplicantInput;
  onChange: (form: ApplicantInput) => void;
};

const FIELDS: {
  key: keyof ApplicantInput;
  label: string;
  min: number;
  max: number;
  step: number;
  tooltip: string;
}[] = [
  { key: "age", label: "Age", min: 18, max: 120, step: 1, tooltip: "Applicant age. Younger (<30) and very old (>70) borrowers default more." },
  { key: "monthly_income", label: "Monthly Income ($)", min: 0, max: 500000, step: 100, tooltip: "Gross monthly income. Higher income = more ability to repay." },
  { key: "debt_ratio", label: "Debt-to-Income", min: 0, max: 10, step: 0.01, tooltip: "Total debt / income. Above 0.5 is risky. Above 1.0 = debt > income." },
  { key: "revolving_util", label: "Credit Utilization", min: 0, max: 1, step: 0.01, tooltip: "% of credit limit used. Above 90% = financial distress signal." },
  { key: "n_30_59_late", label: "30–59 Days Late", min: 0, max: 20, step: 1, tooltip: "Times 30-59 days past due. Mild sign of payment trouble." },
  { key: "n_60_89_late", label: "60–89 Days Late", min: 0, max: 20, step: 1, tooltip: "Times 60-89 days past due. More serious delinquency." },
  { key: "n_90_late", label: "90+ Days Late", min: 0, max: 20, step: 1, tooltip: "Times 90+ days late. Severe — even 1 is a major red flag." },
  { key: "n_open_lines", label: "Open Credit Lines", min: 0, max: 50, step: 1, tooltip: "Open loans & credit cards. Too many indicates over-leveraging." },
  { key: "n_real_estate", label: "Real Estate Loans", min: 0, max: 20, step: 1, tooltip: "Mortgages / home equity loans. 0 = renter, 1-2 = typical owner." },
  { key: "n_dependents", label: "Dependents", min: 0, max: 15, step: 1, tooltip: "People financially dependent on applicant." },
];

export default function ApplicantForm({ form, onChange }: Props) {
  const setField = (key: keyof ApplicantInput, value: number) => {
    onChange({ ...form, [key]: value });
  };

  return (
    <div className="glass-elevated p-6 space-y-5 animate-fade-in-up">
      {/* Template selector */}
      <div className="flex items-center justify-between">
        <h2 className="font-display text-lg font-semibold text-white">
          Applicant Profile
        </h2>
        <select
          className="text-xs px-3 py-1.5 rounded-lg"
          value="Custom"
          onChange={(e) => {
            const t = TEMPLATES[e.target.value];
            if (t) onChange(t);
          }}
        >
          {Object.keys(TEMPLATES).map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </div>

      {/* Input grid */}
      <div className="grid sm:grid-cols-2 gap-4">
        {FIELDS.map((f) => (
          <label key={f.key} className="group block space-y-1.5">
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-on-muted">
                {f.label}
              </span>
              <span
                className="text-[10px] text-outline cursor-help opacity-0 group-hover:opacity-100 transition-opacity"
                title={f.tooltip}
              >
                ⓘ
              </span>
            </div>
            <input
              type="number"
              min={f.min}
              max={f.max}
              step={f.step}
              className="w-full px-3 py-2 text-sm"
              value={form[f.key]}
              onChange={(e) => setField(f.key, Number(e.target.value))}
            />
          </label>
        ))}
      </div>
    </div>
  );
}

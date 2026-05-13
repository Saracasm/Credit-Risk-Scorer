"use client";

import { useState } from "react";
import { ApplicantInput, PredictResult, DEFAULT_APPLICANT } from "@/lib/api";
import ScorerTab from "@/components/tabs/ScorerTab";
import AdvisorTab from "@/components/tabs/AdvisorTab";
import ScenariosTab from "@/components/tabs/ScenariosTab";
import BatchTab from "@/components/tabs/BatchTab";
import DashboardTab from "@/components/tabs/DashboardTab";
import ReportTab from "@/components/tabs/ReportTab";

const TABS = [
  { id: "scorer", label: "Risk Scorer", icon: "🎯" },
  { id: "advisor", label: "Credit Advisor", icon: "🤖" },
  { id: "scenarios", label: "Scenarios", icon: "🔄" },
  { id: "batch", label: "Batch Scoring", icon: "📋" },
  { id: "dashboard", label: "Dashboard", icon: "📊" },
  { id: "report", label: "Report", icon: "📄" },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState("scorer");
  const [form, setForm] = useState<ApplicantInput>(DEFAULT_APPLICANT);
  const [result, setResult] = useState<PredictResult | null>(null);

  return (
    <div className="min-h-screen relative z-10">
      {/* Header */}
      <header className="gradient-bar border-b border-white/10 px-6 py-6 md:px-10">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center gap-3">
            <span className="text-2xl animate-float-icon inline-block">💳</span>
            <h1 className="font-display text-2xl md:text-3xl font-bold gradient-text tracking-tight">
              Credit Risk Scorer
            </h1>
          </div>
          <p className="text-white/60 text-sm mt-2 max-w-2xl leading-relaxed">
            AI-powered credit risk analysis • SHAP explainability • Intelligent advisory • Portfolio analytics
          </p>
        </div>
      </header>

      {/* Tab navigation */}
      <nav className="border-b border-white/5 px-6 md:px-10">
        <div className="max-w-5xl mx-auto flex gap-1 overflow-x-auto py-2 -mb-px">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`tab-button group flex items-center gap-1.5 whitespace-nowrap ${
                activeTab === tab.id ? "active" : ""
              }`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="group-hover:animate-float-icon inline-block transition-transform duration-300">{tab.icon}</span>
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          ))}
        </div>
      </nav>

      {/* Tab content */}
      <main className="max-w-5xl mx-auto px-6 md:px-10 py-8">
        {activeTab === "scorer" && (
          <ScorerTab form={form} setForm={setForm} result={result} setResult={setResult} />
        )}
        {activeTab === "advisor" && <AdvisorTab applicant={form} />}
        {activeTab === "scenarios" && <ScenariosTab baseline={form} />}
        {activeTab === "batch" && <BatchTab />}
        {activeTab === "dashboard" && <DashboardTab />}
        {activeTab === "report" && <ReportTab applicant={form} result={result} />}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 mt-12 py-6 text-center">
        <p className="text-[11px] font-bold text-white/20 uppercase tracking-[0.1em]">
          Deep Credit Intelligence • XGBoost + SHAP + Gemini • FastAPI + React
        </p>
      </footer>
    </div>
  );
}

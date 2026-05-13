"use client";

import { ReactNode } from "react";

/** Glass metric card with icon, value, and optional status badge. */
export default function MetricCard({
  label,
  value,
  icon,
  status,
  statusColor,
}: {
  label: string;
  value: string;
  icon: ReactNode;
  status?: string;
  statusColor?: string;
}) {
  return (
    <div className="glass p-5 animate-fade-in-up">
      <div className="flex justify-between items-start">
        <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-on-muted">
          {label}
        </span>
        <span className="text-lg text-outline">{icon}</span>
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="font-mono text-2xl font-medium text-white">{value}</span>
        {status && (
          <span
            className="text-[10px] font-bold uppercase tracking-wider font-mono"
            style={{ color: statusColor ?? "#10b981" }}
          >
            {status}
          </span>
        )}
      </div>
    </div>
  );
}

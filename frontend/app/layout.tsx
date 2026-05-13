import type { Metadata } from "next";
import { Hanken_Grotesk, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const hanken = Hanken_Grotesk({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-hanken",
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["500"],
  variable: "--font-jetbrains",
});

export const metadata: Metadata = {
  title: "Credit Risk Scorer — AI-Powered Credit Analysis",
  description:
    "Institutional-grade credit risk scoring with XGBoost, SHAP explainability, AI advisor, what-if scenarios, and portfolio analytics. Built with FastAPI + React.",
  keywords: ["credit risk", "machine learning", "SHAP", "XGBoost", "fintech"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${hanken.variable} ${jetbrains.variable} font-sans min-h-screen`}
      >
        {children}
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "ASTRA — Autonomous Strategic Training Agent",
  description: "ML orchestration dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          rel="icon"
          href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚡</text></svg>"
        />
      </head>
      <body>
        <Providers>
          <div className="min-h-screen bg-[#0f172a]">
            <nav
              className="px-6 py-3 flex items-center justify-between"
              style={{ borderBottom: "1px solid rgba(20,184,166,0.1)" }}
            >
              <div className="flex items-center gap-6">
                <Link href="/" className="flex items-center gap-2">
                  <span
                    className="inline-block w-1.5 h-1.5 rounded-full bg-[#14b8a6]"
                    style={{ boxShadow: "0 0 6px #14b8a6", animation: "pulse 3s ease-in-out infinite" }}
                  />
                  <span className="text-[#14b8a6] font-semibold tracking-[0.25em] text-sm hover:text-[#2dd4bf] transition-colors">
                    ASTRA
                  </span>
                </Link>
                <span className="text-[#64748b] text-[10px] tracking-widest hidden sm:block">
                  autonomous·strategic·training·agent
                </span>
              </div>

              <div className="flex items-center gap-1 sm:gap-2">
                <Link
                  href="/"
                  className="px-3 py-1 text-xs text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#1e293b] rounded transition-colors"
                >
                  Missions
                </Link>
                <Link
                  href="/recipes"
                  className="px-3 py-1 text-xs text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#1e293b] rounded transition-colors"
                >
                  Recipes
                </Link>
                <Link
                  href="/models"
                  className="px-3 py-1 text-xs text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#1e293b] rounded transition-colors"
                >
                  Models & Tournaments
                </Link>
              </div>
            </nav>
            <main>{children}</main>

          </div>
        </Providers>
      </body>
    </html>
  );
}

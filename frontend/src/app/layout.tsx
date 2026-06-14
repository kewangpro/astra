import type { Metadata } from "next";
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
              className="px-6 py-3 flex items-center gap-4"
              style={{ borderBottom: "1px solid rgba(20,184,166,0.1)" }}
            >
              <div className="flex items-center gap-2">
                <span
                  className="inline-block w-1.5 h-1.5 rounded-full bg-[#14b8a6]"
                  style={{ boxShadow: "0 0 6px #14b8a6", animation: "pulse 3s ease-in-out infinite" }}
                />
                <span className="text-[#14b8a6] font-semibold tracking-[0.25em] text-sm">ASTRA</span>
              </div>
              <span className="text-[#64748b] text-[10px] tracking-widest hidden sm:block">
                autonomous·strategic·training·agent
              </span>
            </nav>
            <main>{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}

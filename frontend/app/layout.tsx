import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "lolpredictor",
  description: "Find out which friends you play best with in ranked",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

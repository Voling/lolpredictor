import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "lolpredictor",
  description: "Your score with each friend when you duo",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

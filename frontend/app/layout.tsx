import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "lolpredictor",
  description: "Player compatibility from the first fifteen minutes",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

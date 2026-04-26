import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Micro-Certification Recommender",
  description: "Two-tower recommender for L&D micro-certs.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

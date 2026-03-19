import "./globals.css";
import type { Metadata } from "next";
import { Inter, Noto_Serif_Georgian, JetBrains_Mono } from "next/font/google";
import { ReactNode } from "react";

const fontSans = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

const fontSerif = Noto_Serif_Georgian({
  subsets: ["latin"],
  variable: "--font-serif",
});

const fontMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "VideoSceneRAG",
  description: "AI-Powered Video Scene Understanding"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${fontSans.variable} ${fontSerif.variable} ${fontMono.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}

import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import TopBar from "@/components/layout/TopBar";
import CategoryNav from "@/components/layout/CategoryNav";
import { LLMProvider } from "@/context/LLMContext";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Quant Research Dashboard",
  description: "Local-first quant research and portfolio management dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable} h-full`}>
      <body className="h-full flex flex-col bg-bg text-text antialiased">
        <LLMProvider>
          <TopBar />
          <CategoryNav />
          <main className="flex flex-1 overflow-hidden">
            {children}
          </main>
        </LLMProvider>
      </body>
    </html>
  );
}

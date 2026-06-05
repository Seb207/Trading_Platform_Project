import type { Metadata } from "next";
import { Inter, IBM_Plex_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";
import TopBar from "@/components/layout/TopBar";
import CategoryNav from "@/components/layout/CategoryNav";
import { LLMProvider } from "@/context/LLMContext";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const ibmPlexMono = IBM_Plex_Mono({
  variable: "--font-ibm-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
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
    <html lang="en" className={`${inter.variable} ${ibmPlexMono.variable} ${spaceGrotesk.variable} h-full`}>
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

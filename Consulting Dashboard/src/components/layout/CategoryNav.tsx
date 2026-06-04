"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const CATEGORIES = [
  { label: "Research LLM", href: "/research" },
  { label: "Market Regime", href: "/regime" },
  { label: "Portfolio", href: "/portfolio" },
  { label: "Factor Research", href: "/factor" },
] as const;

export default function CategoryNav() {
  const pathname = usePathname();

  return (
    <nav className="flex items-center gap-0.5 px-6 h-[44px] bg-bg2 border-b border-border flex-shrink-0">
      {CATEGORIES.map((cat, i) => {
        const isActive = pathname.startsWith(cat.href);
        return (
          <div key={cat.href} className="flex items-center">
            {/* Divider between items */}
            {i > 0 && (
              <div className="w-px h-5 bg-border mx-2" />
            )}
            <Link
              href={cat.href}
              className={[
                "flex items-center gap-2 px-[18px] py-1.5 rounded text-[12px] font-medium",
                "tracking-[0.5px] uppercase border transition-all duration-150",
                isActive
                  ? "text-accent border-accent bg-accent/5"
                  : "text-text-dim border-transparent hover:text-text hover:border-border",
              ].join(" ")}
            >
              <span
                className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                style={{ backgroundColor: "currentColor" }}
              />
              {cat.label}
            </Link>
          </div>
        );
      })}

      {/* Settings — right-aligned */}
      <Link
        href="/settings"
        className="ml-auto flex items-center gap-2 px-[18px] py-1.5 rounded text-[12px] font-medium tracking-[0.5px] uppercase border border-transparent text-text-dim hover:text-text hover:border-border transition-all duration-150"
      >
        ⚙ Settings
      </Link>
    </nav>
  );
}

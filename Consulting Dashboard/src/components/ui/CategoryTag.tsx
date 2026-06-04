// Color mapping per arXiv category
const CATEGORY_STYLES: Record<string, string> = {
  "q-fin.TR": "bg-accent3/10 text-accent3 border-accent3/20",   // pink  — Trading
  "q-fin.PM": "bg-accent2/10 text-accent2 border-accent2/20",   // cyan  — Portfolio Mgmt
  "q-fin.ST": "bg-accent/10  text-accent  border-accent/20",    // green — Statistical
  "q-fin.CP": "bg-accent4/10 text-accent4 border-accent4/20",   // gold  — Computational
  "q-fin.RM": "bg-neg/10     text-neg     border-neg/20",        // red   — Risk Mgmt
  "q-fin.MF": "bg-accent3/10 text-accent3 border-accent3/20",   // pink  — Math Finance
  "q-fin.PR": "bg-accent4/10 text-accent4 border-accent4/20",   // gold  — Pricing
  "q-fin.GN": "bg-bg3        text-text-mid border-border",       // dim   — General
  "q-fin.EC": "bg-bg3        text-text-mid border-border",       // dim   — Economics
  "cs.AI":    "bg-accent2/10 text-accent2 border-accent2/20",   // cyan  — AI
  "cs.LG":    "bg-accent2/10 text-accent2 border-accent2/20",   // cyan  — ML
};

const DEFAULT_STYLE = "bg-bg3 text-text-mid border-border";

interface CategoryTagProps {
  category: string;
  className?: string;
}

export default function CategoryTag({ category, className = "" }: CategoryTagProps) {
  const style = CATEGORY_STYLES[category] ?? DEFAULT_STYLE;
  return (
    <span
      className={[
        "inline-flex items-center px-1.5 py-0.5 rounded-sm",
        "font-mono text-[9px] font-medium tracking-wide border whitespace-nowrap",
        style,
        className,
      ].join(" ")}
    >
      {category}
    </span>
  );
}

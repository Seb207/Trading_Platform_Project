type BadgeVariant = "green" | "cyan" | "pink" | "gold" | "dim";

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
}

const VARIANT_STYLES: Record<BadgeVariant, string> = {
  green: "bg-accent/10 text-accent border-accent/30",
  cyan:  "bg-accent2/10 text-accent2 border-accent2/30",
  pink:  "bg-accent3/10 text-accent3 border-accent3/30",
  gold:  "bg-accent4/10 text-accent4 border-accent4/30",
  dim:   "bg-bg3 text-text-mid border-border",
};

export default function Badge({
  children,
  variant = "dim",
  className = "",
}: BadgeProps) {
  return (
    <span
      className={[
        "inline-flex items-center px-2 py-0.5 rounded-sm",
        "font-mono text-[10px] tracking-wide border",
        VARIANT_STYLES[variant],
        className,
      ].join(" ")}
    >
      {children}
    </span>
  );
}

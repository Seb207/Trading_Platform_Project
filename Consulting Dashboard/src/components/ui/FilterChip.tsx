"use client";

interface FilterChipProps {
  label: string;
  active?: boolean;
  onClick?: () => void;
}

export default function FilterChip({ label, active = false, onClick }: FilterChipProps) {
  return (
    <button
      onClick={onClick}
      className={[
        "px-2.5 py-1 rounded-sm font-mono text-[10px] tracking-wide border",
        "transition-all duration-100 cursor-pointer",
        active
          ? "border-accent2 text-accent2 bg-accent2/5"
          : "border-border text-text-dim hover:border-border2 hover:text-text",
      ].join(" ")}
    >
      {label}
    </button>
  );
}

"use client";

interface ToggleOption<T extends string> {
  value: T;
  label: string;
}

interface ToggleGroupProps<T extends string> {
  options: ToggleOption<T>[];
  value: T;
  onChange: (value: T) => void;
}

export default function ToggleGroup<T extends string>({
  options,
  value,
  onChange,
}: ToggleGroupProps<T>) {
  return (
    <div className="flex items-center border border-border rounded-sm overflow-hidden">
      {options.map((opt) => {
        const isActive = opt.value === value;
        return (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={[
              "px-3 py-1 font-mono text-[10px] tracking-wide transition-all duration-100",
              "border-r border-border last:border-r-0",
              isActive
                ? "bg-accent/10 text-accent"
                : "text-text-dim hover:text-text hover:bg-bg3",
            ].join(" ")}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

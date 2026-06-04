"use client";

import { useState, useRef, useCallback } from "react";

interface SplitPanelProps {
  left: React.ReactNode;
  right: React.ReactNode;
  defaultRightWidth?: number; // px
  minRightWidth?: number;     // px
  maxRightWidth?: number;     // px
}

export default function SplitPanel({
  left,
  right,
  defaultRightWidth = 380,
  minRightWidth = 240,
  maxRightWidth = 640,
}: SplitPanelProps) {
  const [rightWidth, setRightWidth] = useState(defaultRightWidth);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsDragging(true);
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";

      const handleMouseMove = (e: MouseEvent) => {
        if (!containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const newWidth = rect.right - e.clientX;
        setRightWidth(Math.min(maxRightWidth, Math.max(minRightWidth, newWidth)));
      };

      const handleMouseUp = () => {
        setIsDragging(false);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };

      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    },
    [minRightWidth, maxRightWidth]
  );

  // Double-click to reset to default width
  const handleDoubleClick = useCallback(() => {
    setRightWidth(defaultRightWidth);
  }, [defaultRightWidth]);

  return (
    <div ref={containerRef} className="flex flex-1 overflow-hidden">
      {/* Left panel — flexible */}
      <div className="flex flex-col flex-1 overflow-hidden min-w-0">
        {left}
      </div>

      {/* Draggable divider */}
      <div
        className="relative flex-shrink-0 w-1 cursor-col-resize group"
        style={{
          background: isDragging ? "#00ff88" : "#1f1f1f",
          transition: isDragging ? "none" : "background 0.15s",
        }}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
        title="Drag to resize · Double-click to reset"
      >
        {/* Hover/active hit area (wider than visual line) */}
        <div className="absolute inset-y-0 -left-1 -right-1" />

        {/* Hover glow */}
        <div
          className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-150"
          style={{ background: "#00ff88", boxShadow: "0 0 6px #00ff88" }}
        />

        {/* Drag handle dots — center */}
        <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 flex flex-col items-center justify-center gap-[5px] pointer-events-none z-10">
          {[0, 1, 2, 3].map((i) => (
            <div
              key={i}
              className="w-[3px] h-[3px] rounded-full transition-all duration-150"
              style={{
                background: isDragging
                  ? "#000"
                  : "rgba(255,255,255,0.15)",
                opacity: isDragging ? 1 : undefined,
              }}
            />
          ))}
        </div>
      </div>

      {/* Right panel — fixed, resizable */}
      <div
        className="flex flex-col flex-shrink-0 bg-bg2 overflow-hidden"
        style={{ width: rightWidth }}
      >
        {right}
      </div>
    </div>
  );
}

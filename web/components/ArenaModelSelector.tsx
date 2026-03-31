"use client";

interface ModelPreset {
  label: string;
  value: string;
  description: string;
}

interface ArenaModelSelectorProps {
  models: readonly ModelPreset[];
  selected: string;
  onSelect: (value: string) => void;
  disabled?: boolean;
}

export function ArenaModelSelector({
  models,
  selected,
  onSelect,
  disabled = false,
}: ArenaModelSelectorProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {models.map((model) => {
        const isSelected = model.value === selected;

        return (
          <button
            key={model.value}
            type="button"
            onClick={() => !disabled && onSelect(model.value)}
            disabled={disabled}
            title={model.description}
            className={
              `px-3 py-1 rounded-full text-sm font-medium border transition-colors ` +
              `${isSelected
                ? "border-primary text-primary bg-primary/10"
                : "border-gray-600 text-gray-300 hover:border-gray-400 hover:text-gray-100"
              } ` +
              `${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`
            }
          >
            {model.label}
          </button>
        );
      })}
    </div>
  );
}

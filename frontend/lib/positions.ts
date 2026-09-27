const NAMES: Record<string, string> = { TOP: "top", JUNGLE: "jungle", MIDDLE: "mid", BOTTOM: "bot", UTILITY: "support" };

export const positionName = (position?: string | null) => (position ? NAMES[position] ?? position.toLowerCase() : "");

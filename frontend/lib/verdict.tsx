export const FEW_GAMES = 10;

export function gameCount(count: number): string {
  return `${count} ${count === 1 ? "game" : "games"}`;
}

export type Tone = "great" | "good" | "ok" | "bad";

export function verdict(score: number): { label: string; tone: Tone } {
  const shown = Math.round(score);
  if (shown >= 55) return { label: "Great", tone: "great" };
  if (shown >= 50) return { label: "Good", tone: "good" };
  if (shown > 45) return { label: "Ok", tone: "ok" };
  return { label: "Bad", tone: "bad" };
}

export function Score({ score }: { score: number }) {
  return <span className={`score ${verdict(score).tone}`}>{score.toFixed(0)}</span>;
}

export function Verdict({ score }: { score: number }) {
  const found = verdict(score);
  return <span className={`verdict ${found.tone}`}>{found.label}</span>;
}

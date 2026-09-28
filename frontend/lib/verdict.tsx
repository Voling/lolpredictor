export const FEW_GAMES = 10;

export function gameCount(count: number): string {
  return `${count} ${count === 1 ? "game" : "games"}`;
}

export type Tone = "good" | "ok" | "meh" | "bad";

export function verdict(score: number): { label: string; tone: Tone } {
  if (score >= 60) return { label: "Good", tone: "good" };
  if (score >= 50) return { label: "Ok pairing", tone: "ok" };
  if (score >= 40) return { label: "I've seen better", tone: "meh" };
  return { label: "Definitely reconsider...", tone: "bad" };
}

export function Verdict({ score }: { score: number }) {
  const found = verdict(score);
  return <span className={`verdict ${found.tone}`}>{found.label}</span>;
}

import type { Reading, Standout } from "./api";

const PLURAL: Record<string, string> = { TOP: "top laners", JUNGLE: "junglers", MIDDLE: "mid laners", BOTTOM: "bot laners", UTILITY: "supports" };
const SINGULAR: Record<string, string> = { TOP: "top laner", JUNGLE: "jungler", MIDDLE: "mid laner", BOTTOM: "bot laner", UTILITY: "support" };

export function peers(position: string): string {
  return PLURAL[position] ?? "players";
}

export function peer(position: string): string {
  return SINGULAR[position] ?? "player";
}

export function basisSentence(name: string, games: number, evidence: number | null | undefined, position: string, spoken: string): string {
  const seen = `${name}: ${games} ${games === 1 ? "game" : "games"} as ${spoken}`;
  if (evidence == null) return `${seen}.`;
  if (evidence >= 0.6) return `${seen}, so this is mostly their own play.`;
  if (evidence >= 0.35) return `${seen}, so this is part their own play and part the typical ${peer(position)}.`;
  return `${seen}, so this leans on the typical ${peer(position)} until we see more.`;
}

export function habitSentence(name: string, habit: Standout, position: string): string {
  const more = habit.percentile >= 50;
  const share = Math.round(more ? habit.percentile : 100 - habit.percentile);
  const extreme = share >= 99 ? `almost every ${peer(position)}` : `${share}% of ${peers(position)}`;
  return `${name} ${habit.phrase}, ${more ? "more" : "less"} than ${extreme}.`;
}

export function Habits({ name, position, habits }: { name: string; position: string; habits: Standout[] }) {
  const shown = habits.filter((habit) => habit.percentile >= 80 || habit.percentile <= 20).slice(0, 3);
  if (shown.length === 0) return null;
  return (
    <ul className="habits">
      {shown.map((habit) => (
        <li key={habit.cell}>{habitSentence(name, habit, position)}</li>
      ))}
    </ul>
  );
}

function rounded(value: number): number {
  return Math.round(Math.abs(value) / 10) * 10;
}

export function readingSentence(name: string, reading: Reading, minute: number, position: string): string {
  const total = rounded(reading.gold);
  if (total === 0) return `${name} is projected about even with their opponent at ${minute} minutes, like the typical ${peer(position)}.`;
  return `${name} is projected ${total.toLocaleString("en-US")} gold ${reading.gold > 0 ? "ahead of" : "behind"} their opponent at ${minute} minutes.`;
}

export function togetherSentence(gold: number, games: number, customs: number): string {
  if (games === 0) return "No games together in our data yet.";
  const where = customs > 0 ? `${games} games together, ${customs} customs` : `${games} ranked games together`;
  const rounded = Math.round(Math.abs(gold) / 10) * 10;
  if (rounded === 0) return `${where}: about what your own play predicts.`;
  return `${where}: ${rounded.toLocaleString("en-US")} gold ${gold > 0 ? "better" : "worse"} than your solo play after allowing for luck.`;
}

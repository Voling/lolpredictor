import type { Reading, Standout } from "./api";

const PLURAL: Record<string, string> = { TOP: "top laners", JUNGLE: "junglers", MIDDLE: "mid laners", BOTTOM: "bot laners", UTILITY: "supports" };
const SINGULAR: Record<string, string> = { TOP: "top laner", JUNGLE: "jungler", MIDDLE: "mid laner", BOTTOM: "bot laner", UTILITY: "support" };

export function peers(position: string): string {
  return PLURAL[position] ?? "players";
}

export function peer(position: string): string {
  return SINGULAR[position] ?? "player";
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

function signed(value: number): string {
  const rounded = Math.round(Math.abs(value) / 10) * 10;
  return `${value < 0 ? "\u2212" : "+"}${rounded.toLocaleString("en-US")}`;
}

export function readingSentence(name: string, reading: Reading, minute: number): string {
  const parts = [
    reading.style != null ? `playstyle ${signed(reading.style)}` : null,
    reading.form != null && Math.round(reading.form / 10) !== 0 ? `recent form ${signed(reading.form)}` : null,
    reading.champion != null && Math.round(reading.champion / 10) !== 0 ? `champions ${signed(reading.champion)}` : null,
  ].filter(Boolean);
  return `${name} brings ${signed(reading.gold)} gold at ${minute} minutes${parts.length ? `: ${parts.join(", ")}` : ""}.`;
}

export function togetherSentence(gold: number, games: number, customs: number): string {
  if (games === 0) return "No games together in our data yet.";
  const where = customs > 0 ? `${games} games together, ${customs} of them customs` : `${games} ranked games together`;
  const rounded = Math.round(Math.abs(gold) / 10) * 10;
  if (rounded === 0) return `${where}: about what your own play predicts.`;
  return `${where}: ${rounded.toLocaleString("en-US")} gold ${gold > 0 ? "better" : "worse"} than your own play predicts, after allowing for luck.`;
}

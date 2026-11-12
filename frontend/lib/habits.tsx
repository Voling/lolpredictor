import type { Reading, Standout } from "./api";

const PLURAL: Record<string, string> = { TOP: "top laners", JUNGLE: "junglers", MIDDLE: "mid laners", BOTTOM: "bot laners", UTILITY: "supports" };
const SINGULAR: Record<string, string> = { TOP: "top laner", JUNGLE: "jungler", MIDDLE: "mid laner", BOTTOM: "bot laner", UTILITY: "support" };

export function peers(position: string): string {
  return PLURAL[position] ?? "players";
}

export function peer(position: string): string {
  return SINGULAR[position] ?? "player";
}

export type Side = "left" | "right";

export function Player({ name, side }: { name: string; side: Side }) {
  return <span className={`player ${side}`}>{name}</span>;
}

export function Lead({ gold, even, ahead, behind }: { gold: number; even: string; ahead: string; behind: string }) {
  const shown = rounded(gold);
  if (shown === 0) return <>{even}</>;
  return <span className={gold > 0 ? "ahead" : "behind"}>{shown.toLocaleString("en-US")} gold {gold > 0 ? ahead : behind}</span>;
}

export function Basis({ name, side, games, spoken }: { name: string; side: Side; games: number; spoken: string }) {
  return <p className="sub"><Player name={name} side={side} />: {games} {spoken} {games === 1 ? "game" : "games"} read</p>;
}

export function habitSentence(name: string, habit: Standout, position: string): string {
  const more = habit.percentile >= 50;
  const share = Math.round(more ? habit.percentile : 100 - habit.percentile);
  const extreme = share >= 99 ? `almost every ${peer(position)}` : `${share}% of ${peers(position)}`;
  return `${name} ${habit.phrase}, ${more ? "more" : "less"} than ${extreme}.`;
}

export function Habits({ name, side, position, habits }: { name: string; side: Side; position: string; habits: Standout[] }) {
  const shown = habits.filter((habit) => habit.percentile >= 80 || habit.percentile <= 20).slice(0, 3);
  if (shown.length === 0) return null;
  return (
    <ul className="habits">
      {shown.map((habit) => (
        <li key={habit.cell}><Player name={name} side={side} />{habitSentence("", habit, position)}</li>
      ))}
    </ul>
  );
}

function rounded(value: number): number {
  return Math.round(Math.abs(value) / 10) * 10;
}

export function Projection({ name, side, reading, minute, position }: { name: string; side: Side; reading: Reading; minute: number; position: string }) {
  const even = `about even with their opponent at ${minute} minutes, like the average ${peer(position)}`;
  return (
    <p>
      <Player name={name} side={side} /> is projected <Lead gold={reading.gold} even={even} ahead="ahead of" behind="behind" />{rounded(reading.gold) === 0 ? "." : ` their opponent at ${minute} minutes.`}
    </p>
  );
}

export function togetherSentence(gold: number, games: number, customs: number): string {
  if (games === 0) return "No games together in our data yet.";
  const where = customs > 0 ? `${games} games together, ${customs} customs` : `${games} ranked games together`;
  const rounded = Math.round(Math.abs(gold) / 10) * 10;
  if (rounded === 0) return `${where}: about what your own play predicts.`;
  return `${where}: ${rounded.toLocaleString("en-US")} gold ${gold > 0 ? "better" : "worse"} than your solo play after allowing for luck.`;
}

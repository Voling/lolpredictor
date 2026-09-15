export type Status = {
  ready: boolean;
  informative: boolean;
  players: number;
  known_pairs: number;
  cache: boolean;
  model: Record<string, unknown>;
};

export type PairScore = {
  score: number | null;
  reliable: boolean;
  note: string | null;
  synergy: number;
  projected_gold_at_15: number;
  games_together: number;
  winrate_together: number | null;
  hinge: Record<
    string,
    { converged: number; present: number; nearby: number; triggers: number } | null
  >;
  interaction: {
    synergy: number;
    percentile: number;
    left_seats: number;
    right_seats: number;
  } | null;
  players: { riot_id: string; main_position: string; games: number; winrate: number }[];
};

const base = process.env.API_URL ?? "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${base}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${response.status}: ${await response.text()}`);
  }
  return response.json() as Promise<T>;
}

export const getStatus = () => get<Status>("/api/status");
export const getPair = (a: string, b: string) =>
  get<PairScore>(`/api/pair?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
export const getOutsiderPair = (a: string, b: string) =>
  get<Record<string, unknown>>(
    `/api/outsider-pair?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`,
  );

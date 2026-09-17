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
  projected_gold_at_15: number | null;
  reliable: boolean;
  note: string | null;
  games_together: number;
  winrate_together: number | null;
  hinge: Record<
    string,
    { converged: number; present: number; nearby: number; triggers: number } | null
  >;
  positions: { left: string; right: string } | null;
  warnings: string[];
  interaction: {
    score: number;
    projected_gold_at_15: number | null;
    synergy: number;
    percentile: number;
    reliable: boolean;
    note: string | null;
    positions: { left: string; right: string };
    left_games: number;
    right_games: number;
    left_evidence: number | null;
    right_evidence: number | null;
  } | null;
  players: { riot_id: string; main_position: string; games: number; winrate: number }[];
};

export type Lineup = {
  score: number;
  projected_gold_at_15: number | null;
  synergy: number;
  percentile: number;
  reliable: boolean;
  note?: string;
  warnings: string[];
  pairs: {
    left: string;
    right: string;
    score: number;
    projected_gold_at_15: number | null;
    synergy: number;
    percentile: number;
  }[];
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
export const getPair = (a: string, b: string, aPosition?: string, bPosition?: string) => {
  const query = new URLSearchParams({ a, b });
  if (aPosition) query.set("a_position", aPosition);
  if (bPosition) query.set("b_position", bPosition);
  return get<PairScore>(`/api/pair?${query.toString()}`);
};
export const postLineup = async (players: Record<string, string>) => {
  const response = await fetch(`${base}/api/lineup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ players }),
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`${response.status}: ${await response.text()}`);
  }
  return response.json() as Promise<Lineup>;
};
export const getOutsiderPair = (a: string, b: string) =>
  get<Record<string, unknown>>(
    `/api/outsider-pair?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`,
  );

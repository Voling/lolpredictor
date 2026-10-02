import { accessToken } from "./auth";

export type Status = {
  ready: boolean;
  informative: boolean;
  players: number;
  known_pairs: number;
  cache: boolean;
  model: Record<string, unknown>;
  run: {
    id: string;
    promoted: string | null;
    git: { sha: string | null; dirty: boolean | null } | null;
    matches: number | null;
    metrics: Record<string, unknown> | null;
  } | null;
};

export type RecentSeat = { champion: string | null; tier: string | null; division: string | null; position: string | null };
export type RecentDuo = { left: RecentSeat; right: RecentSeat; score: number; gold: number; minute: number; at: number };
export type Recent = { duos: RecentDuo[] };

export type Me = { riot_id: string | null; verified: boolean; pending: { riot_id: string; icon: number } | null; daily: number; remaining: number };

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export type Reading = { gold: number; score: number; percentile: number; style?: number; form?: number; champion?: number };

export type Standout = { cell: string; words: string; phrase: string; z: number; percentile: number };

export type Together = { gold: number; games: number; customs?: number };

export type Pending = { riot_id: string; status: string; step: string; done: number; total: number; percent: number; message: string | null };

export type Player = { riot_id: string; main_position: string; games: number; winrate: number; latest_game?: number | null; refresh_after?: number | null };

export type Evaluations = { players: Pending[] };

export type SavedCheck = { id: string; at: number; kind: string; names: (string | null)[]; positions: (string | null)[]; score: number | null };

export type History = { checks: SavedCheck[] };

export type Refresh = { message: string | null; pending: Pending | null };

export type PairScore = {
  pending?: Pending | null;
  saved_at?: number;
  score: number | null;
  projected_gold: number | null;
  minute: number | null;
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
    projected_gold: number | null;
    minute: number;
    synergy: number;
    percentile: number;
    reliable: boolean;
    note: string | null;
    positions: { left: string; right: string };
    left_games: number;
    right_games: number;
    left_evidence: number | null;
    right_evidence: number | null;
    edge: { left: Reading; right: Reading; fit: Reading; record?: Together; total: number };
    drivers: { left: string; right: string; contribution: number }[];
    reading: Record<
      string,
      {
        distinctive: Standout[];
        situations: { situation: string; words: string; contribution: number }[];
      }
    >;
  } | null;
  players: Player[];
  remaining?: number;
};

export type FriendRow = {
  standout?: Standout | null;
  pending?: Pending | null;
  riot_id: string;
  note: string | null;
  position?: string;
  score?: number;
  projected_gold?: number | null;
  record?: Together | null;
  reading?: Reading;
  fit?: Reading;
  total?: number;
  games?: number;
  evidence?: number | null;
  games_together?: number;
  thin?: boolean;
};

export type Friends = {
  pending?: Pending | null;
  saved_at?: number;
  me: { riot_id: string; position?: string; reading?: Reading; evidence?: number | null; games?: number; minute?: number };
  friends: FriendRow[];
  remaining?: number;
};


const base = process.env.NEXT_PUBLIC_API_URL ?? "";

async function failure(response: Response): Promise<ApiError> {
  const text = await response.text();
  let detail: unknown = null;
  try {
    detail = JSON.parse(text).detail;
  } catch {
    detail = null;
  }
  return new ApiError(typeof detail === "string" ? detail : `The server answered ${response.status}. Try again in a few minutes.`, response.status);
}

async function authHeaders(): Promise<Record<string, string>> {
  const token = await accessToken();
  return token ? { "x-auth": token } : {};
}

async function hash(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${base}${path}`, { cache: "no-store", headers: await authHeaders() });
  if (!response.ok) {
    throw await failure(response);
  }
  return response.json() as Promise<T>;
}

async function post<T>(path: string, payload: unknown): Promise<T> {
  const body = JSON.stringify(payload ?? {});
  const response = await fetch(`${base}${path}`, {
    method: "POST",
    cache: "no-store",
    headers: { ...(await authHeaders()), "Content-Type": "application/json", "x-amz-content-sha256": await hash(body) },
    body,
  });
  if (!response.ok) {
    throw await failure(response);
  }
  return response.json() as Promise<T>;
}

export const getStatus = () => get<Status>("/api/status");
export const getMe = () => get<Me>("/api/me");
export const getRecent = () => get<Recent>("/api/recent");
export const startLink = (riotId: string) => post<Me>("/api/link", { riot_id: riotId });
export const verifyLink = () => post<Me>("/api/link/verify", {});
export const getPair = (a: string, b: string, aPosition?: string, bPosition?: string) => {
  const query = new URLSearchParams({ a, b });
  if (aPosition) query.set("a_position", aPosition);
  if (bPosition) query.set("b_position", bPosition);
  return get<PairScore>(`/api/pair?${query.toString()}`);
};
export const getFriends = (me: string, friends: string[], mePosition?: string) => {
  const query = new URLSearchParams({ me });
  if (mePosition) query.set("me_position", mePosition);
  for (const friend of friends) query.append("friends", friend);
  return get<Friends>(`/api/friends?${query.toString()}`);
};
export const getEvaluations = (names: string[]) => get<Evaluations>(`/api/evaluations?names=${encodeURIComponent(names.join(","))}`);
export const getHistory = () => get<History>("/api/history");
export const getSaved = <T>(id: string) => get<T>(`/api/history/${encodeURIComponent(id)}`);
export const refreshPlayer = (riotId: string) => post<Refresh>("/api/refresh", { riot_id: riotId });

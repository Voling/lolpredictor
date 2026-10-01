import type { Pending } from "./api";

const WORDS: Record<string, string> = {
  requested: "Waiting for a worker",
  queued: "Waiting for a worker",
  fetching: "Pulling games",
  loading: "Reading games",
  features: "Finding plays",
  positions: "Tracking positions",
  reactions: "Reading reactions",
  tendencies: "Reading tendencies",
  habits: "Reading habits",
  movement: "Reading movement",
  embedding: "Reading each game",
  scoring: "Scoring",
  ready: "Ready",
  failed: "Failed",
  refused: "Not started",
};

export function Progress({ pending }: { pending: Pending }) {
  const words = WORDS[pending.step] ?? WORDS[pending.status] ?? pending.step;
  const count = pending.step === "fetching" && pending.total > 0 ? ` ${pending.done} of ${pending.total}` : "";
  return (
    <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pending.percent} aria-label={`${pending.riot_id}: ${words}`}>
      <div className="progress-bar"><div className="progress-fill" style={{ width: `${Math.max(2, Math.min(100, pending.percent))}%` }} /></div>
      <span className="progress-words">{pending.riot_id}: {words}{count}</span>
    </div>
  );
}

export const POLL_MS = 5000;

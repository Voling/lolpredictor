"use client";

import { useEffect, useRef, useState } from "react";
import { getEvaluations, refreshPlayer, type Pending } from "./api";
import { POLL_MS, Progress } from "./progress";

const WAITING = ["requested", "queued", "running"];

function dateOf(epoch: number): string {
  return new Date(epoch * 1000).toLocaleDateString("en-US", { month: "long", day: "numeric" });
}

export function RefreshButton({ name, after, onReady }: { name: string; after?: number | null; onReady: () => void }) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const [tick, setTick] = useState(0);
  const ready = useRef(onReady);
  ready.current = onReady;
  const waiting = Boolean(pending && WAITING.includes(pending.status));

  useEffect(() => {
    if (!waiting) return;
    const timer = setTimeout(async () => {
      try {
        const found = await getEvaluations([name]);
        const next = found.players[0] ?? null;
        setPending(next);
        if (next?.status === "ready") ready.current();
        else setTick((count) => count + 1);
      } catch (error) {
        setPending(null);
        setMessage(error instanceof Error ? error.message : String(error));
      }
    }, POLL_MS);
    return () => clearTimeout(timer);
  }, [waiting, tick, name]);

  async function start() {
    setBusy(true);
    setMessage(null);
    try {
      const found = await refreshPlayer(name);
      setMessage(found.message);
      setPending(found.pending && WAITING.includes(found.pending.status) ? found.pending : null);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  if (after != null && after * 1000 > Date.now()) {
    return <p className="hint">{`${name}'s games are up to date. New games can be pulled after ${dateOf(after)}.`}</p>;
  }
  return (
    <div className="refresh">
      <button type="button" onClick={start} disabled={busy || waiting}>{`Pull ${name}'s new games`}</button>
      {message && <p className="hint">{message}</p>}
      {pending && <Progress pending={pending} />}
    </div>
  );
}

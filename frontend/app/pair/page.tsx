"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import { useRemote } from "@/lib/remote";
import { getPair, getSaved, type PairScore } from "@/lib/api";
import { positionName } from "@/lib/positions";
import { FEW_GAMES, Verdict, gameCount } from "@/lib/verdict";
import { POLL_MS, Progress } from "@/lib/progress";
import { Habits, peer, readingSentence, togetherSentence } from "@/lib/habits";
import { AccountNotice, accountReady, useAccount } from "@/lib/account";
import { RefreshButton } from "@/lib/refresh";

const POSITIONS = ["", "top", "jungle", "mid", "bot", "support"];

type Query = { a?: string; b?: string; a_position?: string; b_position?: string; saved?: string };

function lead(gold: number): string {
  const rounded = Math.round(Math.abs(gold) / 50) * 50;
  if (rounded === 0) return "about even with";
  return `about ${rounded.toLocaleString()} gold ${gold > 0 ? "ahead of" : "behind"}`;
}

function Result({ pair, onRefreshed }: { pair: PairScore; onRefreshed: () => void }) {
  const found = pair.interaction;
  if (pair.pending) {
    return (
      <>
        <p>{pair.pending.message ?? `We're pulling ${pair.pending.riot_id}'s games.`}</p>
        {pair.pending.status !== "refused" && <Progress pending={pair.pending} />}
      </>
    );
  }
  if (!found) {
    return <div className="gate"><strong>Can&apos;t score this duo.</strong>{pair.note}</div>;
  }
  const [left, right] = pair.players;
  const positions = `${positionName(found.positions.left)} and ${positionName(found.positions.right)}`;
  const thin = [
    { name: left.riot_id, games: found.left_games, position: found.positions.left },
    { name: right.riot_id, games: found.right_games, position: found.positions.right },
  ].filter((player) => player.games < FEW_GAMES);
  return (
    <>
      <h2>Your duo score: {found.score.toFixed(0)}</h2>
      <p className="verdict-line"><Verdict score={found.score} /></p>
      <p>{left.riot_id} as {positionName(found.positions.left)} with {right.riot_id} as {positionName(found.positions.right)}.</p>
      <p>At {found.minute} minutes you two are projected to be {lead(found.edge.total)} the other team&apos;s {positions}.</p>
      {thin.map((player) => (
        <p key={player.name} className="hint">{player.name} has {gameCount(player.games)} as {positionName(player.position)}. Treat this score as rough.</p>
      ))}
      <p className="sub">50 is an average duo for these positions. Higher is better.</p>

      <h3>What each of you brings</h3>
      <p>{readingSentence(left.riot_id, found.edge.left, found.minute)}</p>
      <p>{readingSentence(right.riot_id, found.edge.right, found.minute)}</p>
      {found.edge.record && <p>{togetherSentence(found.edge.record.gold, found.edge.record.games, found.edge.record.customs ?? 0)}</p>}

      <h3>What stands out</h3>
      <Habits name={left.riot_id} position={found.positions.left} habits={found.reading.left.distinctive} />
      <Habits name={right.riot_id} position={found.positions.right} habits={found.reading.right.distinctive} />
      <p className="sub">Compared with every {peer(found.positions.left)} and {peer(found.positions.right)} in our data.</p>

      <h3>Newer games</h3>
      <RefreshButton name={left.riot_id} after={left.refresh_after} onReady={onRefreshed} />
      <RefreshButton name={right.riot_id} after={right.refresh_after} onReady={onRefreshed} />
    </>
  );
}

function PairQuery() {
  const params = useSearchParams();
  const router = useRouter();
  const query: Query = { a: params.get("a") ?? undefined, b: params.get("b") ?? undefined, a_position: params.get("a_position") ?? undefined, b_position: params.get("b_position") ?? undefined, saved: params.get("saved") ?? undefined };
  const account = useAccount();
  const me = account.enabled ? account.me?.riot_id ?? undefined : query.a;
  const ready = accountReady(account) && Boolean(query.saved || (me && query.b));
  const [tick, setTick] = useState(0);
  const { data: pair, error, loading } = useRemote(
    !ready ? null : query.saved ? () => getSaved<PairScore>(query.saved!) : () => getPair(me!, query.b!, query.a_position || undefined, query.b_position || undefined),
    `${params.toString()}|${ready}|${tick}`,
  );
  const liveUrl = pair?.players?.length === 2
    ? `/pair/?a=${encodeURIComponent(pair.players[0].riot_id)}&b=${encodeURIComponent(pair.players[1].riot_id)}${pair.positions ? `&a_position=${positionName(pair.positions.left)}&b_position=${positionName(pair.positions.right)}` : ""}`
    : "/pair/";
  const onRefreshed = useCallback(() => {
    if (query.saved) router.push(liveUrl);
    else setTick((count) => count + 1);
  }, [query.saved, liveUrl, router]);
  const waiting = Boolean(pair?.pending && ["requested", "queued", "running"].includes(pair.pending.status));
  useEffect(() => {
    if (!waiting) return;
    const timer = setTimeout(() => setTick((count) => count + 1), POLL_MS);
    return () => clearTimeout(timer);
  }, [waiting, tick]);
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p className="sub">Enter your Riot ID and a friend&apos;s Riot ID to see how you do as a duo.</p>
      <AccountNotice account={account} remaining={pair?.remaining} />
      {accountReady(account) && <form method="get" action="/pair/" className="pair">
        {!account.enabled && <label>Your Riot ID <input name="a" defaultValue={query.a ?? ""} placeholder="name#tag" required /></label>}
        <label>
          Your position
          <select name="a_position" defaultValue={query.a_position ?? ""}>
            {POSITIONS.map((position) => <option key={position} value={position}>{position || "your main position"}</option>)}
          </select>
        </label>
        <label>Friend&apos;s Riot ID <input name="b" defaultValue={query.b ?? ""} placeholder="name#tag" required /></label>
        <label>
          Their position
          <select name="b_position" defaultValue={query.b_position ?? ""}>
            {POSITIONS.map((position) => <option key={position} value={position}>{position || "their main position"}</option>)}
          </select>
        </label>
        <button type="submit">Check</button>
      </form>}
      {loading && !pair && <p className="sub">Checking your duo…</p>}
      {error && <div className="gate"><strong>Can&apos;t score this duo.</strong>{error}</div>}
      {pair?.saved_at && <p className="hint">Saved on {new Date(pair.saved_at * 1000).toLocaleDateString()}. <Link href={liveUrl}>Check again now</Link>.</p>}
      {pair && <Result pair={pair} onRefreshed={onRefreshed} />}
    </main>
  );
}

export default function PairPage() {
  return (
    <Suspense>
      <PairQuery />
    </Suspense>
  );
}

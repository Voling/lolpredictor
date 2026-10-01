"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { useRemote } from "@/lib/remote";
import { getPair, type PairScore } from "@/lib/api";
import { positionName } from "@/lib/positions";
import { FEW_GAMES, Verdict, gameCount } from "@/lib/verdict";
import { AccountNotice, accountReady, useAccount } from "@/lib/account";

const POSITIONS = ["", "top", "jungle", "mid", "bot", "support"];

type Query = { a?: string; b?: string; a_position?: string; b_position?: string };

function lead(gold: number): string {
  const rounded = Math.round(Math.abs(gold) / 50) * 50;
  if (rounded === 0) return "about even with";
  return `about ${rounded.toLocaleString()} gold ${gold > 0 ? "ahead of" : "behind"}`;
}

function Result({ pair }: { pair: PairScore }) {
  const found = pair.interaction;
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
      {(found.edge.record?.customs ?? 0) > 0 && (
        <p className="hint">Counting {gameCount(found.edge.record!.customs!)} you two played together in customs.</p>
      )}
      <p className="sub">50 is an average duo for these positions. Higher is better.</p>
    </>
  );
}

function PairQuery() {
  const params = useSearchParams();
  const query: Query = { a: params.get("a") ?? undefined, b: params.get("b") ?? undefined, a_position: params.get("a_position") ?? undefined, b_position: params.get("b_position") ?? undefined };
  const account = useAccount();
  const me = account.enabled ? account.me?.riot_id ?? undefined : query.a;
  const ready = accountReady(account) && Boolean(me && query.b);
  const { data: pair, error, loading } = useRemote(
    ready ? () => getPair(me!, query.b!, query.a_position || undefined, query.b_position || undefined) : null,
    `${params.toString()}|${ready}`,
  );
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
      {loading && <p className="sub">Checking your duo…</p>}
      {error && <div className="gate"><strong>Can&apos;t score this duo.</strong>{error}</div>}
      {pair && <Result pair={pair} />}
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

"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { useRemote } from "@/lib/remote";
import { getFriends, getSaved, type Friends } from "@/lib/api";
import { positionName } from "@/lib/positions";
import { FEW_GAMES, Verdict, gameCount } from "@/lib/verdict";
import { AccountNotice, SaveHint, accountReady, typesRiotId, useAccount, useRiotId, visiting } from "@/lib/account";
import { POLL_MS, Progress } from "@/lib/progress";
import { habitSentence, peer } from "@/lib/habits";

const POSITIONS = ["", "top", "jungle", "mid", "bot", "support"];

type Query = { me?: string; me_position?: string; friends?: string; saved?: string };

function Result({ found }: { found: Friends }) {
  const me = found.me;
  return (
    <>
      <h2>Your duo scores{me.position ? ` as ${positionName(me.position)}` : ""}</h2>
      {me.games != null && me.games < FEW_GAMES && (
        <p className="hint">You have only {gameCount(me.games)} as {positionName(me.position)} so your scores lean on the typical {peer(me.position ?? "")} until we see more.</p>
      )}
      <table>
        <thead>
          <tr><th>Friend</th><th>Position</th><th className="num">Score</th><th>Verdict</th></tr>
        </thead>
        <tbody>
          {found.friends.map((row) =>
            row.pending && row.pending.status !== "refused" ? (
              <tr key={row.riot_id}><td>{row.riot_id}</td><td colSpan={3}><Progress pending={row.pending} /></td></tr>
            ) : row.note ? (
              <tr key={row.riot_id}><td>{row.riot_id}</td><td colSpan={3}>{row.note}</td></tr>
            ) : (
              <tr key={row.riot_id}>
                <td>
                  <Link href={`/duo/?a=${encodeURIComponent(me.riot_id)}&b=${encodeURIComponent(row.riot_id)}&b_position=${positionName(row.position)}${me.position ? `&a_position=${positionName(me.position)}` : ""}`}>{row.riot_id}</Link>
                  {row.standout && row.position && (row.standout.percentile >= 80 || row.standout.percentile <= 20) && (
                    <span className="standout">{habitSentence("", row.standout, row.position).replace(/^ /, "").replace(/^\w/, (letter) => letter.toUpperCase())}</span>
                  )}
                </td>
                <td>{positionName(row.position)}</td>
                <td className="num">{row.score?.toFixed(0)}</td>
                <td>
                  {row.score != null && <Verdict score={row.score} />}
                  {row.games != null && row.games < FEW_GAMES && <span className="few">few games</span>}
                </td>
              </tr>
            ),
          )}
        </tbody>
      </table>
      <p className="sub">50 is an average duo. Higher is better but it doesn't equal winrate!</p>
      <p className="sub">Few games means under {FEW_GAMES} games in that position. Lolpredictor fills the gaps with what a typical player might do: their score might hover near 50 until we see more.</p>
    </>
  );
}

function FriendsQuery() {
  const params = useSearchParams();
  const query: Query = { me: params.get("me") ?? undefined, me_position: params.get("me_position") ?? undefined, friends: params.get("friends") ?? undefined, saved: params.get("saved") ?? undefined };
  const friends = (query.friends ?? "").split(/[\n,]/).map((line) => line.trim()).filter(Boolean);
  const account = useAccount();
  const visitor = visiting(account);
  const asks = typesRiotId(account);
  const yours = useRiotId(asks ? query.me : undefined);
  const me = asks ? query.me : account.me?.riot_id ?? undefined;
  const saved = visitor ? undefined : query.saved;
  const allowed = visitor || accountReady(account);
  const ready = allowed && Boolean(saved || (me && friends.length > 0));
  const [tick, setTick] = useState(0);
  const { data: found, error, loading } = useRemote(
    !ready ? null : saved ? () => getSaved<Friends>(saved) : () => getFriends(me!, friends, query.me_position || undefined),
    `${params.toString()}|${ready}|${tick}`,
  );
  const liveUrl = found?.friends
    ? `/friends/?me=${encodeURIComponent(found.me.riot_id)}&me_position=${positionName(found.me.position)}&friends=${encodeURIComponent(found.friends.map((row) => (row.position ? `${row.riot_id}:${positionName(row.position)}` : row.riot_id)).join("\n"))}`
    : "/friends/";
  const waiting = Boolean(
    (found?.pending && ["requested", "queued", "running"].includes(found.pending.status)) ||
      found?.friends?.some((row) => row.pending && ["requested", "queued", "running"].includes(row.pending.status)),
  );
  useEffect(() => {
    if (!waiting) return;
    const timer = setTimeout(() => setTick((count) => count + 1), POLL_MS);
    return () => clearTimeout(timer);
  }, [waiting, tick]);
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p className="sub">Enter your Riot ID and your friends&apos; Riot IDs.</p>
      {!visitor && <AccountNotice account={account} remaining={found?.remaining} />}
      {allowed && <form method="get" action="/friends/" className="pair">
        {asks && <label>Your Riot ID <input key={yours} name="me" defaultValue={yours} placeholder="name#tag" required /></label>}
        <label>
          Your position
          <select name="me_position" defaultValue={query.me_position ?? ""}>
            {POSITIONS.map((position) => <option key={position} value={position}>{position || "your main position"}</option>)}
          </select>
        </label>
        <label>
          Friends, one per line
          <textarea name="friends" defaultValue={query.friends ?? ""} rows={4} placeholder={"friend#tag\nother#tag:support"} required />
        </label>
        <button type="submit">Rank</button>
      </form>}
      {allowed && <p className="hint">To set a friend&apos;s position, add it after their name. For example: friend#tag:jungle</p>}
      {loading && !found && <p className="sub">Ranking your friends…</p>}
      {error && <div className="gate"><strong>Can&apos;t rank these friends.</strong>{error}</div>}
      {found?.saved_at && <p className="hint">Saved on {new Date(found.saved_at * 1000).toLocaleDateString()}. <Link href={liveUrl}>Rank again now</Link>.</p>}
      {found?.pending && (
        <>
          <p>{found.pending.message ?? `Pulling ${found.pending.riot_id}'s games.`}</p>
          <Progress pending={found.pending} />
          <p className="sub">Their last 50 ranked games are enough.</p>
        </>
      )}
      {found && !found.pending && <Result found={found} />}
      {visitor && <SaveHint />}
    </main>
  );
}

export default function FriendsPage() {
  return (
    <Suspense>
      <FriendsQuery />
    </Suspense>
  );
}

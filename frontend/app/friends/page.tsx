"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { useRemote } from "@/lib/remote";
import { getFriends, type Friends } from "@/lib/api";
import { positionName } from "@/lib/positions";
import { FEW_GAMES, Verdict, gameCount } from "@/lib/verdict";
import { AccountNotice, accountReady, useAccount } from "@/lib/account";

const POSITIONS = ["", "top", "jungle", "mid", "bot", "support"];

type Query = { me?: string; me_position?: string; friends?: string };

function Result({ found }: { found: Friends }) {
  const me = found.me;
  return (
    <>
      <h2>Your duo scores{me.position ? ` as ${positionName(me.position)}` : ""}</h2>
      {me.games != null && me.games < FEW_GAMES && (
        <p className="hint">You have {gameCount(me.games)} as {positionName(me.position)}. Your scores are rough until you play more.</p>
      )}
      <table>
        <thead>
          <tr><th>Friend</th><th>Position</th><th className="num">Score</th><th>Verdict</th></tr>
        </thead>
        <tbody>
          {found.friends.map((row) =>
            row.note ? (
              <tr key={row.riot_id}><td>{row.riot_id}</td><td colSpan={3}>{row.note}</td></tr>
            ) : (
              <tr key={row.riot_id}>
                <td>{row.riot_id}</td>
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
      <p className="sub">50 is an average duo for those two positions. Higher is better.</p>
      <p className="sub">Few games means that friend has under {FEW_GAMES} games in that position. Their score stays near 50 until they play more.</p>
    </>
  );
}

function FriendsQuery() {
  const params = useSearchParams();
  const query: Query = { me: params.get("me") ?? undefined, me_position: params.get("me_position") ?? undefined, friends: params.get("friends") ?? undefined };
  const friends = (query.friends ?? "").split(/[\n,]/).map((line) => line.trim()).filter(Boolean);
  const account = useAccount();
  const me = account.enabled ? account.me?.riot_id ?? undefined : query.me;
  const ready = accountReady(account) && Boolean(me && friends.length > 0);
  const { data: found, error, loading } = useRemote(
    ready ? () => getFriends(me!, friends, query.me_position || undefined) : null,
    `${params.toString()}|${ready}`,
  );
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p className="sub">Enter your Riot ID and your friends&apos; Riot IDs. Each friend gets a duo score with you.</p>
      <AccountNotice account={account} remaining={found?.remaining} />
      {accountReady(account) && <form method="get" action="/friends/" className="pair">
        {!account.enabled && <label>Your Riot ID <input name="me" defaultValue={query.me ?? ""} placeholder="name#tag" required /></label>}
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
      {accountReady(account) && <p className="hint">To set a friend&apos;s position, add it after their name. For example: friend#tag:jungle</p>}
      {loading && <p className="sub">Ranking your friends…</p>}
      {error && <div className="gate"><strong>Can&apos;t rank these friends.</strong>{error}</div>}
      {found && <Result found={found} />}
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

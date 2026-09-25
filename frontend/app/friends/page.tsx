"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { useRemote } from "@/lib/remote";
import { getFriends, type Friends } from "@/lib/api";

const POSITIONS = ["", "top", "jungle", "mid", "bot", "support"];

type Query = { me?: string; me_position?: string; friends?: string };

function Result({ found }: { found: Friends }) {
  const me = found.me;
  return (
    <>
      <h2>your score with each friend{me.position ? `, you as ${me.position.toLowerCase()}` : ""}</h2>
      <table>
        <thead>
          <tr><th>friend</th><th>position</th><th className="num">score</th></tr>
        </thead>
        <tbody>
          {found.friends.map((row) =>
            row.note ? (
              <tr key={row.riot_id}><td>{row.riot_id}</td><td colSpan={2}>{row.note}</td></tr>
            ) : (
              <tr key={row.riot_id}>
                <td>{row.riot_id}</td>
                <td>{row.position?.toLowerCase()}</td>
                <td className="num">{row.score?.toFixed(0)}</td>
              </tr>
            ),
          )}
        </tbody>
      </table>
      <p className="sub">50 is an average duo in those two positions.</p>
    </>
  );
}

function FriendsQuery() {
  const params = useSearchParams();
  const query: Query = { me: params.get("me") ?? undefined, me_position: params.get("me_position") ?? undefined, friends: params.get("friends") ?? undefined };
  const friends = (query.friends ?? "").split(/[\n,]/).map((line) => line.trim()).filter(Boolean);
  const ready = Boolean(query.me && friends.length > 0);
  const { data: found, error, loading } = useRemote(
    ready ? () => getFriends(query.me!, friends, query.me_position || undefined) : null,
    params.toString(),
  );
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p className="sub">Which friend to queue with.</p>
      <form method="get" action="/friends/" className="pair">
        <label>you <input name="me" defaultValue={query.me ?? ""} placeholder="name#tag" required /></label>
        <select name="me_position" defaultValue={query.me_position ?? ""}>
          {POSITIONS.map((position) => <option key={position} value={position}>{position || "your main position"}</option>)}
        </select>
        <label>
          friends, one per line, add :jungle to pick their position
          <textarea name="friends" defaultValue={query.friends ?? ""} rows={4} placeholder={"friend#tag\nother#tag:support"} required />
        </label>
        <button type="submit">rank</button>
      </form>
      {loading && <p className="sub">ranking</p>}
      {error && <div className="gate"><strong>No reading</strong>{error}</div>}
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

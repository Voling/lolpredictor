import Link from "next/link";
import { getFriends, type Friends } from "@/lib/api";

const POSITIONS = ["", "top", "jungle", "mid", "bot", "support"];

type Query = { me?: string; me_position?: string; friends?: string };

const signed = (gold: number) => `${gold > 0 ? "+" : ""}${Math.round(gold).toLocaleString()}`;
const share = (evidence?: number | null) => (evidence == null ? "" : `${Math.round(evidence * 100)}%`);

function Result({ found }: { found: Friends }) {
  const me = found.me;
  return (
    <>
      {me.reading && (
        <h2>
          you as {me.position?.toLowerCase()}: {signed(me.reading.gold)} gold at 15, {me.reading.percentile.toFixed(0)}th
          {me.evidence != null && me.evidence < 0.5 ? `, only ${share(me.evidence)} your own evidence` : ""}
        </h2>
      )}
      <p className="sub">
        each friend against a typical opponent in their position, ranked by what the two of you project together
      </p>
      <table>
        <thead>
          <tr><th>friend</th><th>position</th><th className="num">projected gold at 15</th><th className="num">rank</th><th className="num">own evidence</th><th className="num">games known</th><th className="num">fit</th><th className="num">together</th></tr>
        </thead>
        <tbody>
          {found.friends.map((row) =>
            row.note ? (
              <tr key={row.riot_id}><td>{row.riot_id}</td><td colSpan={7}>{row.note}</td></tr>
            ) : (
              <tr key={row.riot_id} className={row.thin ? "thin" : undefined}>
                <td>{row.riot_id}</td>
                <td>{row.position?.toLowerCase()}</td>
                <td className="num">{signed(row.reading!.gold)}</td>
                <td className="num">{row.reading!.percentile.toFixed(0)}th</td>
                <td className="num">{share(row.evidence)}{row.thin ? " thin" : ""}</td>
                <td className="num">{row.games?.toLocaleString()}</td>
                <td className="num">{signed(row.fit!.gold)}</td>
                <td className="num">{row.games_together?.toLocaleString()}</td>
              </tr>
            ),
          )}
        </tbody>
      </table>
      <p>
        The reading is playstyle from each friend's other games, calibrated on games the model never saw, and it
        is the number that decides. A thin row is mostly the position's prior, which the corpus scores low because
        players with few games there are often off role; it says how little is known, not how they play. The fit
        column is what the pairing adds beyond who each of you is, about twenty gold across the corpus, inside the
        noise of one game.
      </p>
    </>
  );
}

export default async function FriendsPage({ searchParams }: { searchParams: Promise<Query> }) {
  const query = await searchParams;
  const friends = (query.friends ?? "").split(/[\n,]/).map((line) => line.trim()).filter(Boolean);
  let found: Friends | null = null;
  let error: string | null = null;
  if (query.me && friends.length > 0) {
    try {
      found = await getFriends(query.me, friends, query.me_position || undefined);
    } catch (exception) {
      error = String(exception);
    }
  }
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p className="sub">Which friend to queue with, from the first fifteen minutes.</p>
      <form method="get" action="/friends" className="pair">
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
      {error && <div className="gate"><strong>No reading</strong>{error}</div>}
      {found && <Result found={found} />}
    </main>
  );
}

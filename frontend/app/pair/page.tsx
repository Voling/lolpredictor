import Link from "next/link";
import { getPair, type PairScore, type Reading } from "@/lib/api";

const POSITIONS = ["", "top", "jungle", "mid", "bot", "support"];
const GAMES_TO_TELL = 2500;

type Query = { a?: string; b?: string; a_position?: string; b_position?: string };

const signed = (gold: number) => `${gold > 0 ? "+" : ""}${Math.round(gold).toLocaleString()}`;

function Row({ label, who, reading, evidence }: { label: string; who: string; reading: Reading; evidence?: number | null }) {
  return (
    <tr>
      <td>{label}</td>
      <td>{who}</td>
      <td className="num">{signed(reading.gold)}{reading.error != null && reading.error > 0 ? ` ± ${Math.round(reading.error)}` : ""}</td>
      <td className="num">{reading.score.toFixed(0)}</td>
      <td className="num">{reading.percentile.toFixed(0)}th</td>
      <td className="num">{evidence == null ? "" : `${Math.round(evidence * 100)}%`}</td>
    </tr>
  );
}

function Result({ pair }: { pair: PairScore }) {
  const fit = pair.interaction;
  if (!fit) {
    return <div className="gate"><strong>No reading</strong>{pair.note}</div>;
  }
  const [left, right] = pair.players;
  return (
    <>
      <h2>{signed(fit.edge.total)} gold at 15 minutes</h2>
      <p className="sub">
        the two of you against the same two enemy positions, from behaviour in your other games
      </p>
      <table>
        <thead>
          <tr><th>part</th><th>who</th><th className="num">gold</th><th className="num">score</th><th className="num">rank</th><th className="num">own evidence</th></tr>
        </thead>
        <tbody>
          <Row label={fit.positions.left.toLowerCase()} who={left.riot_id} reading={fit.edge.left} evidence={fit.left_evidence} />
          <Row label={fit.positions.right.toLowerCase()} who={right.riot_id} reading={fit.edge.right} evidence={fit.right_evidence} />
          <Row label="fit" who="the two together" reading={fit.edge.fit} />
        </tbody>
      </table>
      {!fit.reliable && <div className="gate"><strong>Fit shown for inspection only</strong>{fit.note}</div>}
      <p>
        The fit is what the pairing adds beyond who each of you is, ranked among corpus pairs the model knows
        about as well as it knows you two, with the disagreement between five model seeds as its error. For
        most pairs it sits inside the noise of a single game, so the two seat rows carry the decision.
        {fit.drivers.length > 0 && (
          <> It comes from {fit.drivers.slice(0, 3).map((driver) => `${driver.words} (${signed(driver.contribution)})`).join(", ")}.</>
        )}
      </p>
      <p>
        Games together in the corpus: {pair.games_together.toLocaleString()}.{" "}
        {pair.games_together < GAMES_TO_TELL
          ? `Too few to measure the fit from your own record, which takes about ${GAMES_TO_TELL.toLocaleString()} games.`
          : "Enough games to compare the fit against your own record."}
      </p>
      {pair.warnings.length > 0 && (
        <ul>{pair.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
      )}
    </>
  );
}

export default async function PairPage({ searchParams }: { searchParams: Promise<Query> }) {
  const query = await searchParams;
  let pair: PairScore | null = null;
  let error: string | null = null;
  if (query.a && query.b) {
    try {
      pair = await getPair(query.a, query.b, query.a_position || undefined, query.b_position || undefined);
    } catch (exception) {
      error = String(exception);
    }
  }
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p className="sub">Who to queue with, from the first fifteen minutes.</p>
      <form method="get" action="/pair" className="pair">
        <label>you <input name="a" defaultValue={query.a ?? ""} placeholder="name#tag" required /></label>
        <select name="a_position" defaultValue={query.a_position ?? ""}>
          {POSITIONS.map((position) => <option key={position} value={position}>{position || "any position"}</option>)}
        </select>
        <label>friend <input name="b" defaultValue={query.b ?? ""} placeholder="name#tag" required /></label>
        <select name="b_position" defaultValue={query.b_position ?? ""}>
          {POSITIONS.map((position) => <option key={position} value={position}>{position || "any position"}</option>)}
        </select>
        <button type="submit">read</button>
      </form>
      {error && <div className="gate"><strong>No reading</strong>{error}</div>}
      {pair && <Result pair={pair} />}
    </main>
  );
}

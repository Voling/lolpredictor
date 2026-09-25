"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { useRemote } from "@/lib/remote";
import { getPair, type PairScore } from "@/lib/api";

const POSITIONS = ["", "top", "jungle", "mid", "bot", "support"];

type Query = { a?: string; b?: string; a_position?: string; b_position?: string };

const signed = (gold: number) => `${gold > 0 ? "+" : ""}${Math.round(gold).toLocaleString()}`;

function Result({ pair }: { pair: PairScore }) {
  const found = pair.interaction;
  if (!found) {
    return <div className="gate"><strong>No reading</strong>{pair.note}</div>;
  }
  const [left, right] = pair.players;
  return (
    <>
      <h2>your score together: {found.score.toFixed(0)}</h2>
      <p className="sub">
        {left.riot_id} as {found.positions.left.toLowerCase()} with {right.riot_id} as {found.positions.right.toLowerCase()}:{" "}
        {signed(found.edge.total)} gold at {found.minute} minutes against the same two positions. 50 is an average duo.
      </p>
    </>
  );
}

function PairQuery() {
  const params = useSearchParams();
  const query: Query = { a: params.get("a") ?? undefined, b: params.get("b") ?? undefined, a_position: params.get("a_position") ?? undefined, b_position: params.get("b_position") ?? undefined };
  const ready = Boolean(query.a && query.b);
  const { data: pair, error, loading } = useRemote(
    ready ? () => getPair(query.a!, query.b!, query.a_position || undefined, query.b_position || undefined) : null,
    params.toString(),
  );
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p className="sub">Your score with one friend.</p>
      <form method="get" action="/pair/" className="pair">
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
      {loading && <p className="sub">reading</p>}
      {error && <div className="gate"><strong>No reading</strong>{error}</div>}
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

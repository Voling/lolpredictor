"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import { useRemote } from "@/lib/remote";
import { getPair, getSaved, getShared, type PairScore } from "@/lib/api";
import { positionName } from "@/lib/positions";
import { FEW_GAMES, Score, Verdict, gameCount } from "@/lib/verdict";
import { POLL_MS, Progress } from "@/lib/progress";
import { Basis, Habits, Lead, Player, Projection, peer, togetherSentence } from "@/lib/habits";
import { AccountNotice, SaveHint, accountReady, typesRiotId, useAccount, useRiotId, visiting } from "@/lib/account";
import { RefreshButton } from "@/lib/refresh";
import { DifferenceChart, SituationChart } from "@/lib/posterior";
import { ShareLink } from "@/lib/share";

const POSITIONS = ["", "top", "jungle", "mid", "bot", "support"];

type Query = { a?: string; b?: string; a_position?: string; b_position?: string; saved?: string; share?: string };

function Result({ pair, onRefreshed, shared = false, mine = true }: { pair: PairScore; onRefreshed: () => void; shared?: boolean; mine?: boolean }) {
  const found = pair.interaction;
  if (pair.pending) {
    return (
      <>
        <p>{pair.pending.message ?? `We're pulling ${pair.pending.riot_id}'s games.`}</p>
        {pair.pending.status !== "refused" && <Progress pending={pair.pending} />}
        {pair.pending.status !== "refused" && <p className="sub">Their last 50 ranked games are enough. We read them against every player in the same position, so we don&apos;t need their whole history.</p>}
      </>
    );
  }
  if (pair.expired) {
    const names = pair.players.map((player) => player.riot_id).join(" and ");
    return <div className="gate"><strong>This result has expired.</strong>New games were pulled for {names} after it was shared. Ask for a new link.</div>;
  }
  if (!found) {
    return <div className="gate"><strong>Can&apos;t score this duo.</strong>{pair.note}</div>;
  }
  const [left, right] = pair.players;
  const posteriors = found.posteriors;
  const positions = `${positionName(found.positions.left)} and ${positionName(found.positions.right)}`;
  const thin = [
    { name: left.riot_id, games: found.left_games, position: found.positions.left },
    { name: right.riot_id, games: found.right_games, position: found.positions.right },
  ].filter((player) => player.games < FEW_GAMES);
  return (
    <>
      <h2>{shared || !mine ? "Duo score" : "Your duo score"}: <Score score={found.score} /></h2>
      <p className="verdict-line"><Verdict score={found.score} /></p>
      <p><Player name={left.riot_id} side="left" /> as {positionName(found.positions.left)} with <Player name={right.riot_id} side="right" /> as {positionName(found.positions.right)}.</p>
      <p>At {found.minute} minutes you two are projected to be <Lead gold={found.edge.total} even="about even with" ahead="ahead of" behind="behind" /> the other team&apos;s {positions}.</p>
      {thin.map((player) => (
        <p key={player.name} className="hint">{player.name} has only {gameCount(player.games)} as {positionName(player.position)}, so the score leans on the average {peer(player.position)}.</p>
      ))}
      <p className="sub">50 is an average duo for these positions. Higher is better.</p>

      <h3>What each of you brings</h3>
      <Projection name={left.riot_id} side="left" reading={found.edge.left} minute={found.minute} position={found.positions.left} />
      <Projection name={right.riot_id} side="right" reading={found.edge.right} minute={found.minute} position={found.positions.right} />
      {found.edge.record && <p>{togetherSentence(found.edge.record.gold, found.edge.record.games, found.edge.record.customs ?? 0)}</p>}
      <Basis name={left.riot_id} side="left" games={found.left_games} spoken={positionName(found.positions.left)} />
      <Basis name={right.riot_id} side="right" games={found.right_games} spoken={positionName(found.positions.right)} />

      <h3>What stands out</h3>
      <Habits name={left.riot_id} side="left" position={found.positions.left} habits={found.reading.left.distinctive} />
      <Habits name={right.riot_id} side="right" position={found.positions.right} habits={found.reading.right.distinctive} />
      <p className="sub">Compared with every {peer(found.positions.left)} and {peer(found.positions.right)} we have seen. Habits are only shown when there is enough proof.</p>

      {posteriors && (posteriors.measured ? posteriors.measured.left || posteriors.measured.right : posteriors.left.length > 0 || posteriors.right.length > 0) && (
        <>
          <h3>Your behavior</h3>
          <p className="sub">The three things each of you does most differently from others in your position. Each row is one thing a player can do in that spot. The dot shows how often this player does it, and the bar shows the range we&apos;re 80% sure of. The diamond shows how often the average player in that position does it. A player with few games gets a wide bar and a dot close to the diamond, because we assume average play until their games show otherwise.</p>
          <h4><Player name={left.riot_id} side="left" /></h4>
          {posteriors.left.length === 0 && <p className="sub">Nothing in {left.riot_id}&apos;s games differs clearly from the average {peer(found.positions.left)} yet. That&apos;s normal with few games.</p>}
          {posteriors.left.map((posterior) => <SituationChart key={posterior.situation} posterior={posterior} tone="left" name={left.riot_id} position={found.positions.left} />)}
          <h4><Player name={right.riot_id} side="right" /></h4>
          {posteriors.right.length === 0 && <p className="sub">Nothing in {right.riot_id}&apos;s games differs clearly from the average {peer(found.positions.right)} yet. That&apos;s normal with few games.</p>}
          {posteriors.right.map((posterior) => <SituationChart key={posterior.situation} posterior={posterior} tone="right" name={right.riot_id} position={found.positions.right} />)}
        </>
      )}
      {posteriors && posteriors.differences.length > 0 && (
        <>
          <h3>Your differences</h3>
          <p className="sub">The three situations where you two act least alike. <Player name={left.riot_id} side="left" /> in blue, <Player name={right.riot_id} side="right" /> in amber.</p>
          {posteriors.differences.map((difference) => <DifferenceChart key={difference.situation} difference={difference} />)}
        </>
      )}

      {!shared && (
        <>
          <h3>Newer games</h3>
          <RefreshButton name={left.riot_id} after={left.refresh_after} onReady={onRefreshed} />
          <RefreshButton name={right.riot_id} after={right.refresh_after} onReady={onRefreshed} />
        </>
      )}
      {!shared && pair.share && <ShareLink token={pair.share} />}
    </>
  );
}

function SharedQuery({ token }: { token: string }) {
  const { data: pair, error, loading } = useRemote(() => getShared(token), `share|${token}`);
  const ignore = useCallback(() => undefined, []);
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      {pair?.shared_at && !pair.expired && <p className="hint">Shared result, checked on {new Date(pair.shared_at * 1000).toLocaleDateString()}.</p>}
      {loading && !pair && <p className="sub">Loading the shared result…</p>}
      {error && <div className="gate"><strong>Can&apos;t open this link.</strong>{error}</div>}
      {pair && <Result pair={pair} onRefreshed={ignore} shared />}
      <p><Link href="/duo/">Check your own duo</Link></p>
    </main>
  );
}

function PairQuery() {
  const params = useSearchParams();
  const router = useRouter();
  const query: Query = { a: params.get("a") ?? undefined, b: params.get("b") ?? undefined, a_position: params.get("a_position") ?? undefined, b_position: params.get("b_position") ?? undefined, saved: params.get("saved") ?? undefined, share: params.get("share") ?? undefined };
  const account = useAccount();
  const visitor = visiting(account);
  const asks = typesRiotId(account);
  const member = account.enabled && account.signedIn && Boolean(account.me);
  const linkedId = account.me?.verified ? account.me.riot_id ?? undefined : undefined;
  const yours = useRiotId(asks ? query.a : undefined);
  const first = asks ? yours : query.a || linkedId || "";
  const me = query.a || (asks ? undefined : linkedId);
  const saved = visitor ? undefined : query.saved;
  const allowed = visitor || accountReady(account);
  const ready = allowed && Boolean(saved || (me && query.b));
  const [tick, setTick] = useState(0);
  const { data: pair, error, loading } = useRemote(
    !ready ? null : saved ? () => getSaved<PairScore>(saved) : () => getPair(me!, query.b!, query.a_position || undefined, query.b_position || undefined),
    `${params.toString()}|${ready}|${tick}`,
  );
  const liveUrl = pair?.players?.length === 2
    ? `/duo/?a=${encodeURIComponent(pair.players[0].riot_id)}&b=${encodeURIComponent(pair.players[1].riot_id)}${pair.positions ? `&a_position=${positionName(pair.positions.left)}&b_position=${positionName(pair.positions.right)}` : ""}`
    : "/duo/";
  const onRefreshed = useCallback(() => {
    if (query.saved) router.push(liveUrl);
    else setTick((count) => count + 1);
  }, [query.saved, liveUrl, router]);
  const waiting = Boolean(pair?.pending && ["requested", "queued", "running"].includes(pair.pending.status));
  const mine = !member || Boolean(linkedId && pair?.players?.some((player) => player.riot_id.toLowerCase() === linkedId.toLowerCase()));
  useEffect(() => {
    if (!waiting) return;
    const timer = setTimeout(() => setTick((count) => count + 1), POLL_MS);
    return () => clearTimeout(timer);
  }, [waiting, tick]);
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p className="sub">{member ? "Enter two Riot IDs to see how they do as a duo." : "Enter your Riot ID and a friend's Riot ID to see how you do as a duo."}</p>
      {!visitor && <AccountNotice account={account} remaining={pair?.remaining} named={false} />}
      {allowed && <form method="get" action="/duo/" className="pair">
        <label>{member ? "Player one" : "Your Riot ID"} <input key={first} name="a" defaultValue={first} placeholder="name#tag" required /></label>
        <label>
          {member ? "Player one's position" : "Your position"}
          <select name="a_position" defaultValue={query.a_position ?? ""}>
            {POSITIONS.map((position) => <option key={position} value={position}>{position || (member ? "main position" : "your main position")}</option>)}
          </select>
        </label>
        <label>{member ? "Player two" : "Friend's Riot ID"} <input name="b" defaultValue={query.b ?? ""} placeholder="name#tag" required /></label>
        <label>
          {member ? "Player two's position" : "Their position"}
          <select name="b_position" defaultValue={query.b_position ?? ""}>
            {POSITIONS.map((position) => <option key={position} value={position}>{position || (member ? "main position" : "their main position")}</option>)}
          </select>
        </label>
        <button type="submit">Check</button>
      </form>}
      {loading && !pair && <p className="sub">Checking your duo…</p>}
      {error && <div className="gate"><strong>Can&apos;t score this duo.</strong>{error}</div>}
      {pair?.saved_at && <p className="hint">Saved on {new Date(pair.saved_at * 1000).toLocaleDateString()}. <Link href={liveUrl}>Check again now</Link>.</p>}
      {pair && <Result pair={pair} onRefreshed={onRefreshed} mine={mine} />}
      {visitor && <SaveHint />}
    </main>
  );
}

function PairOrShare() {
  const share = useSearchParams().get("share");
  return share ? <SharedQuery token={share} /> : <PairQuery />;
}

export default function PairPage() {
  return (
    <Suspense>
      <PairOrShare />
    </Suspense>
  );
}

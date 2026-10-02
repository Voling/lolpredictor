"use client";

import Link from "next/link";
import { getHistory, type SavedCheck } from "@/lib/api";
import { AccountNotice, accountReady, useAccount } from "@/lib/account";
import { positionName } from "@/lib/positions";
import { RecentDuos } from "@/lib/recent";
import { useRemote } from "@/lib/remote";
import { Verdict } from "@/lib/verdict";

function title(check: SavedCheck): string {
  const names = check.names.map((name, index) => {
    const position = check.positions[index];
    return position ? `${name ?? "?"} as ${positionName(position)}` : name ?? "?";
  });
  if (check.kind === "pair") return names.join(" with ");
  return `${names[0]} with ${names.length - 1} ${names.length === 2 ? "friend" : "friends"}`;
}

const RECENT = 5;

function href(check: SavedCheck): string {
  return `/${check.kind === "pair" ? "pair" : "friends"}/?saved=${encodeURIComponent(check.id)}`;
}

export default function HistoryPage() {
  const account = useAccount();
  const ready = account.enabled && accountReady(account);
  const { data, error, loading } = useRemote(ready ? getHistory : null, `history|${ready}`);
  const recent = (data?.checks ?? []).flatMap((check) => (check.duos ?? []).map((duo) => ({ ...duo, href: href(check) }))).slice(0, RECENT);
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p className="sub">Your past checks. Open one to see it again without spending a check.</p>
      {!account.enabled && <div className="gate"><strong>Accounts are off here.</strong>They only work on the live site.</div>}
      <AccountNotice account={account} />
      {loading && !data && <p className="sub">Loading your checks…</p>}
      {error && <div className="gate"><strong>Can&apos;t load your checks.</strong>{error}</div>}
      {data && data.checks.length === 0 && <p>Nothing saved yet. Checks appear here after you check a duo or rank your friends.</p>}
      {recent.length > 0 && (
        <>
          <h3>Your recent predictions</h3>
          <RecentDuos duos={recent} />
        </>
      )}
      {data && data.checks.length > 0 && <h3>All checks</h3>}
      {data && data.checks.length > 0 && (
        <table>
          <thead>
            <tr><th>When</th><th>Check</th><th className="num">Score</th><th>Verdict</th></tr>
          </thead>
          <tbody>
            {data.checks.map((check) => (
              <tr key={check.id}>
                <td>{new Date(check.at * 1000).toLocaleDateString()}</td>
                <td><Link href={href(check)}>{title(check)}</Link></td>
                <td className="num">{check.score?.toFixed(0)}</td>
                <td>{check.score != null && <Verdict score={check.score} />}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {data && <p className="sub">For a friends check the score shown is the best friend&apos;s. Checks are kept for 90 days.</p>}
    </main>
  );
}

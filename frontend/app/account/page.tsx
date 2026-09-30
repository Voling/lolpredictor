"use client";

import Link from "next/link";
import { useEffect, useState, type FormEvent } from "react";
import { startLink, verifyLink, type Me } from "@/lib/api";
import { useAccount } from "@/lib/account";
import { signIn, signOut } from "@/lib/auth";

const ICONS = "https://ddragon.leagueoflegends.com/cdn/14.1.1/img/profileicon";
const CHECK_MS = 10_000;

export default function AccountPage() {
  const account = useAccount();
  const [riotId, setRiotId] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [fresh, setFresh] = useState<Me | null>(null);
  const [watching, setWatching] = useState(false);
  const me = fresh ?? account.me;

  useEffect(() => {
    if (!watching) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const check = async () => {
      try {
        const found = await verifyLink();
        if (stopped) return;
        setFresh(found);
        if (found.pending) timer = setTimeout(check, CHECK_MS);
        else setWatching(false);
      } catch (error) {
        if (stopped) return;
        setProblem(error instanceof Error ? error.message : String(error));
        setWatching(false);
      }
    };
    check();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [watching]);

  async function run(action: () => Promise<Me>) {
    setBusy(true);
    setProblem(null);
    try {
      setFresh(await action());
    } catch (error) {
      setProblem(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  function link(event: FormEvent) {
    event.preventDefault();
    setWatching(false);
    run(() => startLink(riotId));
  }

  function watch() {
    setProblem(null);
    setWatching(true);
  }

  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p className="sub">Your account</p>
      {!account.enabled && <div className="gate"><strong>Accounts are off here.</strong>They only work on the live site.</div>}
      {account.enabled && account.loading && <p className="sub">Checking your account…</p>}
      {account.enabled && !account.loading && !account.signedIn && (
        <>
          <p>Sign in or create an account to check duos. Each account gets a few duo checks a day for its own Riot account.</p>
          <p><button type="button" onClick={() => signIn("/account/")}>Sign in</button></p>
        </>
      )}
      {account.error && (
        <div className="gate">
          <strong>Can&apos;t load your account.</strong>
          {account.error}
          <p>{account.signedIn ? <button type="button" onClick={() => signOut()}>Sign out</button> : <button type="button" onClick={() => signIn("/account/")}>Sign in</button>}</p>
        </div>
      )}
      {me && (
        <>
          <h3>Riot account</h3>
          {me.verified && <p>Linked to {me.riot_id}.</p>}
          {me.pending && (
            <div className="link-check">
              <p>To prove {me.pending.riot_id} is yours, set your profile icon to this one in the League client within 15 minutes. Then press Verify.</p>
              <img src={`${ICONS}/${me.pending.icon}.png`} alt={`Profile icon ${me.pending.icon}`} width={64} height={64} />
              {watching && <p>Checking every 10 seconds. Riot can take a few minutes to show a new icon, so keep this page open.</p>}
              <p><button type="button" onClick={watch} disabled={busy || watching}>{watching ? "Checking…" : "Verify"}</button></p>
            </div>
          )}
          {!me.verified && !me.pending && <p>No Riot account linked yet.</p>}
          <form onSubmit={link} className="pair">
            <label>Your Riot ID <input value={riotId} onChange={(event) => setRiotId(event.target.value)} placeholder="name#tag" maxLength={40} required /></label>
            <button type="submit" disabled={busy}>{me.verified ? "Link a different account" : "Link"}</button>
          </form>
          <h3>Today</h3>
          <p>{me.remaining} of {me.daily} duo checks left. They reset at midnight UTC.</p>
          <p><button type="button" onClick={() => signOut()}>Sign out</button></p>
        </>
      )}
      {problem && <div className="gate"><strong>That didn&apos;t work.</strong>{problem}</div>}
    </main>
  );
}

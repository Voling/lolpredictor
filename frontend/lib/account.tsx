"use client";

import { useCallback, useEffect, useState } from "react";
import { ApiError, getMe, type Me } from "./api";
import { authEnabled, forget, signIn, signedIn } from "./auth";

const RIOT_ID = "lolpredictor.riot_id";

type AccountState = { me: Me | null; loading: boolean; error: string | null; signedIn: boolean };

export type Account = AccountState & { enabled: boolean; refresh: () => void };

export function useAccount(): Account {
  const [state, setState] = useState<AccountState>({ me: null, loading: authEnabled, error: null, signedIn: false });
  const refresh = useCallback(() => {
    if (!authEnabled) return;
    if (!signedIn()) {
      setState({ me: null, loading: false, error: null, signedIn: false });
      return;
    }
    setState((current) => ({ ...current, loading: current.me == null, signedIn: true }));
    getMe()
      .then((me) => setState({ me, loading: false, error: null, signedIn: true }))
      .catch((error) => {
        if (error instanceof ApiError && error.status === 401) forget();
        setState({ me: null, loading: false, error: error instanceof Error ? error.message : String(error), signedIn: signedIn() });
      });
  }, []);
  useEffect(refresh, [refresh]);
  return { ...state, enabled: authEnabled, refresh };
}

export function accountReady(account: Account): boolean {
  return !account.enabled || Boolean(account.me);
}

export function visiting(account: Account): boolean {
  return account.enabled && !account.loading && !account.signedIn;
}

export function typesRiotId(account: Account): boolean {
  return !account.enabled || visiting(account) || Boolean(account.me && !account.me.verified);
}

function rememberedRiotId(): string {
  try {
    return localStorage.getItem(RIOT_ID) ?? "";
  } catch {
    return "";
  }
}

function rememberRiotId(riotId: string) {
  try {
    localStorage.setItem(RIOT_ID, riotId);
  } catch {
    return;
  }
}

export function useRiotId(given?: string): string {
  const [remembered, setRemembered] = useState("");
  useEffect(() => {
    if (given) rememberRiotId(given);
    else setRemembered(rememberedRiotId());
  }, [given]);
  return given || remembered;
}

export function SaveHint() {
  return <p className="hint"><button type="button" className="link" onClick={() => signIn()}>Sign in</button> to save your checks.</p>;
}

export function AccountNotice({ account, remaining, named = true }: { account: Account; remaining?: number; named?: boolean }) {
  if (!account.enabled) return null;
  if (account.loading) return <p className="sub">Checking your account…</p>;
  if (!account.signedIn) {
    return (
      <div className="gate">
        <strong>Sign in to see your past checks.</strong>
        Your account saves each check.
        <p><button type="button" onClick={() => signIn()}>Sign in</button></p>
      </div>
    );
  }
  if (account.error || !account.me) return <div className="gate"><strong>Can&apos;t load your account.</strong>{account.error}</div>;
  const left = `${remaining ?? account.me.remaining} of ${account.me.daily} duo checks left today.`;
  return <p className="hint">{named && account.me.verified ? `Checking as ${account.me.riot_id}. ${left}` : left}</p>;
}

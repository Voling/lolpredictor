"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { ApiError, getMe, type Me } from "./api";
import { authEnabled, forget, signIn, signedIn } from "./auth";

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
  return !account.enabled || Boolean(account.me?.verified);
}

export function AccountNotice({ account, remaining }: { account: Account; remaining?: number }) {
  if (!account.enabled) return null;
  if (account.loading) return <p className="sub">Checking your account…</p>;
  if (!account.signedIn) {
    return (
      <div className="gate">
        <strong>Sign in to check duos.</strong>
        Each account gets a few duo checks a day for its own Riot account.
        <p><button type="button" onClick={() => signIn()}>Sign in</button></p>
      </div>
    );
  }
  if (account.error) return <div className="gate"><strong>Can&apos;t load your account.</strong>{account.error}</div>;
  if (!account.me?.verified) {
    return <div className="gate"><strong>Link your Riot account first.</strong><Link href="/account/">Go to your account</Link></div>;
  }
  return <p className="hint">Checking as {account.me.riot_id}. {remaining ?? account.me.remaining} of {account.me.daily} duo checks left today.</p>;
}

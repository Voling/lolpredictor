const DOMAIN = process.env.NEXT_PUBLIC_COGNITO_DOMAIN ?? "";
const CLIENT = process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID ?? "";
const TOKENS = "lolpredictor.tokens";
const PENDING = "lolpredictor.pending";

export const authEnabled = Boolean(DOMAIN && CLIENT);

type Tokens = { access: string; refresh: string | null; expires: number };
type Pending = { verifier: string; state: string; back: string };

function base64url(bytes: ArrayBuffer | Uint8Array): string {
  const array = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let text = "";
  array.forEach((byte) => {
    text += String.fromCharCode(byte);
  });
  return btoa(text).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function random(size: number): string {
  return base64url(crypto.getRandomValues(new Uint8Array(size)));
}

function redirectUri(): string {
  return `${window.location.origin}/auth/`;
}

function read<T>(storage: Storage, key: string): T | null {
  try {
    const text = storage.getItem(key);
    return text ? (JSON.parse(text) as T) : null;
  } catch {
    return null;
  }
}

function write(storage: Storage, key: string, value: unknown) {
  try {
    if (value == null) storage.removeItem(key);
    else storage.setItem(key, JSON.stringify(value));
  } catch {
    return;
  }
}

function safePath(path: string | undefined): string {
  return path && /^\/(?![\/\\])[^\\\s]*$/.test(path) ? path : "/";
}

export async function signIn(back?: string) {
  const verifier = random(48);
  const state = random(24);
  const challenge = base64url(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)));
  write(sessionStorage, PENDING, { verifier, state, back: safePath(back ?? window.location.pathname + window.location.search) });
  const query = new URLSearchParams({
    response_type: "code",
    client_id: CLIENT,
    redirect_uri: redirectUri(),
    scope: "openid email",
    state,
    code_challenge: challenge,
    code_challenge_method: "S256",
  });
  window.location.assign(`${DOMAIN}/oauth2/authorize?${query}`);
}

class Rejected extends Error {}

async function tokenRequest(body: Record<string, string>): Promise<Tokens> {
  const response = await fetch(`${DOMAIN}/oauth2/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ client_id: CLIENT, ...body }),
  });
  if (response.status === 400) throw new Rejected("Sign in failed. Try again.");
  if (!response.ok) throw new Error("Sign in failed. Try again.");
  const found = await response.json();
  return {
    access: found.access_token,
    refresh: found.refresh_token ?? body.refresh_token ?? null,
    expires: Date.now() + (Number(found.expires_in) - 60) * 1000,
  };
}

export async function finishSignIn(params: URLSearchParams): Promise<string> {
  const pending = read<Pending>(sessionStorage, PENDING);
  write(sessionStorage, PENDING, null);
  const code = params.get("code");
  if (!pending || !code || params.get("state") !== pending.state) throw new Error("Sign in failed. Try again.");
  write(localStorage, TOKENS, await tokenRequest({ grant_type: "authorization_code", code, redirect_uri: redirectUri(), code_verifier: pending.verifier }));
  return safePath(pending.back);
}

export async function accessToken(): Promise<string | null> {
  if (!authEnabled) return null;
  const tokens = read<Tokens>(localStorage, TOKENS);
  if (!tokens) return null;
  if (tokens.expires > Date.now()) return tokens.access;
  if (!tokens.refresh) {
    write(localStorage, TOKENS, null);
    return null;
  }
  try {
    const fresh = await tokenRequest({ grant_type: "refresh_token", refresh_token: tokens.refresh });
    write(localStorage, TOKENS, fresh);
    return fresh.access;
  } catch (error) {
    if (error instanceof Rejected) write(localStorage, TOKENS, null);
    return null;
  }
}

export function forget() {
  write(localStorage, TOKENS, null);
}

export function signedIn(): boolean {
  return authEnabled && read<Tokens>(localStorage, TOKENS) != null;
}

export async function signOut() {
  const tokens = read<Tokens>(localStorage, TOKENS);
  write(localStorage, TOKENS, null);
  if (tokens?.refresh) {
    await fetch(`${DOMAIN}/oauth2/revoke`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ client_id: CLIENT, token: tokens.refresh }),
    }).catch(() => undefined);
  }
  const query = new URLSearchParams({ client_id: CLIENT, logout_uri: `${window.location.origin}/` });
  window.location.assign(`${DOMAIN}/logout?${query}`);
}

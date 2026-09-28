const REGION = process.env.NEXT_PUBLIC_COGNITO_REGION ?? "";
const CLIENT = process.env.NEXT_PUBLIC_COGNITO_CLIENT_ID ?? "";
const TOKENS = "lolpredictor.tokens";
const BUSY = "Too many tries. Wait a few minutes and try again.";

export const authEnabled = Boolean(REGION && CLIENT);

type Tokens = { access: string; refresh: string | null; expires: number };
type Issued = { AccessToken: string; RefreshToken?: string; ExpiresIn: number };
type Authenticated = { AuthenticationResult?: Issued };

const MESSAGES: Record<string, string> = {
  UsernameExistsException: "An account with this email already exists. Sign in instead.",
  InvalidPasswordException: "Use at least 12 characters with a lowercase letter and a number.",
  CodeMismatchException: "That code doesn't match. Check the email we sent.",
  ExpiredCodeException: "That code has expired. Send a new one.",
  LimitExceededException: BUSY,
  TooManyRequestsException: BUSY,
  TooManyFailedAttemptsException: BUSY,
  ForbiddenException: "We can't accept sign ins from your network. Try another connection.",
  CodeDeliveryFailureException: "We couldn't send the email. Try again later.",
};

export class SignInError extends Error {
  constructor(readonly kind: string, message: string) {
    super(message);
  }
}

function explain(kind: string, message: string): string {
  if (kind === "NotAuthorizedException") {
    if (/attempts exceeded/i.test(message)) return BUSY;
    if (/current status is confirmed/i.test(message)) return "This email is already confirmed. Sign in.";
    return "Wrong email or password.";
  }
  if (kind === "UserLambdaValidationException") return message.replace(/^\w+ failed with error /, "");
  return MESSAGES[kind] ?? (message || "Something went wrong. Try again.");
}

async function call<T>(operation: string, body: object): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`https://cognito-idp.${REGION}.amazonaws.com/`, {
      method: "POST",
      headers: { "Content-Type": "application/x-amz-json-1.1", "X-Amz-Target": `AWSCognitoIdentityProviderService.${operation}` },
      body: JSON.stringify({ ClientId: CLIENT, ...body }),
    });
  } catch {
    throw new SignInError("Network", "We can't reach the sign in service. Check your connection.");
  }
  const text = await response.text();
  let found: { __type?: string; message?: string } = {};
  try {
    found = text ? JSON.parse(text) : {};
  } catch {
    found = {};
  }
  if (response.ok) return found as T;
  const fallback = response.status === 429 ? "TooManyRequestsException" : response.status === 403 ? "ForbiddenException" : "Unknown";
  const kind = (found.__type ?? fallback).split("#").pop() ?? fallback;
  throw new SignInError(kind, explain(kind, found.message ?? ""));
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

function keep(issued: Issued, refresh: string | null) {
  write(localStorage, TOKENS, { access: issued.AccessToken, refresh: issued.RefreshToken ?? refresh, expires: Date.now() + (issued.ExpiresIn - 60) * 1000 });
}

function username(email: string): string {
  return email.trim().toLowerCase();
}

export function safePath(path: string | null | undefined): string {
  return path && /^\/(?![\/\\])[^\\\s]*$/.test(path) ? path : "/";
}

export function signIn(back?: string) {
  const next = safePath(back ?? window.location.pathname + window.location.search);
  window.location.assign(`/signin/?${new URLSearchParams({ next })}`);
}

export async function passwordSignIn(email: string, password: string) {
  const found = await call<Authenticated>("InitiateAuth", { AuthFlow: "USER_PASSWORD_AUTH", AuthParameters: { USERNAME: username(email), PASSWORD: password } });
  if (!found.AuthenticationResult) throw new SignInError("Challenge", "This account needs a new password. Use Forgot your password.");
  keep(found.AuthenticationResult, null);
}

export async function signUp(email: string, password: string) {
  await call("SignUp", { Username: username(email), Password: password, UserAttributes: [{ Name: "email", Value: username(email) }] });
}

export async function confirmSignUp(email: string, code: string) {
  await call("ConfirmSignUp", { Username: username(email), ConfirmationCode: code.trim() });
}

export async function resendCode(email: string) {
  await call("ResendConfirmationCode", { Username: username(email) });
}

export async function forgotPassword(email: string) {
  await call("ForgotPassword", { Username: username(email) });
}

export async function resetPassword(email: string, code: string, password: string) {
  await call("ConfirmForgotPassword", { Username: username(email), ConfirmationCode: code.trim(), Password: password });
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
    const found = await call<Authenticated>("InitiateAuth", { AuthFlow: "REFRESH_TOKEN_AUTH", AuthParameters: { REFRESH_TOKEN: tokens.refresh } });
    if (!found.AuthenticationResult) return null;
    keep(found.AuthenticationResult, tokens.refresh);
    return found.AuthenticationResult.AccessToken;
  } catch (error) {
    if (error instanceof SignInError && error.kind === "NotAuthorizedException") write(localStorage, TOKENS, null);
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
  if (tokens?.refresh) await call("RevokeToken", { Token: tokens.refresh }).catch(() => undefined);
  window.location.assign("/");
}

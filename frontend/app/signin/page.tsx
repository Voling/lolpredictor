"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState, type FormEvent } from "react";
import { authEnabled, confirmSignUp, forgotPassword, passwordSignIn, resendCode, resetPassword, safePath, SignInError, signedIn, signUp } from "@/lib/auth";

type Mode = "signin" | "signup" | "confirm" | "forgot" | "reset";

const TITLES: Record<Mode, string> = {
  signin: "Sign in",
  signup: "Create an account",
  confirm: "Confirm your email",
  forgot: "Reset your password",
  reset: "Set a new password",
};

const BUTTONS: Record<Mode, string> = {
  signin: "Sign in",
  signup: "Create account",
  confirm: "Confirm",
  forgot: "Send a code",
  reset: "Set password",
};

const PASSWORD_RULE = "At least 12 characters with a lowercase letter and a number.";

function SignInForm() {
  const params = useSearchParams();
  const router = useRouter();
  const next = safePath(params.get("next") ?? "/account/");
  const [mode, setMode] = useState<Mode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (signedIn()) router.replace(next);
  }, [next, router]);

  function go(target: Mode, message: string | null = null) {
    setMode(target);
    setProblem(null);
    setNotice(message);
    setCode("");
  }

  async function run(action: () => Promise<void>) {
    setBusy(true);
    setProblem(null);
    try {
      await action();
    } catch (error) {
      if (error instanceof SignInError && error.kind === "UserNotConfirmedException") {
        await resendCode(email).catch(() => undefined);
        go("confirm", `Confirm your email first. We sent a new code to ${email}.`);
      } else {
        setProblem(error instanceof Error ? error.message : String(error));
      }
    } finally {
      setBusy(false);
    }
  }

  async function finish() {
    await passwordSignIn(email, password);
    router.replace(next);
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (mode === "signin") run(finish);
    if (mode === "signup") {
      run(async () => {
        await signUp(email, password);
        go("confirm", `We sent a code to ${email}. Enter it below.`);
      });
    }
    if (mode === "confirm") {
      run(async () => {
        await confirmSignUp(email, code);
        if (password) await finish();
        else go("signin", "Your email is confirmed. Sign in.");
      });
    }
    if (mode === "forgot") {
      run(async () => {
        await forgotPassword(email);
        setPassword("");
        go("reset", `If ${email} has an account, we sent it a code.`);
      });
    }
    if (mode === "reset") {
      run(async () => {
        await resetPassword(email, code, password);
        await finish();
      });
    }
  }

  function resend() {
    run(async () => {
      await resendCode(email);
      setNotice(`We sent a new code to ${email}.`);
    });
  }

  const asksEmail = mode === "signin" || mode === "signup" || mode === "forgot";
  const asksCode = mode === "confirm" || mode === "reset";
  const asksPassword = mode === "signin" || mode === "signup" || mode === "reset";

  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p className="sub">{TITLES[mode]}</p>
      {!authEnabled && <div className="gate"><strong>Accounts are off here.</strong>They only work on the live site.</div>}
      {authEnabled && (
        <>
          {notice && <p>{notice}</p>}
          <form onSubmit={submit} className="pair stack">
            {asksEmail && (
              <label>Email <input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} maxLength={254} required /></label>
            )}
            {asksCode && (
              <label>Code <input inputMode="numeric" autoComplete="one-time-code" value={code} onChange={(event) => setCode(event.target.value)} maxLength={10} required /></label>
            )}
            {asksPassword && (
              <label>
                {mode === "reset" ? "New password" : "Password"}
                <input
                  type="password"
                  autoComplete={mode === "signin" ? "current-password" : "new-password"}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  minLength={mode === "signin" ? undefined : 12}
                  maxLength={256}
                  required
                />
                {mode !== "signin" && <small>{PASSWORD_RULE}</small>}
              </label>
            )}
            <button type="submit" disabled={busy}>{BUTTONS[mode]}</button>
          </form>
          {problem && <div className="gate"><strong>That didn&apos;t work.</strong>{problem}</div>}
          {mode === "signin" && (
            <>
              <p>New here? <button type="button" className="link" onClick={() => go("signup")}>Create an account</button></p>
              <p><button type="button" className="link" onClick={() => go("forgot")}>Forgot your password?</button></p>
            </>
          )}
          {mode === "signup" && <p>Have an account? <button type="button" className="link" onClick={() => go("signin")}>Sign in</button></p>}
          {mode === "confirm" && <p>No code? <button type="button" className="link" onClick={resend} disabled={busy}>Send a new one</button></p>}
          {(mode === "forgot" || mode === "reset") && <p><button type="button" className="link" onClick={() => go("signin")}>Back to sign in</button></p>}
        </>
      )}
    </main>
  );
}

export default function SignInPage() {
  return (
    <Suspense>
      <SignInForm />
    </Suspense>
  );
}

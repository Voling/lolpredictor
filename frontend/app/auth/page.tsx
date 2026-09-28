"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import { finishSignIn } from "@/lib/auth";

function Callback() {
  const params = useSearchParams();
  const router = useRouter();
  const started = useRef(false);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    finishSignIn(new URLSearchParams(params.toString()))
      .then((back) => router.replace(back))
      .catch((error) => setProblem(error instanceof Error ? error.message : String(error)));
  }, [params, router]);
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      {problem ? <div className="gate"><strong>Sign in failed.</strong>{problem}</div> : <p className="sub">Signing you in…</p>}
    </main>
  );
}

export default function AuthPage() {
  return (
    <Suspense>
      <Callback />
    </Suspense>
  );
}

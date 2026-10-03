"use client";

import { useState } from "react";

export function shareUrl(token: string): string {
  return `${window.location.origin}/pair/?share=${encodeURIComponent(token)}`;
}

export function ShareLink({ token }: { token: string }) {
  const [copied, setCopied] = useState(false);
  const url = shareUrl(token);

  async function copy() {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="share">
      <label>
        Share this result
        <input readOnly value={url} onFocus={(event) => event.currentTarget.select()} />
      </label>
      <button type="button" onClick={copy}>{copied ? "Copied" : "Copy link"}</button>
      <p className="sub">Anyone with the link sees this result. It expires when either of you pulls new games.</p>
    </div>
  );
}

"use client";

import Link from "next/link";
import { useEffect } from "react";

export default function MovedToDuo() {
  useEffect(() => {
    window.location.replace(`/duo/${window.location.search}`);
  }, []);
  return (
    <main>
      <h1><Link href="/">lolpredictor</Link></h1>
      <p>Duo checks now live at <Link href="/duo/">/duo</Link>.</p>
    </main>
  );
}

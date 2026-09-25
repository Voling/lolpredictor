"use client";

import { useEffect, useState } from "react";

export function useRemote<T>(load: (() => Promise<T>) | null, key: string) {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({
    data: null,
    error: null,
    loading: false,
  });
  useEffect(() => {
    if (!load) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    let live = true;
    setState({ data: null, error: null, loading: true });
    load()
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch((error) => live && setState({ data: null, error: String(error), loading: false }));
    return () => {
      live = false;
    };
  }, [key]);
  return state;
}

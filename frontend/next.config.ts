import type { NextConfig } from "next";

const exporting = process.env.STATIC_EXPORT === "1";

const config: NextConfig = exporting
  ? { output: "export", trailingSlash: true }
  : {
      async rewrites() {
        return [
          {
            source: "/api/:path*",
            destination: `${process.env.API_URL ?? "http://localhost:8000"}/api/:path*`,
          },
        ];
      },
    };

export default config;

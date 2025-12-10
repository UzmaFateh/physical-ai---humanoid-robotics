// src/auth/config.js
import { createAuthClient } from "better-auth/client";

export const authClient = createAuthClient({
  baseURL: process.env.NEXT_PUBLIC_BETTER_AUTH_URL || "http://localhost:3000", // This would be your Vercel deployment URL
  fetchOptions: {
    // You can add custom fetch options here
  }
});
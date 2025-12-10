import { betterAuth } from "better-auth";
import { drizzleAdapter } from "@better-auth/drizzle-adapter";
import { db } from "../db"; // Assuming you have a DB setup

export const auth = betterAuth({
  database: drizzleAdapter(db, {
    provider: "pg", // Using PostgreSQL as in your requirements.txt
  }),
  secret: process.env.AUTH_SECRET || "your-super-secret-key-change-in-production",
  baseURL: process.env.AUTH_BASE_URL || "http://localhost:3000",
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false,
    async sendVerificationEmail(user, url) {
      // In a real app, you'd send an email here
      console.log(`Verification email for ${user.email}: ${url}`);
    },
  },
  socialProviders: {
    // Add social providers if needed
  },
  user: {
    // Add custom user fields for background information
    additionalFields: {
      softwareBackground: {
        type: "string",
        required: false,
      },
      hardwareBackground: {
        type: "string",
        required: false,
      },
      experienceLevel: {
        type: "string",
        required: false,
      },
    },
  },
});
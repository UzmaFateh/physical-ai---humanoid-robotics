import { betterAuth } from "better-auth";

export const auth = betterAuth({
  secret: process.env.AUTH_SECRET || "your-super-secret-key-change-in-production",
  baseURL: process.env.AUTH_BASE_URL || "http://localhost:3000",
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false,
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
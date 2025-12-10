// This is a placeholder file for Better Auth API routes that would run on Vercel
// In a real implementation, you would have these files in an `api` directory for Vercel deployment

// pages/api/auth/[...auth].js for Next.js API routes
import { auth } from "better-auth";

export default auth({
  secret: process.env.AUTH_SECRET,
  database: {
    provider: "postgresql", // or your preferred database
    url: process.env.DATABASE_URL,
  },
  emailAndPassword: {
    enabled: true,
  },
  user: {
    // Custom fields for user background
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

export const config = {
  api: {
    externalResolver: true,
  },
};
---
id: 1
title: Integrated RAG Chatbot Constitution Update
stage: constitution
date: 2025-12-09
surface: agent
model: claude-sonnet-4-5-20251101
feature: Integrated RAG Chatbot
branch: main
user: TODO(USER): Ask user for username
command: /sp.constitution
labels: ["constitution", "rag-chatbot"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests:

---

## Prompt

```text
You are an AI system responsible for generating the full technical plan, architecture, code, and integration steps for Part-2 of this project: creating an Integrated RAG Chatbot embedded inside a Docusaurus-based book. The chatbot must use:

- OpenAI Agents / ChatKit SDK (but API key must be Gemini API key inserted in OpenAI-style structure)
- FastAPI backend
- Qdrant Cloud Free Tier for vector storage
- Neon Serverless Postgres database for user sessions
- Text-selection-based context (RAG)
- Frontend embedding inside Docusaurus

You must always produce output that is production-ready, minimal, clean, and follows proper folder structure. API calls must use the OpenAI client format but the key will be the Gemini key provided by the user.

Your responsibilities:
- Avoid hallucinations
- Build the cleanest possible agent plan
- Follow all Specify Kit stages faithfully
```

## Response snapshot

Constitution for Integrated RAG Chatbot updated with core principles, technology stack, responsibilities, and governance rules. Version 1.0.0 ratified.

## Outcome

- ✅ Impact: Constitution successfully updated.
- 🧪 Tests: N/A
- 📁 Files: .specify/memory/constitution.md modified.
- 🔁 Next prompts: Proceed with next Specify Kit stage.
- 🧠 Reflection: Constitution created successfully based on user's prompt.

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A

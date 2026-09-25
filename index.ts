import { generateText } from "ai";

// AI_GATEWAY_API_KEY is read from the environment (loaded from .env.local by `npm run example`).
const { text } = await generateText({
  model: "openai/gpt-5.5",
  prompt: "Invent a new holiday and describe its traditions.",
});

console.log(text);

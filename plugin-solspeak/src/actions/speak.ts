/**
 * SPEAK Action - Generate speech from text using SolSpeak TTS
 */

import type {
  Action,
  ActionExample,
  ActionResult,
  HandlerCallback,
  HandlerOptions,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { ContentType } from "@elizaos/core";
import { SolSpeakService } from "../services/tts-service.js";
import { checkTokenAccess } from "../services/token-gate.js";

/**
 * Extract text content, stripping common TTS command prefixes
 */
function extractText(content: string): string {
  const patterns = [
    /^(say|speak|read|voice|tts|generate\s*tts|create\s*tts|make\s*tts)\s*:?\s*/i,
    /^(read\s*aloud|speak\s*this|say\s*this)\s*:?\s*/i,
  ];

  let text = content.trim();
  for (const pattern of patterns) {
    text = text.replace(pattern, "");
  }

  return text.trim();
}

export const speakAction: Action = {
  name: "SPEAK",
  similes: [
    "SAY",
    "VOICE",
    "TTS",
    "TEXT_TO_SPEECH",
    "READ_ALOUD",
    "GENERATE_TTS",
    "CREATE_TTS",
  ],
  description:
    "Convert text to speech using SolSpeak TTS with turbo model. " +
    "Supports paralinguistic tags: [laugh], [chuckle], [sigh], [cough], [gasp], [groan], [yawn], [clearing throat]. " +
    "Example: 'That's hilarious! [laugh] I can't believe it.'",

  validate: async (runtime: IAgentRuntime, message: Memory): Promise<boolean> => {
    const voice = runtime.getSetting("SOLSPEAK_VOICE");
    if (!voice || typeof voice !== "string") {
      console.warn("[SolSpeak] SOLSPEAK_VOICE not configured");
      return false;
    }

    const text = message.content?.text;
    if (!text || text.length < 2) {
      return false;
    }

    return true;
  },

  handler: async (
    runtime: IAgentRuntime,
    message: Memory,
    state?: State,
    options?: HandlerOptions,
    callback?: HandlerCallback
  ): Promise<ActionResult> => {
    // =========================================================================
    // TOKEN GATING (Optional)
    // Enable by setting SOLSPEAK_TOKEN_GATE=true in character settings
    // Also set: SOLSPEAK_TOKEN_MINT, SOLSPEAK_TOKEN_MIN_BALANCE
    // =========================================================================
    // Get wallet from message metadata or state (depends on your elizaOS setup)
    // const walletAddress = (message.content as any)?.wallet
    //   || (state as any)?.walletAddress
    //   || runtime.getSetting("SOLSPEAK_DEFAULT_WALLET");
    //
    // const access = await checkTokenAccess(runtime, walletAddress as string);
    // if (!access.allowed) {
    //   if (callback) {
    //     await callback({
    //       text: access.message || "Token-gated: You need to hold tokens to use voice features.",
    //     });
    //   }
    //   return { success: false, error: access.message || "Token gate denied" };
    // }
    // =========================================================================

    const service = new SolSpeakService(runtime);

    const rawText = message.content?.text || "";
    const textToSpeak = extractText(rawText);

    if (textToSpeak.length < 2) {
      if (callback) {
        await callback({
          text: "Please provide text to convert to speech.",
        });
      }
      return { success: false, error: "Text too short" };
    }

    // Check for voice override in message
    const voiceMatch = rawText.match(/voice[=:]\s*["']?(\w+)["']?/i);
    const voiceOverride = voiceMatch ? voiceMatch[1] : undefined;

    // Check for exaggeration override
    const exagMatch = rawText.match(/exaggeration[=:]\s*([\d.]+)/i);
    const exaggeration = exagMatch ? parseFloat(exagMatch[1]) : undefined;

    try {
      const result = await service.speak({
        text: textToSpeak,
        voice_name: voiceOverride,
        exaggeration,
      });

      if (result.status === "error") {
        if (callback) {
          await callback({
            text: `Speech generation failed: ${result.message}`,
          });
        }
        return { success: false, error: result.message };
      }

      // Success - return the audio URL
      if (callback) {
        await callback({
          text: `Generated speech for: "${textToSpeak.substring(0, 50)}${textToSpeak.length > 50 ? "..." : ""}"`,
          attachments: [
            {
              id: result.filename,
              url: result.download_url,
              contentType: ContentType.AUDIO,
              title: "Generated Speech",
              source: "solspeak-tts",
              description: `TTS audio (${Math.round(result.size_bytes / 1024)}KB)`,
            },
          ],
        });
      }

      return {
        success: true,
        data: {
          filename: result.filename,
          download_url: result.download_url,
          size_bytes: result.size_bytes,
        },
      };
    } catch (error) {
      console.error("[SolSpeak] Error generating speech:", error);
      if (callback) {
        await callback({
          text: `Failed to generate speech: ${error instanceof Error ? error.message : String(error)}`,
        });
      }
      return {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
  },

  examples: [
    [
      {
        name: "{{user1}}",
        content: { text: "Say hello world" },
      },
      {
        name: "{{agent}}",
        content: {
          text: "Generated speech for: \"hello world\"",
          actions: ["SPEAK"],
        },
      },
    ],
    [
      {
        name: "{{user1}}",
        content: { text: "TTS: That's amazing! [laugh] I love it!" },
      },
      {
        name: "{{agent}}",
        content: {
          text: "Generated speech for: \"That's amazing! [laugh] I love it!\"",
          actions: ["SPEAK"],
        },
      },
    ],
    [
      {
        name: "{{user1}}",
        content: { text: "Generate TTS of Welcome to the future of AI agents" },
      },
      {
        name: "{{agent}}",
        content: {
          text: "Generated speech for: \"Welcome to the future of AI agents\"",
          actions: ["SPEAK"],
        },
      },
    ],
  ] as ActionExample[][],
};

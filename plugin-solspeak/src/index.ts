/**
 * @elizaos/plugin-solspeak
 *
 * SolSpeak TTS Plugin for ElizaOS
 * Give your agents a voice with the turbo model optimized for voice agents.
 *
 * Features:
 * - Fast speech generation with turbo model
 * - Voice cloning support
 * - Paralinguistic tags: [laugh], [sigh], [cough], etc.
 *
 * Configuration:
 * - SOLSPEAK_URL: API endpoint (default: https://mcp.thethirdroom.xyz)
 * - SOLSPEAK_VOICE: Default voice name (REQUIRED)
 * - SOLSPEAK_EXAGGERATION: Expressiveness 0-1 (default: 0.5)
 * - SOLSPEAK_CFG_WEIGHT: Speech pace 0-1 (default: 0.5)
 */

import type { Plugin } from "@elizaos/core";
import { speakAction } from "./actions/speak.js";

// Re-export types and services for advanced usage
export * from "./types.js";
export { SolSpeakService } from "./services/tts-service.js";
export {
  TokenGateService,
  checkTokenAccess,
  checkVerifiedTokenAccess,
  generateChallenge,
  verifyWalletOwnership,
  getVerifiedWallet,
  SOLSPEAK_TOKEN_MINT,
  SOLSPEAK_TOKEN_DECIMALS,
  DEFAULT_MIN_BALANCE,
} from "./services/token-gate.js";
export { speakAction } from "./actions/speak.js";

/**
 * SolSpeak TTS Plugin
 *
 * @example
 * ```typescript
 * import { solspeakPlugin } from "@elizaos/plugin-solspeak";
 *
 * const character = {
 *   name: "MyAgent",
 *   plugins: [solspeakPlugin],
 *   settings: {
 *     SOLSPEAK_VOICE: "my-cloned-voice",
 *   },
 * };
 * ```
 */
export const solspeakPlugin: Plugin = {
  name: "solspeak-tts",
  description:
    "SolSpeak TTS plugin for ElizaOS. " +
    "Generate speech with voice cloning and paralinguistic tags. " +
    "Turbo model optimized for real-time voice agents.",
  actions: [speakAction],
  evaluators: [],
  providers: [],
};

// Default export for convenience
export default solspeakPlugin;

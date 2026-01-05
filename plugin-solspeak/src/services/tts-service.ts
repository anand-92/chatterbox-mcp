/**
 * SolSpeak TTS Service - Turbo Model
 * Optimized for voice agents with paralinguistic tag support
 */

import type { IAgentRuntime } from "@elizaos/core";
import type { TTSOptions, TTSResponse, TTSErrorResponse, ListVoicesResponse } from "../types.js";

const DEFAULT_API_URL = "https://mcp.thethirdroom.xyz";

export class SolSpeakService {
  private apiUrl: string;
  private defaultVoice: string;
  private defaultExaggeration: number;
  private defaultCfgWeight: number;

  constructor(runtime: IAgentRuntime) {
    const url = runtime.getSetting("SOLSPEAK_URL");
    const voice = runtime.getSetting("SOLSPEAK_VOICE");
    const exag = runtime.getSetting("SOLSPEAK_EXAGGERATION");
    const cfg = runtime.getSetting("SOLSPEAK_CFG_WEIGHT");

    this.apiUrl = typeof url === "string" ? url : DEFAULT_API_URL;
    this.defaultVoice = typeof voice === "string" ? voice : "";
    this.defaultExaggeration = parseFloat(typeof exag === "string" ? exag : "0.5");
    this.defaultCfgWeight = parseFloat(typeof cfg === "string" ? cfg : "0.5");
  }

  /**
   * Generate speech from text using turbo model
   */
  async speak(options: Partial<TTSOptions> & { text: string }): Promise<TTSResponse | TTSErrorResponse> {
    const voice = options.voice_name || this.defaultVoice;

    if (!voice) {
      return {
        status: "error",
        message: "Voice name is required for turbo model. Set SOLSPEAK_VOICE in settings.",
      };
    }

    const payload = {
      text: options.text,
      model: "turbo",
      voice_name: voice,
      exaggeration: options.exaggeration ?? this.defaultExaggeration,
      cfg_weight: options.cfg_weight ?? this.defaultCfgWeight,
    };

    try {
      const response = await fetch(`${this.apiUrl}/api/tts`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const result = await response.json();
      return result as TTSResponse | TTSErrorResponse;
    } catch (error) {
      return {
        status: "error",
        message: `Failed to generate speech: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  /**
   * List available voices
   */
  async listVoices(): Promise<ListVoicesResponse | TTSErrorResponse> {
    try {
      const response = await fetch(`${this.apiUrl}/api/voices`);
      return await response.json();
    } catch (error) {
      return {
        status: "error",
        message: `Failed to list voices: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }

  /**
   * Download audio file and return as Buffer
   */
  async downloadAudio(url: string): Promise<Buffer> {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
  }

  /**
   * Check if service is configured correctly
   */
  isConfigured(): boolean {
    return !!this.defaultVoice;
  }

  /**
   * Get the API URL
   */
  getApiUrl(): string {
    return this.apiUrl;
  }

  /**
   * Get the default voice
   */
  getDefaultVoice(): string {
    return this.defaultVoice;
  }
}

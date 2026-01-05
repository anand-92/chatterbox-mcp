/**
 * SolSpeak TTS Plugin Types - Turbo Model Only
 */

/** Paralinguistic tags for turbo model */
export type ParalinguisticTag =
  | "[laugh]"
  | "[chuckle]"
  | "[cough]"
  | "[sigh]"
  | "[gasp]"
  | "[groan]"
  | "[yawn]"
  | "[clearing throat]";

/** TTS generation request options */
export interface TTSOptions {
  /** Text to convert to speech (supports paralinguistic tags) */
  text: string;
  /** Name of a saved voice for cloning (REQUIRED for turbo) */
  voice_name: string;
  /** Expressiveness level 0.0-1.0 (default: 0.5) */
  exaggeration?: number;
  /** CFG weight 0.0-1.0, lower = slower/deliberate (default: 0.5) */
  cfg_weight?: number;
}

/** Successful TTS response */
export interface TTSResponse {
  status: "success";
  filename: string;
  download_url: string;
  size_bytes: number;
  message?: string;
}

/** TTS error response */
export interface TTSErrorResponse {
  status: "error";
  message: string;
}

/** Voice info from list_voices */
export interface VoiceInfo {
  filename: string;
  size_bytes: number;
  size_mb: number;
}

/** List voices response */
export interface ListVoicesResponse {
  voices_directory: string;
  available_voices: Record<string, VoiceInfo>;
  count: number;
  usage: string;
}

/** Plugin configuration settings */
export interface SolSpeakSettings {
  /** SolSpeak API URL (default: https://mcp.thethirdroom.xyz) */
  SOLSPEAK_URL?: string;
  /** Default voice name to use (REQUIRED) */
  SOLSPEAK_VOICE?: string;
  /** Default exaggeration level */
  SOLSPEAK_EXAGGERATION?: number;
  /** Default CFG weight */
  SOLSPEAK_CFG_WEIGHT?: number;
}

/** Paralinguistic tags list */
export const PARALINGUISTIC_TAGS: ParalinguisticTag[] = [
  "[laugh]",
  "[chuckle]",
  "[cough]",
  "[sigh]",
  "[gasp]",
  "[groan]",
  "[yawn]",
  "[clearing throat]",
];

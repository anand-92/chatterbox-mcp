/**
 * Quick test for SolSpeak plugin
 * Run: node test.mjs
 */

import { SolSpeakService } from "./dist/index.js";

// Mock runtime with settings
const mockRuntime = {
  getSetting: (key) => {
    const settings = {
      SOLSPEAK_URL: "https://mcp.thethirdroom.xyz",
      SOLSPEAK_VOICE: "elon",
      SOLSPEAK_EXAGGERATION: "0.6",
      SOLSPEAK_CFG_WEIGHT: "0.5",
    };
    return settings[key];
  },
};

async function test() {
  console.log("Testing SolSpeak TTS Plugin...\n");

  const service = new SolSpeakService(mockRuntime);

  // Test 1: List voices
  console.log("1. Listing available voices...");
  const voices = await service.listVoices();
  if (voices.status === "error") {
    console.error("   Failed:", voices.message);
  } else {
    console.log(`   Found ${voices.count} voices:`, Object.keys(voices.available_voices).join(", "));
  }

  // Test 2: Generate speech
  console.log("\n2. Generating speech with turbo model...");
  const result = await service.speak({
    text: "Hello! Welcome to ElizaOS with SolSpeak TTS. [laugh] This is pretty cool!",
  });

  if (result.status === "error") {
    console.error("   Failed:", result.message);
  } else {
    console.log("   Success!");
    console.log("   Filename:", result.filename);
    console.log("   Download URL:", result.download_url);
    console.log("   Size:", Math.round(result.size_bytes / 1024), "KB");
  }

  // Test 3: Test with different voice
  console.log("\n3. Testing voice override (obama)...");
  const result2 = await service.speak({
    text: "Let me be clear. This is a test of the voice cloning system.",
    voice_name: "obama",
    exaggeration: 0.4,
  });

  if (result2.status === "error") {
    console.error("   Failed:", result2.message);
  } else {
    console.log("   Success!");
    console.log("   Download URL:", result2.download_url);
  }

  console.log("\n✅ All tests completed!");
}

test().catch(console.error);

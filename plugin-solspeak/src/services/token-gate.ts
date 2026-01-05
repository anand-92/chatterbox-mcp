/**
 * Solana Token Gating for SolSpeak Plugin
 *
 * Requires users to:
 * 1. Prove wallet ownership via signature
 * 2. Hold 1M+ SolSpeak tokens
 *
 * Configuration (in character settings):
 * - SOLSPEAK_TOKEN_GATE: "true" to enable (default: disabled)
 * - SOLSPEAK_TOKEN_MIN_BALANCE: Minimum balance required (default: 1000000)
 * - SOLSPEAK_SOLANA_RPC: RPC endpoint (default: mainnet)
 */

import type { IAgentRuntime } from "@elizaos/core";

// ==============================================================================
// SOLSPEAK TOKEN - HARDCODED
// ==============================================================================

/**
 * SolSpeak Token Mint Address (Solana SPL Token)
 * Update this when the token is deployed
 */
export const SOLSPEAK_TOKEN_MINT = "PLACEHOLDER_SOLSPEAK_TOKEN_MINT_ADDRESS";

/**
 * Token decimals (most SPL tokens use 9)
 */
export const SOLSPEAK_TOKEN_DECIMALS = 9;

/**
 * Default minimum balance: 1 million tokens
 */
export const DEFAULT_MIN_BALANCE = 1_000_000;

// ==============================================================================
// CONFIGURATION
// ==============================================================================

const DEFAULT_RPC = "https://api.mainnet-beta.solana.com";
const CACHE_DURATION_MS = 60_000; // 1 minute cache
const CHALLENGE_EXPIRY_MS = 5 * 60_000; // 5 minute challenge validity

// Caches
const balanceCache = new Map<string, { balance: number; timestamp: number }>();
const verifiedWallets = new Map<string, { wallet: string; timestamp: number }>();
const pendingChallenges = new Map<string, { challenge: string; timestamp: number }>();

// ==============================================================================
// WALLET VERIFICATION
// ==============================================================================

/**
 * Generate a challenge message for wallet signature verification
 */
export function generateChallenge(userId: string): { challenge: string; message: string } {
  const timestamp = Date.now();
  const nonce = Math.random().toString(36).substring(2, 15);
  const challenge = `${userId}:${timestamp}:${nonce}`;

  // Human-readable message for wallet to sign
  const message = `SolSpeak Voice Access\n\nSign this message to verify wallet ownership.\n\nChallenge: ${challenge}\n\nThis signature does not authorize any transactions.`;

  // Store challenge for later verification
  pendingChallenges.set(userId, { challenge, timestamp });

  return { challenge, message };
}

/**
 * Verify a wallet signature (requires @solana/web3.js or tweetnacl)
 *
 * For production, you'd use:
 * ```
 * import { PublicKey } from "@solana/web3.js";
 * import nacl from "tweetnacl";
 *
 * const publicKey = new PublicKey(walletAddress);
 * const messageBytes = new TextEncoder().encode(message);
 * const signatureBytes = Buffer.from(signature, "base64");
 * const isValid = nacl.sign.detached.verify(messageBytes, signatureBytes, publicKey.toBytes());
 * ```
 */
export async function verifySignature(
  walletAddress: string,
  message: string,
  signature: string
): Promise<boolean> {
  // NOTE: Signature verification requires crypto libraries
  // This is a placeholder - implement with tweetnacl or @solana/web3.js
  //
  // For now, return false to indicate unverified
  // When implementing, uncomment and use the code pattern above

  console.warn("[TokenGate] Signature verification not implemented - requires tweetnacl");
  console.warn("[TokenGate] To implement, add 'tweetnacl' to dependencies and verify signature");

  // Placeholder: Always return false until properly implemented
  // return false;

  // For development/testing, you might temporarily return true
  // WARNING: Remove this in production!
  return false;
}

/**
 * Verify wallet ownership and cache the result
 */
export async function verifyWalletOwnership(
  userId: string,
  walletAddress: string,
  message: string,
  signature: string
): Promise<{ verified: boolean; error?: string }> {
  // Check if challenge exists and hasn't expired
  const pending = pendingChallenges.get(userId);
  if (!pending) {
    return { verified: false, error: "No pending challenge. Request a new one." };
  }

  if (Date.now() - pending.timestamp > CHALLENGE_EXPIRY_MS) {
    pendingChallenges.delete(userId);
    return { verified: false, error: "Challenge expired. Request a new one." };
  }

  // Verify the message contains our challenge
  if (!message.includes(pending.challenge)) {
    return { verified: false, error: "Message does not match challenge." };
  }

  // Verify signature
  const isValid = await verifySignature(walletAddress, message, signature);
  if (!isValid) {
    return { verified: false, error: "Invalid signature." };
  }

  // Cache verified wallet for this user
  verifiedWallets.set(userId, { wallet: walletAddress, timestamp: Date.now() });
  pendingChallenges.delete(userId);

  return { verified: true };
}

/**
 * Get verified wallet for a user (if any)
 */
export function getVerifiedWallet(userId: string): string | null {
  const verified = verifiedWallets.get(userId);
  if (!verified) return null;

  // Check if verification is still valid (24 hour expiry)
  if (Date.now() - verified.timestamp > 24 * 60 * 60_000) {
    verifiedWallets.delete(userId);
    return null;
  }

  return verified.wallet;
}

// ==============================================================================
// TOKEN GATE SERVICE
// ==============================================================================

export class TokenGateService {
  private enabled: boolean;
  private minBalance: number;
  private rpcUrl: string;

  constructor(runtime: IAgentRuntime) {
    const enabled = runtime.getSetting("SOLSPEAK_TOKEN_GATE");
    const minBal = runtime.getSetting("SOLSPEAK_TOKEN_MIN_BALANCE");
    const rpc = runtime.getSetting("SOLSPEAK_SOLANA_RPC");

    this.enabled = enabled === "true" || enabled === true;
    this.minBalance = parseFloat(typeof minBal === "string" ? minBal : String(DEFAULT_MIN_BALANCE));
    this.rpcUrl = typeof rpc === "string" ? rpc : DEFAULT_RPC;
  }

  /**
   * Get the hardcoded SolSpeak token mint address
   */
  getTokenMint(): string {
    return SOLSPEAK_TOKEN_MINT;
  }

  /**
   * Get minimum balance required
   */
  getMinBalance(): number {
    return this.minBalance;
  }

  /**
   * Check if token gating is enabled
   */
  isEnabled(): boolean {
    return this.enabled && SOLSPEAK_TOKEN_MINT !== "PLACEHOLDER_SOLSPEAK_TOKEN_MINT_ADDRESS";
  }

  /**
   * Get cached balance if still valid
   */
  private getCachedBalance(wallet: string): number | null {
    const cached = balanceCache.get(wallet);
    if (cached && Date.now() - cached.timestamp < CACHE_DURATION_MS) {
      return cached.balance;
    }
    return null;
  }

  /**
   * Cache a balance result
   */
  private setCachedBalance(wallet: string, balance: number): void {
    balanceCache.set(wallet, { balance, timestamp: Date.now() });
  }

  /**
   * Fetch token balance from Solana RPC
   */
  async getTokenBalance(walletAddress: string): Promise<number> {
    // Check cache first
    const cached = this.getCachedBalance(walletAddress);
    if (cached !== null) {
      return cached;
    }

    try {
      const response = await fetch(this.rpcUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 1,
          method: "getTokenAccountsByOwner",
          params: [
            walletAddress,
            { mint: SOLSPEAK_TOKEN_MINT },
            { encoding: "jsonParsed" },
          ],
        }),
      });

      const data = await response.json();

      if (data.error) {
        console.error("[TokenGate] RPC error:", data.error);
        return 0;
      }

      const accounts = data.result?.value || [];
      let totalBalance = 0;

      for (const account of accounts) {
        try {
          const amount = account.account?.data?.parsed?.info?.tokenAmount?.uiAmount || 0;
          totalBalance += amount;
        } catch {
          continue;
        }
      }

      this.setCachedBalance(walletAddress, totalBalance);
      return totalBalance;

    } catch (error) {
      console.error("[TokenGate] Error fetching balance:", error);
      return 0;
    }
  }

  /**
   * Check if wallet holds required token amount
   * NOTE: This only checks balance - use checkVerifiedAccess for full auth
   */
  async checkAccess(walletAddress: string | undefined): Promise<{
    allowed: boolean;
    balance?: number;
    required?: number;
    message?: string;
  }> {
    if (!this.isEnabled()) {
      return { allowed: true };
    }

    if (!walletAddress) {
      return {
        allowed: false,
        required: this.minBalance,
        message: "Wallet address required for token-gated access",
      };
    }

    if (!/^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(walletAddress)) {
      return {
        allowed: false,
        message: "Invalid Solana wallet address",
      };
    }

    const balance = await this.getTokenBalance(walletAddress);

    if (balance >= this.minBalance) {
      return {
        allowed: true,
        balance,
        required: this.minBalance,
      };
    }

    return {
      allowed: false,
      balance,
      required: this.minBalance,
      message: `Insufficient SolSpeak tokens. Have ${balance.toLocaleString()}, need ${this.minBalance.toLocaleString()}`,
    };
  }

  /**
   * Full verification: Check verified wallet + balance
   * Use this in production for proper security
   */
  async checkVerifiedAccess(userId: string): Promise<{
    allowed: boolean;
    wallet?: string;
    balance?: number;
    required?: number;
    message?: string;
  }> {
    if (!this.isEnabled()) {
      return { allowed: true };
    }

    // Get verified wallet for this user
    const wallet = getVerifiedWallet(userId);
    if (!wallet) {
      return {
        allowed: false,
        required: this.minBalance,
        message: "Wallet not verified. Please connect and sign to verify ownership.",
      };
    }

    // Check token balance
    const accessCheck = await this.checkAccess(wallet);
    return {
      ...accessCheck,
      wallet,
    };
  }

  /**
   * Clear the balance cache
   */
  clearCache(): void {
    balanceCache.clear();
  }
}

// ==============================================================================
// CONVENIENCE FUNCTIONS
// ==============================================================================

/**
 * Quick check function for use in actions (balance only, no sig verification)
 */
export async function checkTokenAccess(
  runtime: IAgentRuntime,
  walletAddress?: string
): Promise<{ allowed: boolean; message?: string }> {
  const gate = new TokenGateService(runtime);
  return gate.checkAccess(walletAddress);
}

/**
 * Full verified access check (signature + balance)
 */
export async function checkVerifiedTokenAccess(
  runtime: IAgentRuntime,
  userId: string
): Promise<{ allowed: boolean; wallet?: string; message?: string }> {
  const gate = new TokenGateService(runtime);
  return gate.checkVerifiedAccess(userId);
}

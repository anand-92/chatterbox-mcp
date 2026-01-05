/**
 * Test RPC token balance checking
 */

const WALLET = "BVFMRXNQo24SJUH1yMwaUFLp4R2fCPUFCZPnTV6ma8Nt";
const RPC = "https://api.mainnet-beta.solana.com";

// Test with a real token (USDC for example)
const USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";

async function testRPC() {
  console.log("Testing Solana RPC...\n");

  // 1. Test SOL balance
  console.log("1. SOL Balance:");
  const solRes = await fetch(RPC, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "getBalance",
      params: [WALLET],
    }),
  });
  const solData = await solRes.json();
  const solBalance = solData.result?.value / 1e9;
  console.log(`   ${WALLET.slice(0, 8)}... has ${solBalance.toFixed(4)} SOL\n`);

  // 2. Test token accounts (all tokens)
  console.log("2. All Token Accounts:");
  const tokenRes = await fetch(RPC, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "getTokenAccountsByOwner",
      params: [
        WALLET,
        { programId: "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA" },
        { encoding: "jsonParsed" },
      ],
    }),
  });
  const tokenData = await tokenRes.json();
  const accounts = tokenData.result?.value || [];
  console.log(`   Found ${accounts.length} token account(s)\n`);

  // 3. Test specific token (USDC)
  console.log("3. Specific Token (USDC) Balance:");
  const usdcRes = await fetch(RPC, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "getTokenAccountsByOwner",
      params: [
        WALLET,
        { mint: USDC_MINT },
        { encoding: "jsonParsed" },
      ],
    }),
  });
  const usdcData = await usdcRes.json();
  const usdcAccounts = usdcData.result?.value || [];

  if (usdcAccounts.length > 0) {
    const balance = usdcAccounts[0].account?.data?.parsed?.info?.tokenAmount?.uiAmount || 0;
    console.log(`   USDC balance: ${balance}`);
  } else {
    console.log("   No USDC found (expected for this wallet)");
  }

  console.log("\n✅ RPC is working correctly!");
}

testRPC().catch(console.error);

# SolSpeak ElizaOS Plugin - TODO

## Before Production

- [ ] Set `SOLSPEAK_TOKEN_MINT` to actual token address in `src/services/token-gate.ts`
- [ ] Add `tweetnacl` dependency for wallet signature verification
- [ ] Uncomment token gate check block in `src/actions/speak.ts` (lines 74-92)

## Distribution

- [ ] Publish to npm (`npm publish --access public`)
- [ ] Submit to [elizaos-plugins registry](https://github.com/elizaos-plugins)

## Testing

- [ ] Test with live ElizaOS agent
- [ ] Verify token gating works with real SolSpeak token
- [ ] Test wallet signature verification flow

## Optional Enhancements

- [ ] Add LIST_VOICES action
- [ ] Add streaming support for long responses
- [ ] Add voice upload action (clone via agent commands)

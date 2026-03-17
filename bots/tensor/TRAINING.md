1. Main process: Has torch + GPU model. Creates pipe pairs per game.
2. Game workers (spawn): Run engine with proxy bot. No torch needed.
3. Proxy bot: Sends obs via pipe, blocks on pipe read for actions.
4. Main process inference loop: Reads obs from all pipes, batches on GPU, writes actions back.